import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

# LMConfig remains unchanged
class LMConfig:
    def __init__(self, dim=512, n_heads=8, n_kv_heads=4, dropout=0.02, model_max_length=1024, 
                 vocab_size=32000, n_layers=12, norm_eps=1e-5, hidden_dim=None, 
                 multiple_of=256, rope_theta=1e6):
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.dropout = dropout
        self.model_max_length = model_max_length
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.norm_eps = norm_eps
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.rope_theta = rope_theta

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to the input tensor x.

        Ref: [RMSNorm](https://arxiv.org/abs/1910.07467)

        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_dim)

        Returns:
            torch.Tensor: Normalized tensor of the same shape as x
        """
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)
    return pos_cis

def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # Initialize weights for numerical stability
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        nn.init.xavier_uniform_(self.wo.weight)

    def forward(
        self,
        x: torch.Tensor,
        pos_cis: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache=False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Implement the forward pass of the attention layer.

        Ref: [Attention is all you need](https://arxiv.org/abs/1706.03762)
        Ref: [GQA](https://arxiv.org/abs/2305.13245)

        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_dim)
            pos_cis: Positional Coding tensor of shape (sequence_length, hidden_dim)
            past_key_value: Optional tuple of tensors containing past keys and values
            use_cache: Whether to use cached past keys and values

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: Output tensor and past key values
        """
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        seq_len_kv = xk.shape[1]
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )

        # Scaled dot-product attention with dynamic masking
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(
            torch.full((seq_len, seq_len_kv), float("-inf"), device=x.device),
            diagonal=1 if past_key_value is None else seq_len_kv - seq_len + 1
        )
        mask = mask.view(1, 1, seq_len, seq_len_kv)
        scores = scores + mask
        scores = F.softmax(scores, dim=-1)
        scores = self.attn_dropout(scores)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))

        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        h = x + h_attn
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv

class MiniMindLM(PreTrainedModel):
    config_class = LMConfig

    def __init__(self, params: LMConfig = None):
        self.params = params or LMConfig()
        super().__init__(self.params)
        self.vocab_size, self.n_layers = self.params.vocab_size, self.params.n_layers
        self.tok_embeddings = nn.Embedding(self.params.vocab_size, self.params.dim)
        self.dropout = nn.Dropout(self.params.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(layer, self.params) for layer in range(self.n_layers)])
        self.norm = RMSNorm(self.params.dim, eps=self.params.norm_eps)
        self.output = nn.Linear(self.params.dim, self.params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        self.register_buffer(
            "pos_cis",
            precompute_pos_cis(dim=self.params.dim // self.params.n_heads, theta=self.params.rope_theta),
            persistent=False,
        )
        self.OUT = CausalLMOutputWithPast()

        # Initialize embedding and output weights
        nn.init.xavier_uniform_(self.tok_embeddings.weight)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        **args,
    ):
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = args.get("start_pos", 0)
        h = self.dropout(self.tok_embeddings(input_ids))
        pos_cis = self.pos_cis[start_pos : start_pos + input_ids.size(1)]
        past_kvs = []
        for layer_idx, layer in enumerate(self.layers):
            h, past_kv = layer(h, pos_cis, past_key_value=past_key_values[layer_idx], use_cache=use_cache)
            past_kvs.append(past_kv)
        logits = self.output(self.norm(h))
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("past_key_values", past_kvs)
        return self.OUT

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                out = self(
                    input_ids[:, -1:],
                    past_key_values=past_kvs,
                    use_cache=use_cache,
                    start_pos=input_ids.shape[1] - 1,
                    **args,
                )
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= temperature + 1e-9
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("Inf")
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        eos_token_id=2,
        max_new_tokens=1024,
        temperature=0.75,
        top_p=0.90,
        rp=1.0,
        use_cache=True,
        pad_token_id=0,
        **args,
    ):
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1,
            )
            for seq in generated
        ]
        return torch.cat(generated, dim=0)

# Example usage
if __name__ == "__main__":
    config = LMConfig()
    model = MiniMindLM(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 10))  # Batch size 1, sequence length 10
    output = model.generate(input_ids)
    print(output.shape)