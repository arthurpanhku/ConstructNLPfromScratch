from transformers import PretrainedConfig


class LMConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        vocab_size: int = 6400,
        hidden_dim: int = None,
        multiple_of: int = 64,
        norm_eps: float = 1e-5,
        model_max_length: int = 8192,
        rope_theta: int = 1e6,
        dropout: float = 0.0,
        **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.model_max_length = model_max_length
        self.rope_theta = rope_theta
        self.dropout = dropout
        super().__init__(**kwargs)
