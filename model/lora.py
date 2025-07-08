import torch
from torch import nn

class LoRA(nn.Module):
    """Low-Rank Adaptation module for fine-tuning neural networks."""
    def __init__(self, in_features: int, out_features: int, rank: int):
        """
        Initialize LoRA module with low-rank matrices.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of the low-rank approximation
        """
        super().__init__()
        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.B = nn.Parameter(torch.zeros(rank, in_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LoRA module."""
        return (x @ self.B.T) @ self.A.T

def apply_lora(model: nn.Module, rank: int = 16):
    """
    Apply LoRA to all linear layers in the model.

    Args:
        model: The model to apply LoRA to
        rank: Rank for the LoRA modules
    """
    device = next(model.parameters()).device

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Apply to all linear layers, not just square ones
        in_features = module.weight.shape[1]  # Input dimension
        out_features = module.weight.shape[0]  # Output dimension
        
        lora = LoRA(in_features, out_features, rank=rank).to(device)
        setattr(module, "lora", lora)

        # Store original forward method
        original_forward = module.forward

        # Create new forward method
        def make_forward_with_lora(orig_forward, lora_module):
            def forward_with_lora(x):
                return orig_forward(x) + lora_module(x)
            return forward_with_lora

        module.forward = make_forward_with_lora(original_forward, lora)
    
    return model