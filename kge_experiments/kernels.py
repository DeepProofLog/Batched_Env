"""Custom fused kernels for performance optimization."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedLinearReluLayerNorm(nn.Module):
    """Fused Linear + ReLU + LayerNorm operation.

    This replaces the pattern:
        nn.Sequential(nn.Linear(...), nn.ReLU(), nn.LayerNorm(...))

    By implementing it as a single module, torch.compile can better optimize
    the kernel fusion, reducing:
    - Kernel launch overhead
    - Memory bandwidth (fewer intermediate tensors)
    - Overall latency
    """
    def __init__(self, in_features: int, out_features: int, eps: float = 1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

        # Linear layer parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # LayerNorm parameters
        self.ln_weight = nn.Parameter(torch.empty(out_features))
        self.ln_bias = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming initialization for Linear layer (suitable for ReLU)
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.bias)
        # Standard LayerNorm initialization
        nn.init.ones_(self.ln_weight)
        nn.init.zeros_(self.ln_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute: LayerNorm(ReLU(Linear(x)))

        torch.compile will fuse these operations into optimized kernels.
        """
        # Linear: x @ weight.T + bias
        x = F.linear(x, self.weight, self.bias)
        # ReLU
        x = F.relu(x, inplace=True)
        # LayerNorm
        x = F.layer_norm(x, (self.out_features,), self.ln_weight, self.ln_bias, self.eps)
        return x


class FusedLinearRelu(nn.Module):
    """Fused Linear + ReLU operation for cases without LayerNorm."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute: ReLU(Linear(x))"""
        x = F.linear(x, self.weight, self.bias)
        x = F.relu(x, inplace=True)
        return x
