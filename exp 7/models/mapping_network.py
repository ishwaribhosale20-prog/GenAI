import torch
from torch import nn


class MappingNetwork(nn.Module):
    """Map latent vectors z in Z space to more disentangled vectors w in W space."""

    def __init__(self, latent_dim: int = 128, style_dim: int = 128, depth: int = 4) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Mapping network depth must be at least 1.")
        layers: list[nn.Module] = []
        input_dim = latent_dim
        for _ in range(depth):
            layers.extend([nn.Linear(input_dim, style_dim), nn.LeakyReLU(0.2, inplace=True)])
            input_dim = style_dim
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"Expected z with shape [batch, latent_dim], got {tuple(z.shape)}.")
        z = z * torch.rsqrt(z.square().mean(dim=1, keepdim=True) + 1e-8)
        return self.network(z)
