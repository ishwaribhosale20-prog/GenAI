import torch
from torch import nn

from models.mapping_network import MappingNetwork


class NoiseInjection(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(
        self,
        x: torch.Tensor,
        noise: torch.Tensor | None = None,
        strength: float = 1.0,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn(
                x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype
            )
        return x + noise * self.weight * strength


class AdaptiveInstanceNorm(nn.Module):
    """Normalize feature maps, then use a W vector to select per-channel scale and bias."""

    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False, eps=1e-5)
        self.affine = nn.Linear(style_dim, channels * 2)
        nn.init.zeros_(self.affine.bias)
        with torch.no_grad():
            self.affine.bias[:channels].fill_(1.0)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        scale, bias = self.affine(w).chunk(2, dim=1)
        return self.norm(x) * scale[:, :, None, None] + bias[:, :, None, None]


class StyledConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, style_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.noise = NoiseInjection(out_channels)
        self.adain = AdaptiveInstanceNorm(out_channels, style_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise_strength: float,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.conv(x)
        x = self.noise(x, noise=noise, strength=noise_strength)
        return self.activation(self.adain(x, w))


def mix_styles(styles_a: torch.Tensor, styles_b: torch.Tensor, cutoff: int) -> torch.Tensor:
    """Use A before cutoff and B from cutoff onwards (W+ tensors, shape [B, layers, dim])."""
    if styles_a.shape != styles_b.shape or styles_a.ndim != 3:
        raise ValueError("Both style tensors must have the same [batch, layers, style_dim] shape.")
    if not 0 <= cutoff <= styles_a.shape[1]:
        raise ValueError(f"cutoff must be between 0 and {styles_a.shape[1]}.")
    return torch.cat((styles_a[:, :cutoff], styles_b[:, cutoff:]), dim=1)


class Generator(nn.Module):
    """Educational 64x64 StyleGAN-like generator with W+, AdaIN, and learned noise."""

    def __init__(
        self,
        latent_dim: int = 128,
        style_dim: int = 128,
        mapping_depth: int = 4,
        channel_base: int = 32,
        resolution: int = 64,
    ) -> None:
        super().__init__()
        if resolution != 64:
            raise ValueError("Generator currently supports 64x64 images only.")
        if channel_base < 1:
            raise ValueError("channel_base must be positive.")
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.resolution = resolution
        self.channels = {
            4: channel_base * 8,
            8: channel_base * 8,
            16: channel_base * 4,
            32: channel_base * 2,
            64: channel_base,
        }
        self.num_ws = 9
        self.mapping = MappingNetwork(latent_dim, style_dim, mapping_depth)
        self.constant = nn.Parameter(torch.randn(1, self.channels[4], 4, 4))
        self.initial = StyledConv(self.channels[4], self.channels[4], style_dim)
        self.blocks = nn.ModuleList()
        self.to_rgbs = nn.ModuleDict()
        self.to_rgbs["4"] = nn.Conv2d(self.channels[4], 3, 1)
        in_channels = self.channels[4]
        for size in (8, 16, 32, 64):
            block = nn.ModuleList(
                [
                    StyledConv(in_channels, self.channels[size], style_dim),
                    StyledConv(self.channels[size], self.channels[size], style_dim),
                ]
            )
            self.blocks.append(block)
            self.to_rgbs[str(size)] = nn.Conv2d(self.channels[size], 3, 1)
            in_channels = self.channels[size]

    def styles_from_z(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping(z)
        return w.unsqueeze(1).repeat(1, self.num_ws, 1)

    def forward(
        self,
        z: torch.Tensor | None = None,
        styles: torch.Tensor | None = None,
        noise_seed: int | None = None,
        noise_strength: float = 1.0,
    ) -> torch.Tensor:
        if (z is None) == (styles is None):
            raise ValueError("Provide exactly one of z or styles.")
        if styles is None:
            assert z is not None
            styles = self.styles_from_z(z)
        if styles.ndim != 3 or styles.shape[1:] != (self.num_ws, self.style_dim):
            raise ValueError(
                f"styles must have shape [batch, {self.num_ws}, {self.style_dim}]."
            )

        noise_generator = (
            torch.Generator(device=styles.device).manual_seed(int(noise_seed))
            if noise_seed is not None
            else None
        )
        x = self.constant.expand(styles.shape[0], -1, -1, -1)
        x = self._apply_styled(
            self.initial, x, styles[:, 0], noise_strength, noise_generator
        )
        rgb = self.to_rgbs["4"](x)
        style_index = 1
        for size, block in zip((8, 16, 32, 64), self.blocks):
            x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            for layer in block:
                x = self._apply_styled(
                    layer, x, styles[:, style_index], noise_strength, noise_generator
                )
                style_index += 1
            rgb = nn.functional.interpolate(rgb, scale_factor=2, mode="nearest")
            rgb = rgb + self.to_rgbs[str(size)](x)
        return torch.tanh(rgb)

    @staticmethod
    def _apply_styled(
        layer: StyledConv,
        x: torch.Tensor,
        w: torch.Tensor,
        noise_strength: float,
        noise_generator: torch.Generator | None,
    ) -> torch.Tensor:
        noise = None
        if noise_generator is not None:
            noise = torch.randn(
                (x.shape[0], 1, x.shape[2], x.shape[3]),
                generator=noise_generator,
                device=x.device,
                dtype=x.dtype,
            )
        return layer(x, w, noise_strength, noise)
