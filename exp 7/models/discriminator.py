import torch
from torch import nn


class Discriminator(nn.Module):
    """A compact CNN discriminator that returns one real/fake logit per image."""

    def __init__(self, channel_base: int = 32, resolution: int = 64) -> None:
        super().__init__()
        if resolution != 64:
            raise ValueError("Discriminator currently supports 64x64 images only.")
        channels = {
            64: channel_base,
            32: channel_base * 2,
            16: channel_base * 4,
            8: channel_base * 8,
            4: channel_base * 8,
        }
        layers: list[nn.Module] = []
        in_channels = 3
        for size in (64, 32, 16, 8):
            out_channels = channels[size]
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.AvgPool2d(2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[8] * 4 * 4, channels[8]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels[8], 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 64, 64):
            raise ValueError(f"Expected images with shape [batch, 3, 64, 64], got {tuple(images.shape)}.")
        return self.classifier(self.features(images)).squeeze(1)
