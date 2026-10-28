from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """DCGAN generator model that maps latent noise -> 64x64 RGB image."""

    def __init__(self, latent_dim: int = 100, ngf: int = 64, img_channels: int = 3, img_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.ngf = ngf
        self.img_channels = img_channels
        self.img_size = img_size

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.main(noise)


class Discriminator(nn.Module):
    """DCGAN discriminator that classifies images as real or fake."""

    def __init__(self, ndf: int = 64, img_channels: int = 3, img_size: int = 64):
        super().__init__()
        self.ndf = ndf
        self.img_channels = img_channels
        self.img_size = img_size

        self.main = nn.Sequential(
            nn.Conv2d(img_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.main(image)


def weights_init(model: nn.Module) -> None:
    """Apply the standard DCGAN weight initialization."""
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0.0)


def initialize_dcgan(generator: nn.Module, discriminator: nn.Module | None = None) -> None:
    """Initialize both generator and discriminator using the DCGAN convention."""
    generator.apply(weights_init)
    if discriminator is not None:
        discriminator.apply(weights_init)


if __name__ == "__main__":
    g = Generator()
    d = Discriminator()
    z = torch.randn(2, 100, 1, 1)
    fake = g(z)
    logits = d(fake)
    print("Generator output shape:", tuple(fake.shape))
    print("Discriminator output shape:", tuple(logits.shape))
