from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class TrainConfig:
    resolution: int = 64
    latent_dim: int = 128
    style_dim: int = 128
    mapping_depth: int = 4
    channel_base: int = 32
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 0.0002
    beta1: float = 0.0
    beta2: float = 0.99
    num_workers: int = 2
    seed: int = 42
    sample_every: int = 1
    checkpoint_every: int = 1
    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        if self.resolution != 64:
            raise ValueError("This educational model currently supports resolution=64.")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive.")
        if self.num_workers < 0 or self.sample_every < 1 or self.checkpoint_every < 1:
            raise ValueError("num_workers must be non-negative and save intervals must be positive.")
        if self.channel_base < 1 or self.latent_dim < 1 or self.style_dim < 1:
            raise ValueError("Model dimensions must be positive.")
        self.output_dir = Path(self.output_dir)

    @property
    def num_ws(self) -> int:
        # One styled 4x4 convolution and two styled convolutions at each later scale.
        return 1 + 2 * (self.resolution.bit_length() - 3)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainConfig":
        return cls(**values)
