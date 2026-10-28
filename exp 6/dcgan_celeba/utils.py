from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid


TensorLike = Union[torch.Tensor, Iterable[torch.Tensor]]


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """Convert [-1, 1] normalized tensors back to [0, 1]."""
    return ((images.detach().cpu().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def save_image_grid(images: torch.Tensor, output_path: str | Path, nrow: int = 8) -> Path:
    output = Path(output_path).expanduser().resolve()
    ensure_dir(output.parent)
    grid = make_grid(denormalize_images(images), nrow=nrow, value_range=(0, 1), pad_value=1.0)
    ndarr = grid.permute(1, 2, 0).numpy()
    image = Image.fromarray((ndarr * 255).astype(np.uint8))
    image.save(output)
    return output


def save_tensor_image(image: torch.Tensor, output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    ensure_dir(output.parent)
    image = denormalize_images(image)
    arr = image.permute(1, 2, 0).numpy()
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.save(output)
    return output


def save_training_history(history: Dict[str, list], output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    ensure_dir(output.parent)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)
    return output


def plot_losses(history: Dict[str, list], output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    ensure_dir(output.parent)

    plt.figure(figsize=(10, 5))
    plt.plot(history.get("g_loss", []), label="Generator Loss", color="royalblue")
    plt.plot(history.get("d_loss", []), label="Discriminator Loss", color="crimson")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DCGAN training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def save_checkpoint(checkpoint: Dict[str, object], output_path: str | Path) -> Path:
    output = Path(output_path).expanduser().resolve()
    ensure_dir(output.parent)
    torch.save(checkpoint, output)
    return output


def load_checkpoint(checkpoint_path: str | Path, device: torch.device | str = "cpu") -> Dict[str, object]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


if __name__ == "__main__":
    image = torch.rand(3, 64, 64) * 2.0 - 1.0
    save_tensor_image(image, "./outputs/test_image.png")
