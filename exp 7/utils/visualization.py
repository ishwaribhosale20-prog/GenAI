from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert one [-1, 1] CHW tensor to an RGB PIL image."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected one image tensor with shape [3, height, width].")
    array = image.detach().cpu().clamp(-1, 1).add(1).div(2)
    array = array.mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def save_image_grid(
    images: torch.Tensor,
    path: str | Path,
    nrow: int = 4,
    title: str | None = None,
) -> None:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected images with shape [batch, 3, height, width].")
    count = images.shape[0]
    columns = min(max(1, nrow), count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for index, (axis, image) in enumerate(zip(axes.flat, images)):
        axis.imshow(np.asarray(tensor_to_pil(image)))
        axis.set_title(str(index + 1), fontsize=8)
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=120)
    plt.close(figure)
