from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms


def maybe_download_kaggle_dataset(dataset_dir: str | Path, kaggle_dataset: str | None = None) -> Path:
    """Download a dataset from Kaggle only when the user explicitly opts in and credentials are present."""
    if not kaggle_dataset:
        return Path(dataset_dir).expanduser().resolve()

    root = Path(dataset_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle
    except ImportError as exc:
        raise RuntimeError(
            "The optional Kaggle download feature requires the 'kaggle' package. "
            "Install it with 'pip install kaggle' and configure KAGGLE_USERNAME and KAGGLE_KEY."
        ) from exc

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        raise RuntimeError(
            "Kaggle download requested but KAGGLE_USERNAME and KAGGLE_KEY are not set in the environment."
        )

    api = kaggle.api
    api.authenticate()
    api.dataset_download_files(kaggle_dataset, path=str(root), unzip=True)

    extracted = None
    for candidate in sorted(root.iterdir()):
        if candidate.is_dir() and any(file.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS for file in candidate.rglob("*")):
            extracted = candidate
            break
    if extracted is not None:
        return extracted

    zip_files = sorted(root.glob("*.zip"))
    for archive in zip_files:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
        archive.unlink()

    for candidate in sorted(root.iterdir()):
        if candidate.is_dir() and any(file.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS for file in candidate.rglob("*")):
            return candidate

    raise RuntimeError(
        f"Kaggle dataset '{kaggle_dataset}' was downloaded but no valid image files were found in '{root}'."
    )


DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_dataset_directory(dataset_dir: str | Path) -> Path:
    """Find a valid CelebA image directory while giving clear errors to beginners."""
    root = Path(dataset_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}. "
            "Download CelebA or point --dataset-dir to the folder containing the aligned face images."
        )

    candidates = [root]
    for subfolder in ["img_align_celeba", "aligned", "CelebA", "images", "celeba"]:
        candidates.append(root / subfolder)

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            files = list(candidate.rglob("*"))
            if any(p.is_file() and p.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS for p in files):
                return candidate

    if root.is_dir():
        raise ValueError(
            f"No valid image files were found under {root}. "
            "Expected a CelebA-aligned folder such as 'img_align_celeba' with JPG/PNG images."
        )

    raise ValueError(f"The dataset path '{root}' is not a valid directory.")


def make_default_transform(image_size: int = 64):
    """Apply the standard preprocessing used for CelebA face images."""
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


class CelebADataset(Dataset):
    """A lightweight CelebA dataset loader for aligned face images stored on disk."""

    def __init__(self, dataset_dir: str | Path, transform=None, extensions: Optional[Iterable[str]] = None):
        self.dataset_dir = resolve_dataset_directory(dataset_dir)
        self.transform = transform or make_default_transform()
        self.extensions = {ext.lower() for ext in (extensions or DEFAULT_IMAGE_EXTENSIONS)}
        self.samples = self._load_samples()

        if not self.samples:
            raise ValueError(
                f"No valid image files were found in '{self.dataset_dir}'. "
                "Check that the dataset directory contains aligned face images and that the path is correct."
            )

    def _load_samples(self) -> List[Path]:
        image_files: List[Path] = []
        for path in self.dataset_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.extensions:
                try:
                    with Image.open(path) as img:
                        img.verify()
                    image_files.append(path)
                except (UnidentifiedImageError, OSError, ValueError):
                    continue
        return sorted(image_files)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path = self.samples[index]
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
                return image
        except (UnidentifiedImageError, OSError, ValueError):
            raise RuntimeError(f"Failed to read image at '{image_path}'. The file may be corrupt or missing.")


if __name__ == "__main__":
    sample_dir = "./data/celeba"
    try:
        dataset = CelebADataset(sample_dir)
        print(f"Loaded {len(dataset)} images from {dataset.dataset_dir}")
        sample = dataset[0]
        print("Sample tensor shape:", tuple(sample.shape))
    except Exception as exc:
        print(str(exc))
