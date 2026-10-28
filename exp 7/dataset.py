from io import BytesIO
from pathlib import Path
from typing import Callable
from zipfile import BadZipFile, ZipFile

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def face_transform(resolution: int = 64) -> transforms.Compose:
    if resolution < 8:
        raise ValueError("resolution must be at least 8 pixels.")
    resize_to = max(resolution, int(round(resolution * 1.12)))
    return transforms.Compose(
        [
            transforms.Resize(resize_to),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class ImageFolderDataset(Dataset[torch.Tensor]):
    def __init__(self, root: Path, transform: Callable) -> None:
        self.files = sorted(
            path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.files:
            raise FileNotFoundError(
                f"No supported images were found under {root}. Expected JPG, PNG, WEBP, or BMP files."
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB"))


class ZipImageDataset(Dataset[torch.Tensor]):
    def __init__(self, archive_path: Path, transform: Callable) -> None:
        self.archive_path = archive_path
        try:
            with ZipFile(archive_path) as archive:
                self.members = sorted(
                    name
                    for name in archive.namelist()
                    if Path(name).suffix.lower() in IMAGE_SUFFIXES and not name.endswith("/")
                )
        except (BadZipFile, OSError) as exc:
            raise ValueError(f"Could not read image ZIP archive: {archive_path}") from exc
        if not self.members:
            raise FileNotFoundError(
                f"No supported images were found in {archive_path}. "
                "The ZIP must contain JPG, PNG, WEBP, or BMP files."
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.members)

    def __getitem__(self, index: int) -> torch.Tensor:
        try:
            with ZipFile(self.archive_path) as archive:
                data = archive.read(self.members[index])
            with Image.open(BytesIO(data)) as image:
                return self.transform(image.convert("RGB"))
        except (BadZipFile, KeyError, OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(
                f"Could not decode image {self.members[index]!r} from {self.archive_path}."
            ) from exc


def load_dataset(
    dataset_path: str | Path,
    resolution: int = 64,
    download_celeba: bool = False,
) -> Dataset[torch.Tensor]:
    """Load a directory, image ZIP, or torchvision CelebA dataset."""
    path = Path(dataset_path).expanduser()
    transform = face_transform(resolution)
    if path.suffix.lower() == ".zip":
        if not path.is_file():
            raise FileNotFoundError(f"Dataset ZIP file does not exist: {path}")
        return ZipImageDataset(path, transform)
    if path.is_dir():
        image_count = sum(
            1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
        if image_count:
            return ImageFolderDataset(path, transform)
    elif download_celeba:
        path.mkdir(parents=True, exist_ok=True)
    else:
        raise FileNotFoundError(
            f"Dataset path does not exist: {path}. For CelebA, create the root or enable download; "
            "for custom images, provide an existing directory or ZIP."
        )
    if path.is_dir():
        try:
            return datasets.CelebA(
                root=str(path),
                split="all",
                target_type="attr",
                download=download_celeba,
                transform=transform,
            )
        except (RuntimeError, OSError) as exc:
            action = "Set download_celeba=True to download CelebA." if not download_celeba else (
                "Check that the CelebA archive is accessible and retry."
            )
            raise RuntimeError(
                f"Could not load CelebA from {path}. Place the extracted dataset there or enable download. {action}"
            ) from exc
    raise FileNotFoundError(f"Dataset path is not a readable directory: {path}")


def make_dataloader(
    dataset_path: str | Path,
    batch_size: int = 16,
    resolution: int = 64,
    num_workers: int = 2,
    download_celeba: bool = False,
    seed: int = 42,
) -> DataLoader:
    dataset = load_dataset(dataset_path, resolution, download_celeba)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(dataset) >= batch_size,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
