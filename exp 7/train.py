import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import TrainConfig
from dataset import make_dataloader
from models import Discriminator, Generator
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.visualization import save_image_grid


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def _make_smoke_dataloader(config: TrainConfig) -> DataLoader:
    random_images = torch.rand(8, 3, config.resolution, config.resolution) * 2 - 1
    return DataLoader(TensorDataset(random_images), batch_size=config.batch_size, shuffle=True)


def _real_images(batch: Any) -> torch.Tensor:
    # torchvision CelebA returns (image, attributes); the folder/ZIP loaders return images.
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def train(
    config: TrainConfig,
    dataset_path: str | Path | None,
    download_celeba: bool = False,
    resume: str | Path | None = None,
    max_steps: int | None = None,
) -> dict[str, list[float]]:
    seed_everything(config.seed)
    if dataset_path is None:
        raise ValueError("Provide --dataset for CelebA/images, or use --smoke-test for synthetic smoke data.")
    if str(dataset_path) == "__synthetic_smoke__":
        dataloader = _make_smoke_dataloader(config)
    else:
        dataloader = make_dataloader(
            dataset_path,
            batch_size=config.batch_size,
            resolution=config.resolution,
            num_workers=config.num_workers,
            download_celeba=download_celeba,
            seed=config.seed,
        )

    device = config.device
    print(f"Device: {device}; batches per epoch: {len(dataloader)}")
    generator = Generator(
        config.latent_dim,
        config.style_dim,
        config.mapping_depth,
        config.channel_base,
        config.resolution,
    ).to(device)
    discriminator = Discriminator(config.channel_base, config.resolution).to(device)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    loss_function = nn.BCEWithLogitsLoss()
    history: dict[str, list[float]] = {"generator": [], "discriminator": []}
    start_epoch = 0
    if resume:
        checkpoint = load_checkpoint(
            resume,
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            map_location=device,
        )
        start_epoch = int(checkpoint["epoch"])
        history = checkpoint.get("history", history)
        print(f"Resuming from epoch {start_epoch}: {resume}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    preview_count = min(16, len(dataloader.dataset))
    preview_images = torch.stack(
        [_real_images(dataloader.dataset[index]) for index in range(preview_count)]
    )
    save_image_grid(
        preview_images,
        config.output_dir / "dataset_samples.png",
        title=f"Preprocessed dataset samples ({config.resolution}x{config.resolution})",
    )
    print(f"Preprocessed dataset preview: {config.output_dir / 'dataset_samples.png'}")
    fixed_z = torch.randn(16, config.latent_dim, device=device)
    step_count = 0
    generator.train()
    discriminator.train()
    for epoch in range(start_epoch, config.epochs):
        generator_losses: list[float] = []
        discriminator_losses: list[float] = []
        for batch in dataloader:
            real = _real_images(batch).to(device, non_blocking=True)
            if real.shape[-2:] != (config.resolution, config.resolution):
                raise ValueError(
                    f"Expected {config.resolution}x{config.resolution} dataset images, got {tuple(real.shape[-2:])}."
                )
            batch_size = real.shape[0]

            try:
                discriminator_optimizer.zero_grad(set_to_none=True)
                z = torch.randn(batch_size, config.latent_dim, device=device)
                fake = generator(z).detach()
                real_logits = discriminator(real)
                fake_logits = discriminator(fake)
                discriminator_loss = loss_function(real_logits, torch.ones_like(real_logits))
                discriminator_loss += loss_function(fake_logits, torch.zeros_like(fake_logits))
                discriminator_loss.backward()
                discriminator_optimizer.step()

                generator_optimizer.zero_grad(set_to_none=True)
                z = torch.randn(batch_size, config.latent_dim, device=device)
                generated = generator(z)
                generated_logits = discriminator(generated)
                generator_loss = loss_function(generated_logits, torch.ones_like(generated_logits))
                generator_loss.backward()
                generator_optimizer.step()
            except torch.cuda.OutOfMemoryError as exc:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA ran out of memory during a training step. Retry with a smaller --batch-size "
                    "(for example 4 or 8) or a smaller model."
                ) from exc

            generator_losses.append(float(generator_loss.detach().cpu()))
            discriminator_losses.append(float(discriminator_loss.detach().cpu()))
            step_count += 1
            if max_steps is not None and step_count >= max_steps:
                break

        if not generator_losses:
            raise RuntimeError("No training batches were produced; check the dataset and batch size.")
        history["generator"].append(float(np.mean(generator_losses)))
        history["discriminator"].append(float(np.mean(discriminator_losses)))
        completed_epoch = epoch + 1
        print(
            f"Epoch {completed_epoch}/{config.epochs} | "
            f"G: {history['generator'][-1]:.4f} | D: {history['discriminator'][-1]:.4f}"
        )

        if completed_epoch % config.sample_every == 0:
            generator.eval()
            with torch.inference_mode():
                samples = generator(fixed_z, noise_seed=config.seed)
            save_image_grid(
                samples,
                config.output_dir / "samples" / f"epoch_{completed_epoch:04d}.png",
                title=f"Generated samples after epoch {completed_epoch}",
            )
            generator.train()
        if completed_epoch % config.checkpoint_every == 0 or completed_epoch == config.epochs:
            save_checkpoint(
                config.output_dir / "latest.pt",
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                completed_epoch,
                history,
                config.to_dict(),
            )
        with (config.output_dir / "losses.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
        if max_steps is not None and step_count >= max_steps:
            break
    return history


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the educational 64x64 StyleGAN-like model on CelebA or face images."
    )
    parser.add_argument("--dataset", type=Path, help="CelebA root, image directory, or image ZIP file.")
    parser.add_argument("--download-celeba", action="store_true", help="Download CelebA into --dataset if needed.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, help="Resume from a saved latest.pt checkpoint.")
    parser.add_argument("--steps", type=int, help="Stop after this many optimizer iterations (useful for a smoke test).")
    parser.add_argument("--smoke-test", action="store_true", help="Run one tiny synthetic-data training epoch.")
    parser.add_argument("--save-to-drive", action="store_true", help="Mount Google Drive and save outputs there.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_path = args.dataset
    if args.save_to_drive:
        try:
            from google.colab import drive
        except ImportError as exc:
            raise RuntimeError("--save-to-drive is available only in Google Colab.") from exc
        drive.mount("/content/drive")
        if args.output_dir == Path("outputs"):
            args.output_dir = Path("/content/drive/MyDrive/stylegan-celeba/outputs")
    if args.smoke_test:
        if args.dataset is not None:
            dataset_path = args.dataset
        else:
            dataset_path = Path("__synthetic_smoke__")
        config = TrainConfig(
            latent_dim=32,
            style_dim=32,
            mapping_depth=2,
            channel_base=4,
            batch_size=2,
            epochs=1,
            num_workers=0,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        steps = args.steps or 1
    else:
        config = TrainConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        steps = args.steps
    train(config, dataset_path, args.download_celeba, args.resume, steps)


if __name__ == "__main__":
    main()
