from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .dataset import CelebADataset, make_default_transform, maybe_download_kaggle_dataset
    from .model import Discriminator, Generator, initialize_dcgan
    from .utils import ensure_dir, plot_losses, save_checkpoint, save_image_grid, save_training_history
except ImportError:  # pragma: no cover - allows running the file directly as a script
    from dataset import CelebADataset, make_default_transform, maybe_download_kaggle_dataset
    from model import Discriminator, Generator, initialize_dcgan
    from utils import ensure_dir, plot_losses, save_checkpoint, save_image_grid, save_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DCGAN on the CelebA dataset.")
    parser.add_argument("--dataset-dir", type=str, default="./data/celeba", help="Directory containing CelebA face images.")
    parser.add_argument("--kaggle-dataset", type=str, default=None, help="Optional Kaggle dataset slug to download when a local dataset is not available. Example: user/dataset-name")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory where training artifacts are saved.")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory where checkpoints are saved.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs to run.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size used for training. Reduce if GPU memory is limited.")
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent noise dimension.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Adam learning rate.")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1 value used in DCGAN training.")
    parser.add_argument("--img-size", type=int, default=64, help="Image size used by the generator and discriminator.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save a periodic checkpoint every N epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    root = Path(__file__).resolve().parent
    output_dir = ensure_dir(root / args.output_dir if not Path(args.output_dir).is_absolute() else args.output_dir)
    checkpoint_dir = ensure_dir(root / args.checkpoint_dir if not Path(args.checkpoint_dir).is_absolute() else args.checkpoint_dir)
    sample_dir = ensure_dir(output_dir / "samples")
    plot_dir = ensure_dir(output_dir / "plots")

    try:
        dataset_root = maybe_download_kaggle_dataset(args.dataset_dir, args.kaggle_dataset)
        dataset = CelebADataset(dataset_root, transform=make_default_transform(args.img_size))
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load CelebA images. Confirm that the dataset is downloaded and the path is correct, or supply a valid --kaggle-dataset with KAGGLE_USERNAME and KAGGLE_KEY configured.\n{exc}"
        ) from exc

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Found {len(dataset)} training images.")

    generator = Generator(latent_dim=args.latent_dim, img_size=args.img_size).to(device)
    discriminator = Discriminator(img_size=args.img_size).to(device)
    initialize_dcgan(generator, discriminator)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)
    history = {"g_loss": [], "d_loss": []}

    for epoch in range(1, args.epochs + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        for real_images in dataloader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1, 1, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, 1, 1, device=device)

            optimizer_d.zero_grad()
            real_logits = discriminator(real_images)
            d_loss_real = criterion(real_logits, real_labels)
            d_loss_real.backward()

            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            with torch.no_grad():
                fake_images = generator(noise)
            fake_logits = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_logits, fake_labels)
            d_loss_fake.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            noise = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images)
            g_loss = criterion(fake_logits, real_labels)
            g_loss.backward()
            optimizer_g.step()

            batch_total_loss = d_loss_real + d_loss_fake
            epoch_d_loss += batch_total_loss.item()
            epoch_g_loss += g_loss.item()
            num_batches += 1

        avg_g_loss = epoch_g_loss / max(1, num_batches)
        avg_d_loss = epoch_d_loss / max(1, num_batches)
        history["g_loss"].append(avg_g_loss)
        history["d_loss"].append(avg_d_loss)

        with torch.no_grad():
            fake_sample = generator(fixed_noise)
        save_image_grid(fake_sample, sample_dir / f"epoch_{epoch:03d}.png", nrow=8)

        print(
            f"Epoch [{epoch}/{args.epochs}] | "
            f"Generator loss: {avg_g_loss:.4f} | D loss: {avg_d_loss:.4f}"
        )

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            config = {
                "latent_dim": args.latent_dim,
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "beta1": args.beta1,
                "epochs": args.epochs,
                "dataset_dir": str(Path(args.dataset_dir).resolve()),
                "output_dir": str(output_dir.resolve()),
            }
            checkpoint = {
                "epoch": epoch,
                "config": config,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "history": history,
            }
            save_checkpoint(checkpoint, checkpoint_dir / f"dcgan_epoch_{epoch:03d}.pth")
            save_training_history(history, output_dir / "training_history.json")

    final_config = {
        "latent_dim": args.latent_dim,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "beta1": args.beta1,
        "epochs": args.epochs,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
    }

    save_checkpoint({"config": final_config, "generator_state_dict": generator.state_dict()}, checkpoint_dir / "generator.pth")
    save_checkpoint({"config": final_config, "discriminator_state_dict": discriminator.state_dict()}, checkpoint_dir / "discriminator.pth")
    save_checkpoint(
        {
            "epoch": args.epochs,
            "config": final_config,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "history": history,
        },
        checkpoint_dir / "dcgan_final_checkpoint.pth",
    )

    save_training_history(history, output_dir / "training_history.json")
    plot_losses(history, plot_dir / "loss_history.png")

    print(f"Finished training. Final artifacts saved in: {output_dir}")
    print(f"Generator checkpoint: {checkpoint_dir / 'generator.pth'}")
    print(f"Full checkpoint: {checkpoint_dir / 'dcgan_final_checkpoint.pth'}")


if __name__ == "__main__":
    main()
