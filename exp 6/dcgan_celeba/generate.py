from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from .model import Generator
    from .utils import ensure_dir, save_image_grid, save_tensor_image
except ImportError:  # pragma: no cover - allows running the file directly as a script
    from model import Generator
    from utils import ensure_dir, save_image_grid, save_tensor_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic face images using the trained DCGAN generator.")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/dcgan_final_checkpoint.pth", help="Path to the generator checkpoint.")
    parser.add_argument("--output-dir", type=str, default="./outputs/generated", help="Directory to write generated images.")
    parser.add_argument("--count", type=int, default=1, help="How many images to generate.")
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent dimension used by the generator.")
    parser.add_argument("--seed", type=int, default=None, help="Optional fixed seed for reproducible generation.")
    return parser.parse_args()


def load_generator(checkpoint_path: str | Path, latent_dim: int, device: torch.device):
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_file}. "
            "Train the model first or point --checkpoint to a valid file."
        )

    checkpoint = torch.load(checkpoint_file, map_location=device)
    if "generator_state_dict" in checkpoint:
        config = checkpoint.get("config", {})
        latent_dim = config.get("latent_dim", latent_dim)
        generator = Generator(latent_dim=latent_dim, img_size=64)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        return generator

    if "state_dict" in checkpoint:
        generator = Generator(latent_dim=latent_dim, img_size=64)
        generator.load_state_dict(checkpoint["state_dict"])
        return generator

    raise ValueError(f"Checkpoint '{checkpoint_file}' does not contain a valid generator state dict.")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    output_dir = ensure_dir(args.output_dir)
    generator = load_generator(args.checkpoint, args.latent_dim, device).to(device)
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(args.count, args.latent_dim, 1, 1, device=device)
        generated = generator(noise)

    if args.count == 1:
        save_tensor_image(generated[0], output_dir / "generated_face.png")
        print(f"Saved generated image to {output_dir / 'generated_face.png'}")
    else:
        save_image_grid(generated, output_dir / "generated_faces_grid.png", nrow=8)
        for idx, image in enumerate(generated):
            save_tensor_image(image, output_dir / f"generated_face_{idx:03d}.png")
        print(f"Saved {args.count} generated images to {output_dir}")


if __name__ == "__main__":
    main()
