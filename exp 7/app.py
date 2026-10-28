import argparse
import json
import os
from pathlib import Path

import torch

from config import TrainConfig
from models import Discriminator, Generator, mix_styles
from utils.visualization import tensor_to_pil


def load_generator(checkpoint_path: str | Path) -> tuple[Generator, dict, torch.device]:
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train a model first with `python train.py --dataset <path>`."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = TrainConfig.from_dict(checkpoint["config"])
        generator = Generator(
            config.latent_dim,
            config.style_dim,
            config.mapping_depth,
            config.channel_base,
            config.resolution,
        )
        discriminator = Discriminator(config.channel_base, config.resolution)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        raise ValueError(f"Could not load a compatible project checkpoint from {path}.") from exc
    generator.to(device).eval()
    return generator, checkpoint, device


def build_demo(generator: Generator, checkpoint: dict, device: torch.device):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install the optional interface dependency with `pip install gradio`.") from exc

    latent_dim = generator.latent_dim

    def random_z(seed: int) -> torch.Tensor:
        return torch.randn(
            1,
            latent_dim,
            generator=torch.Generator(device=device).manual_seed(int(seed)),
            device=device,
        )

    def generate_faces(count: int, seed: int, noise_strength: float):
        z = torch.randn(
            int(count),
            latent_dim,
            generator=torch.Generator(device=device).manual_seed(int(seed)),
            device=device,
        )
        with torch.inference_mode():
            images = generator(z, noise_seed=seed, noise_strength=noise_strength)
        return [tensor_to_pil(image) for image in images]

    def show_mapping(seed: int):
        z = random_z(seed)
        with torch.inference_mode():
            w = generator.mapping(z)
            image = generator(z, noise_seed=seed)
        message = (
            f"Z vector: {tuple(z.shape)} | W vector: {tuple(w.shape)} | "
            f"mean(Z)={z.mean().item():.3f}, std(Z)={z.std().item():.3f} | "
            f"mean(W)={w.mean().item():.3f}, std(W)={w.std().item():.3f}. "
            "The learned mapping changes coordinates; this compact model does not guarantee "
            "perfectly disentangled semantic directions."
        )
        return tensor_to_pil(image[0]), message

    def control_styles(seed: int, coarse: float, middle: float, fine: float):
        z = random_z(seed)
        with torch.inference_mode():
            styles = generator.styles_from_z(z).clone()
            styles[:, :3] *= coarse
            styles[:, 3:6] *= middle
            styles[:, 6:] *= fine
            image = generator(styles=styles, noise_seed=seed)
        return tensor_to_pil(image[0])

    def style_mix(seed_a: int, seed_b: int, cutoff: int):
        z_a, z_b = random_z(seed_a), random_z(seed_b)
        with torch.inference_mode():
            styles_a = generator.styles_from_z(z_a)
            styles_b = generator.styles_from_z(z_b)
            styles_mixed = mix_styles(styles_a, styles_b, int(cutoff))
            images = [
                generator(z=z_a, noise_seed=0),
                generator(z=z_b, noise_seed=0),
                generator(styles=styles_mixed, noise_seed=0),
            ]
        return [(tensor_to_pil(image[0]), label) for image, label in zip(images, ["Identity A", "Identity B", "Mixed"])]

    def compare_noise(seed: int, strength: float):
        z = random_z(seed)
        with torch.inference_mode():
            first = generator(z, noise_seed=int(seed), noise_strength=strength)
            second = generator(z, noise_seed=int(seed) + 1, noise_strength=strength)
        return [
            (tensor_to_pil(first[0]), f"Noise seed {int(seed)}"),
            (tensor_to_pil(second[0]), f"Noise seed {int(seed) + 1}"),
        ]

    def show_training_progress():
        output_dir = Path(checkpoint.get("config", {}).get("output_dir", "outputs"))
        history_path = output_dir / "losses.json"
        sample_paths = sorted((output_dir / "samples").glob("epoch_*.png"))
        if history_path.is_file():
            with history_path.open(encoding="utf-8") as file:
                history = json.load(file)
            g_loss = history.get("generator", [])
            d_loss = history.get("discriminator", [])
            summary = (
                f"Recorded epochs: {len(g_loss)} | Last G loss: "
                f"{g_loss[-1]:.4f}" if g_loss else "No training loss history found."
            )
            if d_loss:
                summary += f" | Last D loss: {d_loss[-1]:.4f}"
        else:
            summary = "No loss history file found next to the checkpoint."
        if not sample_paths:
            summary += " No saved sample grid found."
            return None, summary
        from PIL import Image

        with Image.open(sample_paths[-1]) as sample:
            return sample.copy(), summary

    with gr.Blocks(title="Educational StyleGAN Face Lab") as demo:
        gr.Markdown(
            "# Educational StyleGAN Face Lab\n"
            "This is a compact teaching implementation, not the original NVIDIA StyleGAN."
        )
        with gr.Tab("Generate Faces"):
            count = gr.Slider(1, 8, value=4, step=1, label="Number of images")
            seed = gr.Number(value=42, precision=0, label="Latent seed")
            strength = gr.Slider(0, 2, value=1, step=0.05, label="Noise strength")
            generate_button = gr.Button("Generate")
            gallery = gr.Gallery(label="Generated samples", columns=4, height="auto")
            generate_button.click(generate_faces, [count, seed, strength], gallery)

        with gr.Tab("Latent Space"):
            latent_seed = gr.Number(value=7, precision=0, label="Latent seed")
            latent_button = gr.Button("Map Z to W")
            latent_image = gr.Image(label="Generated from mapped latent", type="pil")
            latent_text = gr.Textbox(label="Mapping information", interactive=False)
            latent_button.click(show_mapping, latent_seed, [latent_image, latent_text])

        with gr.Tab("Coarse-to-Fine Control"):
            style_seed = gr.Number(value=11, precision=0, label="Latent seed")
            coarse = gr.Slider(0, 2, value=1, step=0.05, label="Coarse styles (layers 1-3)")
            middle = gr.Slider(0, 2, value=1, step=0.05, label="Middle styles (layers 4-6)")
            fine = gr.Slider(0, 2, value=1, step=0.05, label="Fine styles (layers 7-9)")
            style_button = gr.Button("Apply layer controls")
            style_image = gr.Image(label="Controlled image", type="pil")
            style_button.click(control_styles, [style_seed, coarse, middle, fine], style_image)
            gr.Markdown(
                "These broad/middle/fine layer groups are approximate depth controls. "
                "Semantic effects are not guaranteed, especially after short training."
            )

        with gr.Tab("Style Mixing"):
            seed_a = gr.Number(value=19, precision=0, label="Identity A seed")
            seed_b = gr.Number(value=29, precision=0, label="Identity B seed")
            cutoff = gr.Slider(1, generator.num_ws - 1, value=5, step=1, label="First B style layer")
            mix_button = gr.Button("Mix identities")
            mix_gallery = gr.Gallery(label="A, B, and mixed output", columns=3, height="auto")
            mix_button.click(style_mix, [seed_a, seed_b, cutoff], mix_gallery)
            gr.Markdown("Earlier layers use A and layers from the selected cutoff onward use B.")

        with gr.Tab("Noise Injection"):
            noise_seed = gr.Number(value=31, precision=0, label="First noise seed")
            noise_level = gr.Slider(0, 2, value=1, step=0.05, label="Noise strength")
            noise_button = gr.Button("Compare noise patterns")
            noise_gallery = gr.Gallery(label="Same latent, different seeded noise", columns=2, height="auto")
            noise_button.click(compare_noise, [noise_seed, noise_level], noise_gallery)

        with gr.Tab("Training Progress"):
            progress_button = gr.Button("Refresh training progress")
            progress_image = gr.Image(label="Latest saved sample grid", type="pil")
            progress_text = gr.Textbox(label="Loss summary", interactive=False)
            progress_button.click(show_training_progress, outputs=[progress_image, progress_text])

        gr.Markdown(
            f"Loaded checkpoint epoch: {checkpoint.get('epoch', 'unknown')} | Device: {device}"
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Gradio StyleGAN demonstration.")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("STYLEGAN_CHECKPOINT"),
        help="Path to outputs/latest.pt (or set STYLEGAN_CHECKPOINT).",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("Provide --checkpoint or set STYLEGAN_CHECKPOINT.")
    generator, checkpoint, device = load_generator(args.checkpoint)
    demo = build_demo(generator, checkpoint, device)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
