from __future__ import annotations

import time
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request

try:
    from .model import Generator
    from .utils import save_tensor_image
except ImportError:  # pragma: no cover - allows direct execution in the same folder
    from model import Generator
    from utils import save_tensor_image

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATHS = [
    BASE_DIR / "checkpoints" / "dcgan_final_checkpoint.pth",
    BASE_DIR / "checkpoints" / "generator.pth",
]

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

generator = None
model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_generator():
    global generator
    if generator is not None:
        return generator

    checkpoint_path = next((path for path in CHECKPOINT_PATHS if path.exists()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No trained DCGAN checkpoint is available. Train the model with 'python train.py' first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=model_device)
    config = checkpoint.get("config", {})
    latent_dim = config.get("latent_dim", 100)
    generator_model = Generator(latent_dim=latent_dim, img_size=64)

    if "generator_state_dict" in checkpoint:
        generator_model.load_state_dict(checkpoint["generator_state_dict"])
    elif "state_dict" in checkpoint:
        generator_model.load_state_dict(checkpoint["state_dict"])
    else:
        raise ValueError(f"Checkpoint '{checkpoint_path}' does not contain a valid Generator state dict.")

    generator_model.to(model_device)
    generator_model.eval()
    generator = generator_model
    return generator


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        payload = request.get_json(silent=True) or {}
        count = int(payload.get("count", 1))
        count = max(1, min(count, 12))
        model = load_generator()
        with torch.no_grad():
            noise = torch.randn(count, 100, 1, 1, device=model_device)
            output = model(noise)

        image_paths = []
        for index, image in enumerate(output):
            filename = f"generated_{time.time_ns()}_{index}.png"
            target = BASE_DIR / "static" / "generated" / filename
            save_tensor_image(image, target)
            image_paths.append(f"/static/generated/{filename}")

        return jsonify({"images": image_paths})
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        return jsonify({"error": f"Unable to generate a face: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
