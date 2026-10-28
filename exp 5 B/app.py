import os
import base64
from io import BytesIO

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, jsonify, request
from PIL import Image


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "models"
)

DECODER_PATH = os.path.join(
    MODEL_DIR,
    "decoder.keras"
)

LATENT_POINTS_PATH = os.path.join(
    MODEL_DIR,
    "latent_points.npy"
)


# ============================================================
# LOAD DECODER
# ============================================================

print("Loading decoder...")

decoder = tf.keras.models.load_model(
    DECODER_PATH,
    compile=False
)

print("Decoder loaded successfully!")


# ============================================================
# LOAD LATENT POINTS
# ============================================================

print("Loading latent points...")

latent_data = np.load(
    LATENT_POINTS_PATH
)

# Columns:
# 0 → Z1
# 1 → Z2
# 2 → digit label

z1_values = latent_data[:, 0]
z2_values = latent_data[:, 1]
labels = latent_data[:, 2].astype(int)


print(
    "Latent points loaded:",
    latent_data.shape
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def generate_image(z1, z2):
    """
    Generate an MNIST image from a 2D latent vector.
    """

    # Create latent vector
    latent_vector = np.array(
        [[z1, z2]],
        dtype=np.float32
    )

    # Generate image using decoder
    generated = decoder.predict(
        latent_vector,
        verbose=0
    )

    # Extract 28x28 image
    image = generated[0, :, :, 0]

    # Convert 0-1 → 0-255
    image = (
        np.clip(image, 0, 1) * 255
    ).astype(np.uint8)

    # Convert NumPy array → PIL image
    pil_image = Image.fromarray(
        image,
        mode="L"
    )

    # Resize for better browser display
    pil_image = pil_image.resize(
        (280, 280),
        Image.Resampling.NEAREST
    )

    # Save image in memory
    buffer = BytesIO()

    pil_image.save(
        buffer,
        format="PNG"
    )

    # Convert image to Base64
    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return image_base64


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# GET LATENT POINTS
# ============================================================

@app.route("/latent-points")
def latent_points():

    # Send all 70,000 points to browser

    points = []

    for i in range(len(latent_data)):

        points.append({
            "z1": float(z1_values[i]),
            "z2": float(z2_values[i]),
            "label": int(labels[i])
        })

    return jsonify(points)


# ============================================================
# GENERATE IMAGE
# ============================================================

@app.route(
    "/generate",
    methods=["POST"]
)
def generate():

    try:

        data = request.get_json()

        z1 = float(
            data["z1"]
        )

        z2 = float(
            data["z2"]
        )

        image = generate_image(
            z1,
            z2
        )

        return jsonify({
            "success": True,
            "image": image,
            "z1": z1,
            "z2": z2
        })

    except Exception as e:

        print(
            "Generation error:",
            e
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("MNIST VAE LATENT SPACE EXPLORER")
    print("=" * 60)

    print(
        "\nOpen this URL in your browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()

    app.run(
        debug=True
    )