import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow import keras

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "cvae_decoder.keras")
LATENT_DIM = 16
NUM_CLASSES = 10

decoder = None

def load_model_once():
    global decoder
    if decoder is None:
        if not os.path.exists(MODEL_PATH):
            return False
        decoder = keras.models.load_model(MODEL_PATH, compile=False)
    return True

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    if not load_model_once():
        return jsonify({
            "error": "Model not found. Run 'python train.py' first."
        }), 500

    try:
        digit = int(request.form.get("digit", 0))

        if digit < 0 or digit > 9:
            return jsonify({"error": "Digit must be between 0 and 9."}), 400

        # Random point in latent space.
        z = np.random.normal(
            0, 1, size=(1, LATENT_DIM)
        ).astype("float32")

        # Condition decoder on the selected digit.
        label = np.zeros(
            (1, NUM_CLASSES), dtype="float32"
        )
        label[0, digit] = 1.0

        generated = decoder.predict(
            [z, label], verbose=0
        )[0, :, :, 0]

        # Convert 0..1 image to PNG.
        import io
        from PIL import Image

        image_array = np.clip(
            generated * 255, 0, 255
        ).astype("uint8")

        image = Image.fromarray(image_array, mode="L")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        # Return a data URL so the page can display it.
        import base64
        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return jsonify({
            "digit": digit,
            "image": "data:image/png;base64," + encoded
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("\nWARNING: Model not found.")
        print("First run: python train.py\n")

    app.run(debug=True)
