# Conditional VAE MNIST Digit Generator

This project implements a Conditional Variational Autoencoder (CVAE) using TensorFlow/Keras and MNIST.

## What it does

1. Downloads the MNIST handwritten digit dataset.
2. Trains a VAE conditioned on the digit label.
3. Learns a latent representation of handwritten digits.
4. Saves the trained conditional decoder.
5. Provides a Flask web application.
6. When the user selects 0-9, the decoder generates a new handwritten image of that selected digit.

## Important

A normal VAE is not guaranteed to generate a requested digit.

Because the requirement is:

> Select 3 -> generate a new image of 3

the correct model is a **Conditional VAE (CVAE)**. The digit label is supplied to the encoder and decoder during training and generation.

## Run in VS Code

Open this folder in VS Code and open the terminal.

### 1. Create a virtual environment

Windows:

    python -m venv venv
    venv\Scripts\activate

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Train the model

    python train.py

This downloads MNIST and creates:

    models/cvae_decoder.keras

The first training can take some time depending on your CPU/GPU.

### 4. Start Flask

    python app.py

Open:

    http://127.0.0.1:5000/

### 5. Use the application

Select 3 -> click Generate Image.

The application samples a random latent vector and sends:

- latent vector z
- one-hot label for digit 3

to the trained decoder.

The output is a newly generated handwritten-style 3.

## Why your screenshot shows "Model not found"

The Flask application is running correctly, but it is trying to load:

    models/cvae_decoder.keras

That file does not exist at the path expected by the application.

Do NOT hard-code your old computer path such as:

    D:\My Courses\...\cvae_decoder.keras

Instead, keep the model inside this project:

    cvae_mnist_digit_generator/
        models/
            cvae_decoder.keras

Run `python train.py` once to create it.
