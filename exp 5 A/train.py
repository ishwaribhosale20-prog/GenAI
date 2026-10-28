import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
LATENT_DIM = 16
NUM_CLASSES = 10
IMG_SHAPE = (28, 28, 1)
EPOCHS = 10
BATCH_SIZE = 128

os.makedirs("models", exist_ok=True)

# -----------------------------
# Load and prepare MNIST
# -----------------------------
(x_train, y_train), _ = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)

y_train_onehot = keras.utils.to_categorical(y_train, NUM_CLASSES).astype("float32")

dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train_onehot)
).shuffle(60000).batch(BATCH_SIZE)

# -----------------------------
# Sampling layer
# z = mean + exp(0.5 * log_var) * epsilon
# -----------------------------
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# -----------------------------
# Encoder
# Inputs: image + digit label
# Outputs: z_mean, z_log_var, z
# -----------------------------
image_input = keras.Input(shape=IMG_SHAPE, name="image")
label_input = keras.Input(shape=(NUM_CLASSES,), name="label")

x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(image_input)
x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Concatenate()([x, label_input])
x = layers.Dense(128, activation="relu")(x)

z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(
    [image_input, label_input],
    [z_mean, z_log_var, z],
    name="encoder"
)

# -----------------------------
# Conditional Decoder
# Inputs: latent vector + digit label
# Output: 28x28 generated image
# -----------------------------
z_input = keras.Input(shape=(LATENT_DIM,), name="z")
decoder_label_input = keras.Input(
    shape=(NUM_CLASSES,), name="label"
)

x = layers.Concatenate()([z_input, decoder_label_input])
x = layers.Dense(7 * 7 * 64, activation="relu")(x)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(
    64, 3, strides=2, padding="same", activation="relu"
)(x)
x = layers.Conv2DTranspose(
    32, 3, strides=2, padding="same", activation="relu"
)(x)
decoder_output = layers.Conv2D(
    1, 3, padding="same", activation="sigmoid", name="generated_image"
)(x)

decoder = keras.Model(
    [z_input, decoder_label_input],
    decoder_output,
    name="decoder"
)

# -----------------------------
# Conditional VAE
# -----------------------------
class CVAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        images, labels = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(
                [images, labels], training=True
            )
            reconstructed = self.decoder(
                [z, labels], training=True
            )

            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(
                        images, reconstructed
                    ),
                    axis=(1, 2)
                )
            )

            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var
                    - tf.square(z_mean)
                    - tf.exp(z_log_var),
                    axis=1
                )
            )

            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights)
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

cvae = CVAE(encoder, decoder)
cvae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

print("\nStarting Conditional VAE training...")
print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Latent dimension: {LATENT_DIM}\n")

cvae.fit(dataset, epochs=EPOCHS)

# Save only the decoder needed by Flask.
decoder.save("models/cvae_decoder.keras")

print("\nTraining completed successfully!")
print("Saved model: models/cvae_decoder.keras")
print("Now run: python app.py")
