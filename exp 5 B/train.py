import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# SETTINGS
# ============================================================

LATENT_DIM = 2
BATCH_SIZE = 128
EPOCHS = 15

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("MNIST VARIATIONAL AUTOENCODER")
print("=" * 60)

# ============================================================
# 1. LOAD MNIST DATASET
# ============================================================

print("\nLoading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Training images :", x_train.shape)
print("Training labels :", y_train.shape)
print("Test images     :", x_test.shape)
print("Test labels     :", y_test.shape)

# ============================================================
# 2. PREPROCESS IMAGES
# ============================================================

# Convert pixels from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
# 28 x 28 → 28 x 28 x 1
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("\nAfter preprocessing:")
print("x_train:", x_train.shape)
print("x_test :", x_test.shape)

# ============================================================
# 3. SAMPLING LAYER
# ============================================================

class Sampling(layers.Layer):
    """
    Reparameterization trick:

    z = mean + exp(0.5 * log_variance) * epsilon

    This allows the VAE to learn a smooth latent space.
    """

    def call(self, inputs):

        z_mean, z_log_var = inputs

        epsilon = tf.random.normal(
            shape=tf.shape(z_mean)
        )

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ============================================================
# 4. ENCODER
# ============================================================

print("\nBuilding Encoder...")

encoder_inputs = keras.Input(
    shape=(28, 28, 1),
    name="encoder_input"
)

x = layers.Conv2D(
    32,
    kernel_size=3,
    strides=2,
    padding="same",
    activation="relu"
)(encoder_inputs)

x = layers.Conv2D(
    64,
    kernel_size=3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

x = layers.Flatten()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

# Mean of latent distribution
z_mean = layers.Dense(
    LATENT_DIM,
    name="z_mean"
)(x)

# Log variance of latent distribution
z_log_var = layers.Dense(
    LATENT_DIM,
    name="z_log_var"
)(x)

# Sample from latent distribution
z = Sampling()(
    [z_mean, z_log_var]
)

encoder = keras.Model(
    encoder_inputs,
    [z_mean, z_log_var, z],
    name="encoder"
)

encoder.summary()

# ============================================================
# 5. DECODER
# ============================================================

print("\nBuilding Decoder...")

latent_inputs = keras.Input(
    shape=(LATENT_DIM,),
    name="latent_input"
)

x = layers.Dense(
    7 * 7 * 64,
    activation="relu"
)(latent_inputs)

x = layers.Reshape(
    (7, 7, 64)
)(x)

x = layers.Conv2DTranspose(
    64,
    kernel_size=3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

x = layers.Conv2DTranspose(
    32,
    kernel_size=3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

decoder_outputs = layers.Conv2D(
    1,
    kernel_size=3,
    padding="same",
    activation="sigmoid"
)(x)

decoder = keras.Model(
    latent_inputs,
    decoder_outputs,
    name="decoder"
)

decoder.summary()

# ============================================================
# 6. VAE MODEL
# ============================================================

class VAE(keras.Model):

    def __init__(
        self,
        encoder,
        decoder,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(
            name="total_loss"
        )

        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

        self.kl_loss_tracker = keras.metrics.Mean(
            name="kl_loss"
        )

    @property
    def metrics(self):

        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def train_step(self, data):

        # MNIST image
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:

            # Encode image
            z_mean, z_log_var, z = self.encoder(
                data,
                training=True
            )

            # Decode latent vector
            reconstruction = self.decoder(
                z,
                training=True
            )

            # Reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(
                        data,
                        reconstruction
                    ),
                    axis=(1, 2)
                )
            )

            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1
                    + z_log_var
                    - tf.square(z_mean)
                    - tf.exp(z_log_var),
                    axis=1
                )
            )

            # Total VAE loss
            total_loss = (
                reconstruction_loss
                + kl_loss
            )

        # Calculate gradients
        gradients = tape.gradient(
            total_loss,
            self.trainable_weights
        )

        # Update weights
        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.trainable_weights
            )
        )

        # Update metrics
        self.total_loss_tracker.update_state(
            total_loss
        )

        self.reconstruction_loss_tracker.update_state(
            reconstruction_loss
        )

        self.kl_loss_tracker.update_state(
            kl_loss
        )

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss":
                self.reconstruction_loss_tracker.result(),
            "kl_loss":
                self.kl_loss_tracker.result()
        }


# ============================================================
# 7. CREATE VAE
# ============================================================

print("\nCreating VAE...")

vae = VAE(
    encoder,
    decoder
)

vae.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    )
)

# ============================================================
# 8. TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

vae.fit(
    x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ============================================================
# 9. SAVE MODELS
# ============================================================

print("\nSaving models...")

encoder.save(
    os.path.join(
        MODEL_DIR,
        "encoder.keras"
    )
)

decoder.save(
    os.path.join(
        MODEL_DIR,
        "decoder.keras"
    )
)

print("\nModels saved successfully!")

print(
    "\nEncoder : models/encoder.keras"
)

print(
    "Decoder : models/decoder.keras"
)

# ============================================================
# 10. GENERATE LATENT COORDINATES
# ============================================================

print("\nEncoding all MNIST images...")

# We use both training and test data.
all_images = np.concatenate(
    [x_train, x_test],
    axis=0
)

all_labels = np.concatenate(
    [y_train, y_test],
    axis=0
)

# Get mean latent coordinates
z_mean, z_log_var, z = encoder.predict(
    all_images,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Since LATENT_DIM = 2,
# z_mean contains:
#
# column 0 → Z1
# column 1 → Z2

latent_points = z_mean

print(
    "\nLatent points shape:",
    latent_points.shape
)

# ============================================================
# 11. SAVE LATENT DATA
# ============================================================

latent_data = np.column_stack(
    [
        latent_points[:, 0],
        latent_points[:, 1],
        all_labels
    ]
)

np.save(
    os.path.join(
        MODEL_DIR,
        "latent_points.npy"
    ),
    latent_data
)

print(
    "\nSaved latent data:"
)

print(
    "models/latent_points.npy"
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

print("\nYour project now has:")

print("1. encoder.keras")
print("2. decoder.keras")
print("3. latent_points.npy")

print(
    "\nThe latent_points.npy file contains "
    "70,000 MNIST latent coordinates."
)