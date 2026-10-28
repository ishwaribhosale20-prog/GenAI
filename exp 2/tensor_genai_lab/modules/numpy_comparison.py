"""Compare TensorFlow tensors with NumPy arrays."""

import numpy as np
import tensorflow as tf


def run_demo() -> None:
    """Convert values between TensorFlow and NumPy and compare calculations."""
    print("\nMODULE 7 - TENSORFLOW AND NUMPY")
    print("-" * 58)
    tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    array = tensor.numpy()
    print(f"TensorFlow tensor: values={tensor.numpy()}, dtype={tensor.dtype.name}")
    print(f"NumPy array: values={array}, dtype={array.dtype}")

    numpy_result = np.square(array) + 1
    tensorflow_result = tf.square(tensor) + 1
    print(f"NumPy: square(values) + 1 = {numpy_result}")
    print(
        "TensorFlow: square(values) + 1 = "
        f"{tensorflow_result.numpy()} (dtype={tensorflow_result.dtype.name})"
    )
    print(
        "Both libraries provide array-style numerical operations. NumPy is a general-purpose "
        "CPU array library; TensorFlow tensors can participate in automatic differentiation "
        "and execute on supported accelerators."
    )
