"""Reshape tensors and add, remove, or reorder dimensions."""

import tensorflow as tf


def run_demo() -> None:
    """Show tensor values and shapes before and after dimension operations."""
    print("\nMODULE 6 - RESHAPING AND DIMENSIONS")
    print("-" * 58)
    original = tf.constant([[1, 2, 3], [4, 5, 6]])
    print(f"Original, shape={original.shape}:\n{original.numpy()}")

    reshaped = tf.reshape(original, (3, 2))
    print(f"tf.reshape to (3, 2), shape={reshaped.shape}:\n{reshaped.numpy()}")

    expanded = tf.expand_dims(original, axis=0)
    print(f"tf.expand_dims at axis 0, shape={expanded.shape}:\n{expanded.numpy()}")

    squeezed = tf.squeeze(expanded, axis=0)
    print(f"tf.squeeze at axis 0, shape={squeezed.shape}:\n{squeezed.numpy()}")

    transposed = tf.transpose(original)
    print(f"tf.transpose, shape={transposed.shape}:\n{transposed.numpy()}")
    print("Reshape changes the arrangement of dimensions; transpose swaps their order.")
