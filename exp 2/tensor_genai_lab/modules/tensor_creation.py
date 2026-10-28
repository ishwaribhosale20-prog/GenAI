"""Create tensors of different ranks and inspect their properties."""

import tensorflow as tf


def create_examples() -> dict[str, tf.Tensor]:
    """Return one tensor for each of the demonstrated ranks."""
    return {
        "Scalar (0-D)": tf.constant(7),
        "Vector (1-D)": tf.constant([1, 2, 3]),
        "Matrix (2-D)": tf.constant([[1, 2], [3, 4]]),
        "3-D tensor": tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        "Higher-dimensional (4-D)": tf.reshape(tf.range(16), (2, 2, 2, 2)),
    }


def describe_tensor(name: str, tensor: tf.Tensor) -> None:
    """Print a tensor's contents and structural properties."""
    print(
        f"{name}\n"
        f"  values:\n{tensor.numpy()}\n"
        f"  shape: {tensor.shape}\n"
        f"  rank: {tf.rank(tensor).numpy()}\n"
        f"  dtype: {tensor.dtype.name}"
    )


def run_demo() -> None:
    """Print scalar, vector, matrix, 3-D, and 4-D tensor examples."""
    print("\nMODULE 3 - TENSOR CREATION")
    print("-" * 58)
    for name, tensor in create_examples().items():
        describe_tensor(name, tensor)
