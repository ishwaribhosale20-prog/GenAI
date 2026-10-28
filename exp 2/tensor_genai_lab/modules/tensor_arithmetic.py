"""Tensor arithmetic, element-wise operations, and shape error handling."""

import tensorflow as tf


def add_compatible_tensors(left: tf.Tensor, right: tf.Tensor) -> tf.Tensor:
    """Add broadcast-compatible tensors and explain incompatible shapes."""
    try:
        return tf.add(left, right)
    except (ValueError, tf.errors.InvalidArgumentError) as error:
        raise ValueError(
            f"These tensor shapes cannot be added: {left.shape} and {right.shape}. "
            "Use tensors with equal or broadcast-compatible shapes."
        ) from error


def run_demo() -> None:
    """Demonstrate common arithmetic operations on compatible tensors."""
    print("\nMODULE 4 - TENSOR ARITHMETIC")
    print("-" * 58)
    first = tf.constant([[8.0, 6.0], [4.0, 2.0]])
    second = tf.constant([[2.0, 3.0], [2.0, 1.0]])
    print(f"First tensor:\n{first.numpy()}")
    print(f"Second tensor:\n{second.numpy()}")
    print(f"Addition (element by element):\n{add_compatible_tensors(first, second).numpy()}")
    print(f"Subtraction (element by element):\n{tf.subtract(first, second).numpy()}")
    print(f"Multiplication (element by element):\n{tf.multiply(first, second).numpy()}")
    print(f"Division (element by element):\n{tf.divide(first, second).numpy()}")

    row_offsets = tf.constant([10.0, 20.0])
    print(
        "Compatible shapes (2, 2) and (2,) use broadcasting:\n"
        f"{add_compatible_tensors(first, row_offsets).numpy()}"
    )
    print("Arithmetic above is element-wise; matrix multiplication is a separate operation.")
