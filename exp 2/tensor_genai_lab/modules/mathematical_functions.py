"""Examples of useful TensorFlow mathematical functions and reductions."""

import tensorflow as tf


def run_demo() -> None:
    """Print element-wise math and aggregate statistics."""
    print("\nMODULE 5 - MATHEMATICAL FUNCTIONS")
    print("-" * 58)
    values = tf.constant([1.0, 4.0, 9.0, 16.0])
    print(f"Input values: {values.numpy()}")
    print(f"tf.square: square each value -> {tf.square(values).numpy()}")
    print(f"tf.sqrt: square root of each value -> {tf.sqrt(values).numpy()}")
    print(f"tf.reduce_mean: average of all values -> {tf.reduce_mean(values).numpy():.2f}")
    print(f"tf.reduce_max: largest value -> {tf.reduce_max(values).numpy():.2f}")
    print(f"tf.reduce_min: smallest value -> {tf.reduce_min(values).numpy():.2f}")
    print(f"tf.reduce_sum: sum of all values -> {tf.reduce_sum(values).numpy():.2f}")
