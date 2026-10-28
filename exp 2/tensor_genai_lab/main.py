"""Run every demonstration in the TensorFlow and Python basics lab."""

import tensorflow as tf

from modules import (
    genai_demo,
    mathematical_functions,
    numpy_comparison,
    python_basics,
    tensor_arithmetic,
    tensor_creation,
    tensor_reshaping,
)


def main() -> None:
    """Print environment information and run each lab module."""
    print("TensorFlow and NumPy Generative AI Foundations Lab")
    print("=" * 58)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Devices available: {', '.join(device.name for device in tf.config.list_physical_devices()) or 'CPU'}")
    print(f"CPU available: {'Yes' if tf.config.list_physical_devices('CPU') else 'No'}")
    print(f"GPU available: {'Yes' if tf.config.list_physical_devices('GPU') else 'No (CPU is ready)'}")

    demonstrations = (
        python_basics.run_demo,
        tensor_creation.run_demo,
        tensor_arithmetic.run_demo,
        mathematical_functions.run_demo,
        tensor_reshaping.run_demo,
        numpy_comparison.run_demo,
        genai_demo.run_demo,
    )
    for demonstration in demonstrations:
        demonstration()


if __name__ == "__main__":
    main()
