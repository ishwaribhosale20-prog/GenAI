"""Python values, arithmetic, lists, and introductory NumPy operations."""

import numpy as np


def run_demo() -> None:
    """Show common Python data types and basic array operations."""
    print("\nMODULE 1 - PYTHON BASICS")
    print("-" * 58)

    student_name = "Asha"  # str: text
    experiment_number = 1  # int: whole number
    learning_rate = 0.01  # float: decimal number
    is_learning = True  # bool: true/false value
    print(
        f"Variables: name={student_name!r} ({type(student_name).__name__}), "
        f"experiment={experiment_number} ({type(experiment_number).__name__}), "
        f"learning_rate={learning_rate} ({type(learning_rate).__name__}), "
        f"is_learning={is_learning} ({type(is_learning).__name__})"
    )

    first_number, second_number = 12, 4
    print(
        f"Arithmetic with {first_number} and {second_number}: "
        f"add={first_number + second_number}, subtract={first_number - second_number}, "
        f"multiply={first_number * second_number}, divide={first_number / second_number}"
    )

    scores = [8, 9, 10]
    feature_rows = [[1, 0], [0, 1]]
    print(f"List: {scores}; first item={scores[0]}")
    print(f"Nested list: {feature_rows}; second row={feature_rows[1]}")

    array = np.array(feature_rows, dtype=np.float32)
    print(f"NumPy array:\n{array}")
    print(f"Array + 2:\n{array + 2}")
    print(f"Array shape={array.shape}, dtype={array.dtype}, mean={array.mean():.2f}")
