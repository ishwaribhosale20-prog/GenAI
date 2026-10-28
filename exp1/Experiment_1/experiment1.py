"""Comprehensive dataset analysis and preprocessing for a CSV dataset."""

from __future__ import annotations

import argparse
import importlib
import itertools
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Any


try:
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
except ImportError as exc:
    raise SystemExit(
        "A required package could not be imported. Install the core dependencies with:\n"
        "  py -m pip install -r Experiment_1\\requirements.txt\n"
        f"Details: {exc}"
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLE = PROJECT_DIR / "data" / "sample_dataset.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs"
RANDOM_STATE = 42


def emit(lines: list[str], message: str = "") -> None:
    """Print a message and retain it for the text report."""
    print(message)
    lines.append(message)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a non-empty CSV file without changing the source file."""
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise ValueError(f"CSV file not found: {path}") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The CSV file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse the CSV file: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Could not decode the CSV as UTF-8. Save it as UTF-8 and try again."
        ) from exc

    if frame.empty:
        raise ValueError("The CSV has column headers but contains no data rows.")
    if len(frame.columns) == 0:
        raise ValueError("The CSV does not contain any usable columns.")
    if not frame.columns.is_unique:
        raise ValueError("The CSV contains duplicate column names; rename them first.")
    return frame


def describe_dataset(frame: pd.DataFrame, lines: list[str]) -> None:
    """Print and record the primary inspection information."""
    emit(lines, "\n=== Dataset overview ===")
    emit(lines, f"Shape: {frame.shape} (rows, columns)")
    emit(lines, f"Rows: {len(frame):,} | Columns: {len(frame.columns):,}")
    emit(lines, f"Column names: {list(frame.columns)}")
    emit(lines, "\nFirst five records:")
    emit(lines, frame.head().to_string())
    emit(lines, "\nLast five records:")
    emit(lines, frame.tail().to_string())
    emit(lines, "\nColumn data types:")
    emit(lines, frame.dtypes.to_string())

    info_buffer = StringIO()
    frame.info(buf=info_buffer, memory_usage="deep")
    emit(lines, "\nPandas df.info():")
    emit(lines, info_buffer.getvalue().rstrip())

    numeric = frame.select_dtypes(include="number")
    categorical = frame.select_dtypes(exclude="number")
    emit(lines, "\nNumerical descriptive statistics:")
    emit(lines, numeric.describe().to_string() if not numeric.empty else "No numerical columns.")
    emit(lines, "\nCategorical descriptive statistics:")
    emit(
        lines,
        categorical.describe().to_string()
        if not categorical.empty
        else "No categorical columns.",
    )
    emit(lines, "\nUnique values per column:")
    emit(lines, frame.nunique(dropna=False).to_string())

    missing = pd.DataFrame(
        {
            "missing_count": frame.isna().sum(),
            "missing_percent": (frame.isna().mean() * 100).round(2),
        }
    )
    emit(lines, "\nMissing values by column:")
    emit(lines, missing.to_string())
    emit(lines, f"\nDuplicate rows: {int(frame.duplicated().sum()):,}")
    memory_mib = frame.memory_usage(index=True, deep=True).sum() / (1024**2)
    emit(lines, f"Memory usage: {memory_mib:.3f} MiB")


def normalize_numeric_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns to numbers only when every present value is numeric."""
    normalized = frame.copy()
    for column in normalized.columns:
        series = normalized[column]
        if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(
            series.dtype
        ):
            present = series.notna()
            if not present.any():
                continue
            converted = pd.to_numeric(series, errors="coerce")
            if converted[present].notna().all():
                normalized[column] = converted
    return normalized


def clean_dataset(
    original: pd.DataFrame,
    target: str | None,
    drop_constant: bool,
    lines: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a cleaned export and a pre-imputation copy for model splitting."""
    cleaned = normalize_numeric_strings(original.copy())
    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()
    emit(lines, "\n=== Data cleaning ===")
    emit(lines, f"Removed duplicate rows: {duplicate_count}")

    constants = [
        column
        for column in cleaned.columns
        if cleaned[column].nunique(dropna=False) <= 1
    ]
    emit(lines, f"Constant columns detected: {constants or 'None'}")
    removable = [column for column in constants if column != target]
    if drop_constant and removable:
        cleaned = cleaned.drop(columns=removable)
        emit(lines, f"Removed constant feature columns: {removable}")
    elif removable:
        emit(
            lines,
            "Constant columns are retained by default; pass --drop-constant to remove them.",
        )

    model_frame = cleaned.copy()
    for column in cleaned.columns:
        if column == target:
            missing_targets = int(cleaned[column].isna().sum())
            if missing_targets:
                emit(
                    lines,
                    f"Target '{target}' has {missing_targets} missing labels; "
                    "they are preserved and excluded from supervised training.",
                )
            continue

        missing_count = int(cleaned[column].isna().sum())
        if missing_count == 0:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            median = cleaned[column].median()
            fill_value: Any = median if pd.notna(median) else 0
            strategy = "median" if pd.notna(median) else "0 (all values were missing)"
        else:
            modes = cleaned[column].mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else "(Missing)"
            strategy = "most frequent" if not modes.empty else "explicit '(Missing)' category"
        cleaned[column] = cleaned[column].fillna(fill_value)
        emit(
            lines,
            f"Filled {missing_count} missing value(s) in '{column}' using {strategy}.",
        )

    remaining = int(cleaned.isna().sum().sum())
    emit(lines, f"Missing cells after cleaning: {remaining}")
    emit(lines, f"Cleaned shape: {cleaned.shape}")
    if target:
        emit(
            lines,
            "The supervised path retains pre-imputation feature values so its imputer "
            "can be fitted on training rows only.",
        )
    return cleaned, model_frame


def get_feature_columns(
    frame: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical input columns for sklearn."""
    numeric = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
        and not pd.api.types.is_complex_dtype(frame[column])
    ]
    categorical = [column for column in frame.columns if column not in numeric]
    return numeric, categorical


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer | None:
    """Build a trainable numerical/categorical preprocessing transformer."""
    numeric_columns, categorical_columns = get_feature_columns(frame)
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        return None
    return ColumnTransformer(transformers=transformers, remainder="drop")


def to_dense_array(values: Any) -> np.ndarray:
    """Return a dense float array for use by downstream ML frameworks."""
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def preprocess_for_export(
    cleaned: pd.DataFrame, target: str | None, output_path: Path, lines: list[str]
) -> np.ndarray | None:
    """Create an all-row preview export; this is not used for model evaluation."""
    feature_frame = cleaned.drop(columns=[target]) if target else cleaned.copy()
    if feature_frame.empty or feature_frame.shape[1] == 0:
        emit(lines, "\nNo feature columns are available for preprocessing.")
        return None

    transformer = make_preprocessor(feature_frame)
    if transformer is None:
        emit(lines, "\nNo supported feature columns are available for preprocessing.")
        return None
    transformed = to_dense_array(transformer.fit_transform(feature_frame))
    feature_names = transformer.get_feature_names_out()
    processed = pd.DataFrame(transformed, columns=feature_names, index=cleaned.index)
    if target:
        processed[target] = cleaned[target]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    emit(lines, "\n=== Scikit-learn preprocessing ===")
    emit(
        lines,
        f"Automatic feature detection: {len(get_feature_columns(feature_frame)[0])} numerical, "
        f"{len(get_feature_columns(feature_frame)[1])} categorical.",
    )
    emit(lines, "Numerical features use median imputation + StandardScaler.")
    emit(lines, "Categorical features use most-frequent imputation + OneHotEncoder.")
    emit(lines, "OneHotEncoder(handle_unknown='ignore') safely handles unseen categories.")
    emit(
        lines,
        "This full-data CSV is a convenient preparation preview only; model evaluation "
        "uses a separate transformer fitted on training data.",
    )
    emit(lines, f"Processed feature array shape: {transformed.shape}")
    emit(lines, f"Processed dataset saved to: {output_path}")
    return transformed


def determine_task(target: pd.Series, task_choice: str) -> str:
    """Choose classification for labels and small integer-valued targets."""
    if task_choice != "auto":
        return task_choice
    if (
        pd.api.types.is_numeric_dtype(target)
        and target.nunique() > 10
        and not pd.api.types.is_bool_dtype(target)
    ):
        return "regression"
    return "classification"


def prepare_supervised_data(
    cleaned: pd.DataFrame,
    target: str,
    test_size: float,
    task_choice: str,
    lines: list[str],
) -> dict[str, Any] | None:
    """Split first, then fit preprocessing only on the training features."""
    labeled = cleaned.loc[cleaned[target].notna()].copy()
    if len(labeled) < 6:
        emit(lines, "\nSupervised training skipped: at least 6 labeled rows are required.")
        return None
    if labeled[target].nunique() < 2:
        emit(lines, "\nSupervised training skipped: the target has fewer than 2 values.")
        return None

    feature_frame = labeled.drop(columns=[target])
    if feature_frame.shape[1] == 0:
        emit(lines, "\nSupervised training skipped: no feature columns remain.")
        return None
    task = determine_task(labeled[target], task_choice)

    class_labels: np.ndarray | None = None
    if task == "classification":
        encoder = LabelEncoder()
        y_values = encoder.fit_transform(labeled[target].astype(str))
        class_labels = encoder.classes_
    else:
        y_values = pd.to_numeric(labeled[target], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y_values)
        feature_frame = feature_frame.loc[valid]
        y_values = y_values[valid]
        if len(y_values) < 6:
            emit(lines, "\nSupervised training skipped: too few valid numeric target values.")
            return None

    class_count = len(np.unique(y_values)) if task == "classification" else 0
    stratify: np.ndarray | None = None
    if task == "classification":
        counts = np.unique(y_values, return_counts=True)[1]
        test_rows = int(np.ceil(len(y_values) * test_size))
        train_rows = len(y_values) - test_rows
        if counts.min() >= 2 and test_rows >= class_count and train_rows >= class_count:
            stratify = y_values
        else:
            emit(
                lines,
                "Class stratification is not feasible for this small dataset; "
                "using a reproducible unstratified split.",
            )

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            feature_frame,
            y_values,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
    except ValueError as exc:
        emit(lines, f"\nSupervised training skipped: train/test split failed: {exc}")
        return None

    transformer = make_preprocessor(x_train)
    if transformer is None:
        emit(lines, "\nSupervised training skipped: no supported feature columns.")
        return None
    x_train_array = to_dense_array(transformer.fit_transform(x_train))
    x_test_array = to_dense_array(transformer.transform(x_test))
    if x_train_array.shape[1] == 0:
        emit(lines, "\nSupervised training skipped: preprocessing produced no features.")
        return None

    emit(lines, "\n=== Supervised train/test preparation ===")
    emit(lines, f"Target: {target} | task: {task}")
    emit(lines, f"Training rows: {len(y_train)} | Test rows: {len(y_test)}")
    emit(lines, f"Transformed train shape: {x_train_array.shape}")
    emit(lines, f"Transformed test shape: {x_test_array.shape}")
    emit(lines, "Leakage check: preprocessing was fitted on X_train only.")
    return {
        "task": task,
        "x_train": x_train_array,
        "x_test": x_test_array,
        "y_train": np.asarray(y_train),
        "y_test": np.asarray(y_test),
        "class_labels": class_labels,
    }


def make_eda_plots(frame: pd.DataFrame, output_dir: Path, lines: list[str]) -> None:
    """Save useful plots only when the dataset has suitable columns."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    numeric, categorical = get_feature_columns(frame)
    numeric = [
        column
        for column in numeric
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    created = 0

    for column in numeric[:8]:
        values = frame[column].dropna()
        if values.empty:
            continue
        fig, axis = plt.subplots(figsize=(7, 4))
        sns.histplot(values, kde=values.nunique() > 1, ax=axis, color="#4169e1")
        axis.set(title=f"Distribution of {column}", xlabel=column, ylabel="Frequency")
        fig.tight_layout()
        fig.savefig(plot_dir / f"histogram_{created + 1}.png", dpi=140)
        plt.close(fig)
        created += 1

        fig, axis = plt.subplots(figsize=(7, 4))
        sns.boxplot(x=values, ax=axis, color="#77bde0")
        axis.set(title=f"Box plot of {column}", xlabel=column, ylabel="")
        fig.tight_layout()
        fig.savefig(plot_dir / f"boxplot_{created}.png", dpi=140)
        plt.close(fig)
        created += 1

    for column in categorical[:6]:
        counts = frame[column].astype("string").fillna("(Missing)").value_counts().head(15)
        if counts.empty:
            continue
        fig, axis = plt.subplots(figsize=(8, 4.5))
        sns.barplot(x=counts.values, y=counts.index.astype(str), ax=axis, color="#6a5acd")
        axis.set(
            title=f"Category frequencies: {column}",
            xlabel="Frequency",
            ylabel=column,
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"categories_{created + 1}.png", dpi=140)
        plt.close(fig)
        created += 1

    numeric_pairs = list(itertools.islice(itertools.combinations(numeric, 2), 3))
    for first, second in numeric_pairs:
        pair = frame[[first, second]].dropna()
        if len(pair) < 2:
            continue
        fig, axis = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=pair, x=first, y=second, ax=axis, color="#008b8b")
        axis.set(title=f"{second} vs. {first}", xlabel=first, ylabel=second)
        fig.tight_layout()
        fig.savefig(plot_dir / f"scatter_{created + 1}.png", dpi=140)
        plt.close(fig)
        created += 1

    if len(numeric) >= 2:
        corr = frame[numeric].corr(numeric_only=True)
        if corr.shape[0] >= 2:
            fig, axis = plt.subplots(figsize=(max(6, len(numeric) * 0.8), 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=axis)
            axis.set(title="Numerical feature correlation", xlabel="Feature", ylabel="Feature")
            fig.tight_layout()
            fig.savefig(plot_dir / "correlation_heatmap.png", dpi=140)
            plt.close(fig)
            created += 1

    pair_columns = numeric[:4]
    if len(pair_columns) >= 2:
        pair_data = frame[pair_columns].dropna()
        if len(pair_data) > 1:
            pair_data = pair_data.sample(min(500, len(pair_data)), random_state=RANDOM_STATE)
            grid = sns.pairplot(pair_data, corner=True, diag_kind="hist")
            grid.figure.suptitle("Pair plot of numerical features", y=1.02)
            grid.figure.savefig(plot_dir / "pairplot.png", dpi=120, bbox_inches="tight")
            plt.close(grid.figure)
            created += 1

    emit(lines, "\n=== Exploratory data analysis ===")
    emit(lines, f"Plots generated: {created}")
    emit(
        lines,
        f"Plot directory: {plot_dir}"
        if created
        else "No plots generated: the dataset has no suitable non-empty feature columns.",
    )


def run_tensorflow_demo(
    transformed: np.ndarray | None,
    supervised: dict[str, Any] | None,
    lines: list[str],
) -> dict[str, Any] | None:
    """Show TensorFlow tensors and optionally train a tiny Keras model."""
    emit(lines, "\n=== TensorFlow / Keras demonstration ===")
    try:
        tf = importlib.import_module("tensorflow")
    except (ImportError, OSError) as exc:
        emit(
            lines,
            "TensorFlow is not installed or could not be loaded; core analysis continues. "
            "Install with: py -m pip install tensorflow",
        )
        emit(lines, f"TensorFlow detail: {exc}")
        return None

    emit(lines, f"TensorFlow version: {tf.__version__}")
    if transformed is not None:
        tensor = tf.convert_to_tensor(transformed, dtype=tf.float32)
        emit(lines, f"Feature tensor shape: {tuple(tensor.shape)} | dtype: {tensor.dtype.name}")
        emit(lines, f"Tensor sample (first 3 rows):\n{tensor[:3].numpy()}")
    if supervised is None:
        emit(lines, "No supervised target was selected; tensor preparation only.")
        return None

    x_train = supervised["x_train"]
    x_test = supervised["x_test"]
    y_train = supervised["y_train"]
    y_test = supervised["y_test"]
    task = supervised["task"]
    classes = supervised["class_labels"]
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x_train.shape[1],)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
        ]
    )
    if task == "regression":
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        train_targets = y_train.astype(np.float32)
        test_targets = y_test.astype(np.float32)
    elif len(classes) == 2:
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        train_targets = y_train.astype(np.float32)
        test_targets = y_test.astype(np.float32)
    else:
        model.add(tf.keras.layers.Dense(len(classes), activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        train_targets = y_train.astype(np.int32)
        test_targets = y_test.astype(np.int32)

    history = model.fit(
        x_train,
        train_targets,
        epochs=5,
        batch_size=min(16, len(x_train)),
        verbose=0,
    )
    training_metrics = {
        key: float(values[-1])
        for key, values in history.history.items()
        if values
    }
    metric_values = model.evaluate(x_test, test_targets, verbose=0, return_dict=True)
    metrics = {key: float(value) for key, value in metric_values.items()}
    emit(lines, f"Keras final training epoch: {training_metrics}")
    emit(lines, f"Keras evaluation (held-out test set; 5 epochs): {metrics}")
    return {"TensorFlow/Keras": {"training": training_metrics, "evaluation": metrics}}


def run_pytorch_demo(
    transformed: np.ndarray | None,
    supervised: dict[str, Any] | None,
    lines: list[str],
) -> dict[str, Any] | None:
    """Show PyTorch tensors and optionally train a small illustrative network."""
    emit(lines, "\n=== PyTorch demonstration ===")
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        emit(
            lines,
            "PyTorch is not installed or could not be loaded; core analysis continues. "
            "Install with: py -m pip install torch",
        )
        emit(lines, f"PyTorch detail: {exc}")
        return None

    emit(lines, f"PyTorch version: {torch.__version__}")
    if transformed is not None:
        tensor = torch.as_tensor(transformed, dtype=torch.float32)
        emit(lines, f"Feature tensor shape: {tuple(tensor.shape)} | dtype: {tensor.dtype}")
        emit(lines, f"Tensor sample (first 3 rows):\n{tensor[:3].cpu().numpy()}")
    if supervised is None:
        emit(lines, "No supervised target was selected; tensor preparation only.")
        return None

    torch.manual_seed(RANDOM_STATE)
    x_train = torch.as_tensor(supervised["x_train"], dtype=torch.float32)
    x_test = torch.as_tensor(supervised["x_test"], dtype=torch.float32)
    y_train_np = supervised["y_train"]
    y_test_np = supervised["y_test"]
    task = supervised["task"]
    classes = supervised["class_labels"]
    if task == "regression":
        y_train = torch.as_tensor(y_train_np, dtype=torch.float32).reshape(-1, 1)
        y_test = torch.as_tensor(y_test_np, dtype=torch.float32).reshape(-1, 1)
        output_count = 1
        loss_function = torch.nn.MSELoss()
    elif len(classes) == 2:
        y_train = torch.as_tensor(y_train_np, dtype=torch.float32).reshape(-1, 1)
        y_test = torch.as_tensor(y_test_np, dtype=torch.float32).reshape(-1, 1)
        output_count = 1
        loss_function = torch.nn.BCEWithLogitsLoss()
    else:
        y_train = torch.as_tensor(y_train_np, dtype=torch.long)
        y_test = torch.as_tensor(y_test_np, dtype=torch.long)
        output_count = len(classes)
        loss_function = torch.nn.CrossEntropyLoss()

    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, output_count),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    final_training_loss = 0.0
    for _ in range(5):
        order = torch.randperm(len(x_train))
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, len(order), 16):
            batch = order[start : start + 16]
            optimizer.zero_grad()
            loss = loss_function(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batch_count += 1
        final_training_loss = epoch_loss / batch_count

    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
        test_loss = float(loss_function(predictions, y_test).item())
        if task == "regression":
            metric_name = "mae"
            metric = float(torch.mean(torch.abs(predictions - y_test)).item())
        elif len(classes) == 2:
            metric_name = "accuracy"
            metric = float(((torch.sigmoid(predictions) >= 0.5) == y_test.bool()).float().mean().item())
        else:
            metric_name = "accuracy"
            metric = float((predictions.argmax(dim=1) == y_test).float().mean().item())
    metrics = {
        "final_training_loss": final_training_loss,
        "test_loss": test_loss,
        metric_name: metric,
    }
    emit(lines, f"PyTorch final training epoch loss: {final_training_loss:.4f}")
    emit(lines, f"PyTorch evaluation (held-out test set; 5 epochs): {metrics}")
    return {"PyTorch": metrics}


def make_dataset_context(frame: pd.DataFrame) -> str:
    """Build concise, data-derived context without including raw data rows."""
    numeric = frame.select_dtypes(include="number")
    stats = numeric.describe().round(3).to_string() if not numeric.empty else "None"
    dtypes = ", ".join(f"{name}: {dtype}" for name, dtype in frame.dtypes.items())
    return (
        f"Rows: {len(frame)}\nColumns: {len(frame.columns)}\n"
        f"Column types: {dtypes}\n"
        f"Missing values by column: {frame.isna().sum().to_dict()}\n"
        f"Duplicate rows: {int(frame.duplicated().sum())}\n"
        f"Numerical descriptive statistics:\n{stats}"
    )


def answer_offline_question(question: str, frame: pd.DataFrame) -> str:
    """Answer common dataset questions using verified Pandas calculations."""
    text = question.casefold()
    if "row" in text and ("column" in text or "shape" in text or "many" in text):
        return f"The dataset contains {len(frame)} rows and {len(frame.columns)} columns."
    if "missing" in text or "null" in text:
        missing = frame.isna().sum()
        values = missing[missing > 0]
        return (
            "Columns with missing values: "
            + (", ".join(f"{name} ({count})" for name, count in values.items()) or "none.")
        )
    if "duplicate" in text:
        return f"The dataset contains {int(frame.duplicated().sum())} duplicate row(s)."
    if "summar" in text or "numerical" in text or "numeric" in text:
        numeric = frame.select_dtypes(include="number")
        return (
            numeric.describe().round(3).to_string()
            if not numeric.empty
            else "There are no numerical columns to summarize."
        )
    if "distribution" in text or "frequency" in text or "category" in text:
        for column in frame.select_dtypes(exclude="number").columns:
            if str(column).casefold() in text:
                return frame[column].value_counts(dropna=False).to_string()
        return "Include the categorical column name to show its frequency distribution."
    return (
        "Offline mode supports questions about row/column counts, missing values, "
        "duplicates, numerical summaries, and categorical frequencies."
    )


def answer_dataset_question(
    question: str, frame: pd.DataFrame, use_llm: bool, lines: list[str]
) -> str:
    """Use LangChain only after explicit opt-in; otherwise calculate locally."""
    if not use_llm or not os.environ.get("OPENAI_API_KEY"):
        if use_llm:
            emit(
                lines,
                "OPENAI_API_KEY is not set; answering offline without contacting an API.",
            )
        return answer_offline_question(question, frame)

    try:
        chat_model = importlib.import_module("langchain_openai").ChatOpenAI
    except (ImportError, OSError) as exc:
        emit(
            lines,
            "Optional LangChain packages are unavailable. Install with: "
            "py -m pip install langchain langchain-openai",
        )
        emit(lines, f"LangChain detail: {exc}")
        return answer_offline_question(question, frame)

    prompt = (
        "Answer the question using only the computed dataset summary below. "
        "If the summary does not contain the answer, say so. Do not invent statistics. "
        "The response is an LLM explanation, not a verified statistical calculation.\n\n"
        f"Dataset summary:\n{make_dataset_context(frame)}\n\n"
        f"Question: {question}"
    )
    openai_errors = importlib.import_module("openai").APIError
    try:
        response = chat_model(model="gpt-4o-mini", temperature=0).invoke(prompt)
    except (openai_errors, OSError, ValueError) as exc:
        emit(
            lines,
            f"The optional LLM request failed ({type(exc).__name__}); "
            "answering offline without retrying.",
        )
        return answer_offline_question(question, frame)
    return str(response.content)


def run_langchain_demo(
    frame: pd.DataFrame, question: str | None, use_llm: bool, lines: list[str]
) -> None:
    """Demonstrate summary-based natural-language questions with offline fallback."""
    emit(lines, "\n=== LangChain / generative AI demonstration ===")
    emit(lines, "Dataset context prepared from dimensions, types, missing values, and statistics.")
    emit(lines, "An LLM is optional; the local Pandas question-answering fallback needs no key.")
    emit(
        lines,
        "Example questions: 'How many rows and columns?', "
        "'Which columns contain missing values?', 'Summarize numerical columns'.",
    )
    prompt = question or "How many rows and columns?"
    emit(lines, f"Question: {prompt}")
    answer = answer_dataset_question(prompt, frame, use_llm, lines)
    emit(lines, f"Answer: {answer}")
    if use_llm and os.environ.get("OPENAI_API_KEY"):
        emit(
            lines,
            "The LLM received only the computed summary and your question; "
            "its explanation is not a verified statistical calculation.",
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect, clean, visualize, and preprocess a structured CSV dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_SAMPLE,
        help="Path to a CSV file (default: the included synthetic sample dataset).",
    )
    parser.add_argument(
        "--target",
        help="Optional target column name for a train/test split and illustrative training.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of labeled rows held out for evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--task",
        choices=("auto", "classification", "regression"),
        default="auto",
        help="Target task type; auto treats text/small-cardinality targets as classification.",
    )
    parser.add_argument(
        "--drop-constant",
        action="store_true",
        help="Remove constant feature columns from the cleaned copy (target is retained).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for CSV, report, and plot outputs.",
    )
    parser.add_argument(
        "--ask",
        help="Ask a natural-language question about computed dataset statistics.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Opt in to LangChain/OpenAI using OPENAI_API_KEY; sends summary metadata, not raw rows.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if not 0.05 <= args.test_size <= 0.5:
        print("Error: --test-size must be between 0.05 and 0.5.", file=sys.stderr)
        return 2

    try:
        original = load_dataset(args.data)
        if args.target is not None and args.target not in original.columns:
            raise ValueError(
                f"Target column '{args.target}' was not found. "
                f"Available columns: {list(original.columns)}"
            )

        lines: list[str] = []
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        emit(lines, "Experiment 1: Comprehensive Dataset Analysis and Preprocessing")
        emit(lines, f"Input CSV: {args.data.resolve()}")
        emit(lines, f"NumPy {np.__version__} | Pandas {pd.__version__}")
        emit(
            lines,
            f"Matplotlib {matplotlib.__version__} | Seaborn {sns.__version__} | "
            f"Scikit-learn {importlib.import_module('sklearn').__version__}",
        )
        emit(lines, "\nOriginal input is read-only; all cleaning uses a separate copy.")
        describe_dataset(original, lines)

        cleaned, model_frame = clean_dataset(
            original, args.target, args.drop_constant, lines
        )
        cleaned_path = output_dir / "cleaned_dataset.csv"
        cleaned.to_csv(cleaned_path, index=False)
        emit(lines, f"\nCleaned dataset saved to: {cleaned_path}")

        processed = preprocess_for_export(
            cleaned, args.target, output_dir / "processed_dataset.csv", lines
        )
        if processed is not None:
            emit(lines, "\n=== NumPy conversion ===")
            emit(lines, f"Array shape: {processed.shape} | dtype: {processed.dtype}")
            emit(lines, f"Array sample (first 3 rows):\n{processed[:3]}")
            emit(
                lines,
                "Dense arrays are a common input format for Scikit-learn, TensorFlow/Keras, "
                "and PyTorch models.",
            )

        supervised = None
        if args.target:
            supervised = prepare_supervised_data(
                model_frame, args.target, args.test_size, args.task, lines
            )
        else:
            emit(
                lines,
                "\nNo target column supplied: supervised training and train/test splitting skipped.",
            )

        make_eda_plots(cleaned, output_dir, lines)
        metrics: dict[str, Any] = {}
        try:
            tf_metrics = run_tensorflow_demo(processed, supervised, lines)
        except (RuntimeError, TypeError, ValueError) as exc:
            emit(
                lines,
                f"TensorFlow/Keras demonstration could not complete "
                f"({type(exc).__name__}): {exc}",
            )
            tf_metrics = None
        try:
            torch_metrics = run_pytorch_demo(processed, supervised, lines)
        except (RuntimeError, TypeError, ValueError) as exc:
            emit(
                lines,
                f"PyTorch demonstration could not complete ({type(exc).__name__}): {exc}",
            )
            torch_metrics = None
        if tf_metrics:
            metrics.update(tf_metrics)
        if torch_metrics:
            metrics.update(torch_metrics)
        if metrics:
            metrics_path = output_dir / "model_evaluation.txt"
            metrics_path.write_text(
                "\n".join(f"{name}: {values}" for name, values in metrics.items()),
                encoding="utf-8",
            )
            emit(lines, f"Evaluation results saved to: {metrics_path}")

        run_langchain_demo(original, args.ask, args.use_llm, lines)
        report_path = output_dir / "dataset_summary.txt"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSummary report saved to: {report_path}")
        return 0
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
