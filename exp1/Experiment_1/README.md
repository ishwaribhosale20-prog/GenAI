# Experiment 1: Comprehensive Dataset Analysis and Preprocessing

**Subtitle:** Understand any structured dataset like a data scientist

## Aim

To inspect, clean, explore, visualize, and prepare a structured CSV dataset for
machine-learning and deep-learning workflows, while demonstrating optional
TensorFlow/Keras, PyTorch, and LangChain integrations.

## Objectives

- Inspect dataset dimensions, schema, descriptive statistics, missing values,
  duplicates, and memory use with Pandas.
- Clean a copy of the input without overwriting the original dataset.
- Automatically identify numerical and categorical columns.
- Demonstrate leakage-safe Scikit-learn preprocessing and train/test splitting.
- Generate relevant exploratory visualizations and save them to disk.
- Convert prepared feature data to NumPy and, when installed, TensorFlow and
  PyTorch tensors.
- Demonstrate a small supervised model only when a target is explicitly given.
- Ask simple dataset questions offline, with an opt-in LangChain LLM path.

## Project structure

```text
Experiment_1/
├── experiment1.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_dataset.csv
└── outputs/                       # Created automatically when run
    ├── cleaned_dataset.csv
    ├── processed_dataset.csv
    ├── dataset_summary.txt
    ├── model_evaluation.txt        # Only if an optional model is trained
    └── plots/
```

## Libraries

| Library | Use in this experiment |
|---|---|
| NumPy | Dense arrays and numerical inputs for ML frameworks |
| Pandas | CSV loading, inspection, cleaning, summaries, and exports |
| Matplotlib and Seaborn | Histograms, category bars, box plots, scatter plots, correlations, and pair plots |
| Scikit-learn | Imputation, `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, pipelines, and splitting |
| TensorFlow / Keras (optional) | Tensor conversion and a small illustrative supervised model |
| PyTorch (optional) | Tensor conversion and a small illustrative supervised model |
| LangChain (optional) | Optional LLM explanation grounded in a computed dataset summary |

TensorFlow, PyTorch, and LangChain are optional so the core experiment runs
without large deep-learning downloads or API credentials. LangChain is an
orchestration library, not a machine-learning model and not an automatic data
cleaner.

## Installation

Use Python 3.10 or later. In PowerShell, from the project directory:

```powershell
cd Experiment_1
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, either allow it for the current
terminal (`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`) or run
the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Optional packages can be installed separately, as needed:

```powershell
python -m pip install tensorflow
python -m pip install torch
python -m pip install langchain langchain-openai
```

Deep-learning wheels can be large and may have platform-specific Python
compatibility. Install only the framework you intend to demonstrate. Check the
framework's official install selector if a wheel is unavailable for your
Python version or hardware.

## Run the experiment

Run the complete experiment using the included synthetic dataset:

```powershell
python .\experiment1.py
```

Specify a custom CSV file:

```powershell
python .\experiment1.py --data "D:\datasets\my_data.csv"
```

### Interactive Streamlit dashboard

The separate `ui_app.py` dashboard provides dataset upload and preview, data
quality details, cleaning and preprocessing controls, interactive charts, and
dataset question answering. It uses the analysis and preprocessing functions
from `experiment1.py`; it does not replace the command-line workflow.

If the UI dependency is not already installed, install it, then launch the
dashboard from this directory:

```powershell
python -m pip install "streamlit>=1.40"
python -m streamlit run .\ui_app.py
```

The included sample CSV is selected by default. Uploaded CSV files are read
without changing the original, and generated CSVs and the analysis report can
be downloaded from the dashboard.

Specify a target to run a train/test split and, if optional frameworks are
installed, small illustrative Keras/PyTorch training runs:

```powershell
python .\experiment1.py --data "D:\datasets\my_data.csv" --target "label"
```

Other useful options:

```powershell
python .\experiment1.py --help
python .\experiment1.py --data .\data\sample_dataset.csv --target passed --test-size 0.25
python .\experiment1.py --data .\data\sample_dataset.csv --target score --task regression
python .\experiment1.py --drop-constant
python .\experiment1.py --ask "Which columns contain missing values?"
```

`--test-size` must be from 0.05 through 0.5. The target is optional. If a
target is provided but its name is not present in the CSV, the program reports
an error and lists the available columns. `--task auto` treats text and
small-cardinality targets as classification and high-cardinality numeric
targets as regression; use `--task` to override that heuristic when the
meaning of a numeric target is known.

### Optional LangChain question answering

The default question-answering demonstration runs locally with Pandas and does
not require credentials or network access. To opt in to an LLM explanation:

```powershell
$env:OPENAI_API_KEY = "your-key"
python .\experiment1.py --use-llm --ask "Summarize the numerical columns"
```

The API key is read from the environment and is never printed. This explicitly
enabled path sends the computed dataset summary and question—not raw dataset
rows—to the provider. An LLM-generated response is an explanation, not a
verified statistical calculation. Without a key or the optional LangChain
packages, the experiment falls back to local Pandas answers.

## Workflow

1. **Load and inspect:** read the selected CSV, display first/last rows, schema,
   `info()`, numeric and categorical summaries, unique counts, missing-value
   percentages, duplicates, and memory use.
2. **Clean a copy:** remove exact duplicate rows, convert text columns to
   numeric only when every non-missing value can be parsed, impute numerical
   features with their median, and impute categorical features with their mode.
   If a feature is entirely missing, use zero for a numeric feature or an
   explicit `(Missing)` category. Missing target labels are not invented;
   those rows are excluded from supervised training. Constant columns are
   reported and retained unless `--drop-constant` is set.
3. **Preprocess:** numerical features use median imputation and
   `StandardScaler`; categorical features use most-frequent imputation and
   `OneHotEncoder(handle_unknown="ignore")`. If a target is selected, raw
   missing feature values are retained in a separate model-input copy; the
   train/test split is made before fitting its imputer and preprocessing
   transformer, which are fitted only on training features. The imputed
   cleaned CSV and full-data transformed export are for inspection/export, not
   held-out model evaluation.
4. **Explore:** save suitable histograms, bar charts, box plots, numerical
   scatter plots, correlation heatmaps, and a pair plot. Plots are skipped
   when data do not support them.
5. **Convert and integrate:** display a NumPy sample, then attempt optional
   TensorFlow and PyTorch tensor conversion. If a target is supplied, use a
   compact classification or regression example when an optional framework is
   installed. No pretrained model is downloaded.
6. **Ask dataset questions:** answer supported questions with Pandas offline;
   optionally use LangChain with an environment-supplied key for an
   explanation based on computed summary statistics.

`processed_dataset.csv` is a full-data transformed export for exploration and
is **not** used to calculate held-out model metrics. When a target is supplied,
the separate model path fits its preprocessing only on the training split,
then transforms the test split without refitting. Do not use the
full-data-transformed export as an evaluation dataset.

## Expected outputs

The `outputs` directory is created automatically. The program writes:

- `cleaned_dataset.csv` — cleaned copy; source CSV remains unchanged.
- `processed_dataset.csv` — transformed features and, if specified, the
  original target column.
- `dataset_summary.txt` — inspection results, preprocessing notes, optional
  component status, and question-answer demonstration.
- `plots/` — generated visualizations.
- `model_evaluation.txt` — held-out results only when an optional framework
  trains a model.

## Common errors and solutions

- **CSV not found:** verify the path after `--data`; quote paths containing
  spaces.
- **Empty or malformed CSV:** ensure the file has a header and data rows and
  is a valid comma-separated UTF-8 text file.
- **Target column not found:** check exact spelling and capitalization with
  the displayed column list.
- **Too few labeled rows or only one target value:** training is skipped with
  an explanation; the dataset is still inspected and exported.
- **Missing optional TensorFlow/PyTorch:** install the selected framework
  separately, or continue with the core experiment.
- **Optional package incompatible with Python/platform:** use a supported
  Python version and the package's official installation instructions.
- **No LLM response:** the local Pandas fallback is available without a key;
  for LLM use, install `langchain` and `langchain-openai` and set
  `OPENAI_API_KEY`.
- **No plot for a chart type:** plots require suitable non-empty columns; for
  example, correlation and pair plots require at least two numerical columns.

## Conclusion

This experiment demonstrates a reproducible data-preparation workflow from
raw CSV inspection to cleaned, visualized, and model-ready features. It
emphasizes preservation of the source data, automatic handling of mixed
column types, appropriate treatment of missing values, and prevention of
train/test leakage. Optional deep-learning and generative-AI components
illustrate downstream integration without making credentials or large model
downloads prerequisites for learning the core workflow.
