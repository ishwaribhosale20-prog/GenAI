"""Streamlit dashboard for Experiment 1."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

import experiment1


def load_uploaded_dataset(contents: bytes) -> pd.DataFrame:
    """Pass an uploaded CSV through the experiment's existing file loader."""
    with TemporaryDirectory(prefix="experiment1-upload-") as temporary_directory:
        uploaded_path = Path(temporary_directory) / "uploaded.csv"
        uploaded_path.write_bytes(contents)
        return experiment1.load_dataset(uploaded_path)


st.set_page_config(
    page_title="Experiment 1 | Dataset Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(87, 130, 255, .12), transparent 34%),
            #f6f8fc;
    }
    [data-testid="stHeader"] { background: rgba(246, 248, 252, .82); }
    [data-testid="stSidebar"] { background: #101a2b; }
    [data-testid="stSidebar"] * { color: #eef3ff; }
    [data-testid="stMetric"] {
        background: #fff;
        border: 1px solid #e6eaf2;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 5px 18px rgba(18, 38, 63, .04);
    }
    div[data-testid="stTabs"] button { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 📊 Dataset Lab")
    st.caption("Experiment 1 · Comprehensive Dataset Analysis")
    st.divider()
    source = st.radio("Choose a dataset", ("Included sample", "Upload a CSV"))
    uploaded_file = None
    if source == "Upload a CSV":
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="The original file is read only; cleaning creates a separate copy.",
        )
    st.divider()

    if source == "Included sample":
        try:
            original = experiment1.load_dataset(experiment1.DEFAULT_SAMPLE)
        except (OSError, ValueError) as exc:
            st.error(f"Could not load the included sample: {exc}")
            st.stop()
        dataset_name = experiment1.DEFAULT_SAMPLE.name
    elif uploaded_file is None:
        st.info("Upload a CSV to start exploring your dataset.")
        st.stop()
    else:
        try:
            original = load_uploaded_dataset(uploaded_file.getvalue())
        except (OSError, ValueError) as exc:
            st.error(f"Could not load this CSV: {exc}")
            st.stop()
        dataset_name = uploaded_file.name

    st.markdown("### Preparation settings")
    target_choice = st.selectbox(
        "Optional target column",
        options=[None, *original.columns],
        format_func=lambda value: "No target" if value is None else str(value),
        help="Select a label or outcome to prepare a classification or regression split.",
    )
    target = target_choice
    drop_constant = st.checkbox(
        "Remove constant feature columns",
        help="The target column is retained even when it is constant.",
    )
    task_choice = st.selectbox(
        "Target task",
        ("auto", "classification", "regression"),
        help="Auto-detect uses the experiment's existing target-type heuristic.",
        disabled=target is None,
    )
    test_size = st.slider(
        "Test-set fraction",
        min_value=0.05,
        max_value=0.50,
        value=0.20,
        step=0.05,
        disabled=target is None,
    )

analysis_lines: list[str] = []
report_output = StringIO()
with redirect_stdout(report_output):
    experiment1.describe_dataset(original, analysis_lines)
    cleaned, model_frame = experiment1.clean_dataset(
        original, target, drop_constant, analysis_lines
    )

    with TemporaryDirectory(prefix="experiment1-output-") as temporary_directory:
        processed_path = Path(temporary_directory) / "processed_dataset.csv"
        transformed = experiment1.preprocess_for_export(
            cleaned, target, processed_path, analysis_lines
        )
        processed_csv = (
            processed_path.read_bytes() if transformed is not None else None
        )

if target is not None:
    with redirect_stdout(report_output):
        supervised = experiment1.prepare_supervised_data(
            model_frame, target, test_size, task_choice, analysis_lines
        )
else:
    supervised = None
    analysis_lines.append(
        "\nNo target column supplied: supervised train/test preparation skipped."
    )

cleaned_csv = cleaned.to_csv(index=False).encode("utf-8")
summary_report = ("\n".join(analysis_lines) + "\n").encode("utf-8")

st.title("Comprehensive Dataset Analysis & Preprocessing")
st.markdown(
    "A focused workspace to **inspect, clean, explore, and prepare** structured data "
    "for machine-learning workflows."
)
st.caption(f"Current dataset: **{dataset_name}** · Source data is never overwritten.")

overview_tab, quality_tab, prepare_tab, explore_tab, questions_tab = st.tabs(
    ["Overview", "Data quality", "Prepare data", "Explore", "Ask your dataset"]
)

with overview_tab:
    st.subheader("Your dataset at a glance")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{len(original):,}")
    metric_columns[1].metric("Columns", f"{len(original.columns):,}")
    metric_columns[2].metric("Missing cells", f"{int(original.isna().sum().sum()):,}")
    metric_columns[3].metric("Duplicate rows", f"{int(original.duplicated().sum()):,}")

    st.markdown("#### Preview")
    st.dataframe(original.head(12), use_container_width=True, hide_index=True)
    with st.expander("Column types and dataset summary"):
        column_types = pd.DataFrame(
            {
                "Column": original.columns.map(str),
                "Type": original.dtypes.astype(str).to_numpy(),
                "Non-null values": original.notna().sum().to_numpy(),
                "Distinct values": original.nunique(dropna=False).to_numpy(),
            }
        )
        st.dataframe(column_types, use_container_width=True, hide_index=True)
    st.download_button(
        "Download dataset analysis report",
        data=summary_report,
        file_name="dataset_summary.txt",
        mime="text/plain",
        key="download_summary",
    )

with quality_tab:
    st.subheader("Data quality")
    missing_count = original.isna().sum()
    quality = pd.DataFrame(
        {
            "Data type": original.dtypes.astype(str),
            "Missing": missing_count,
            "Missing (%)": (missing_count / len(original) * 100).round(2),
            "Distinct values": original.nunique(dropna=False),
        }
    )
    quality.index.name = "Column"
    st.dataframe(quality, use_container_width=True)
    st.markdown("#### Cleaning summary")
    st.write(f"Duplicate rows removed: **{int(original.duplicated().sum()):,}**")
    st.write(
        f"Missing cells after cleaning: **{int(cleaned.isna().sum().sum()):,}**"
    )
    st.caption(
        "Feature missing values use the experiment's median/mode strategy. "
        "Missing target labels are preserved and excluded from supervised preparation."
    )
    st.download_button(
        "Download cleaned CSV",
        data=cleaned_csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
        key="download_cleaned",
    )

with prepare_tab:
    st.subheader("Preprocessing controls")
    st.write(
        "Use the settings in the sidebar to choose a target, optionally remove "
        "constant features, and select the task and test fraction."
    )
    numeric_columns, categorical_columns = experiment1.get_feature_columns(
        cleaned.drop(columns=[target]) if target is not None else cleaned
    )
    stat_columns = st.columns(2)
    stat_columns[0].metric("Numerical features", len(numeric_columns))
    stat_columns[1].metric("Categorical features", len(categorical_columns))
    st.info(
        "Numerical features use median imputation and standard scaling. "
        "Categorical features use most-frequent imputation and one-hot encoding."
    )

    if target is None:
        st.info("Select a target in the sidebar to prepare a supervised data split.")
    elif supervised is None:
        skip_message = next(
            (
                line.strip()
                for line in reversed(analysis_lines)
                if "Supervised training skipped" in line
            ),
            "Supervised preparation could not be completed for this dataset.",
        )
        st.warning(skip_message)
    else:
        st.success(
            f"Prepared a {supervised['task']} split: "
            f"{len(supervised['y_train']):,} training rows and "
            f"{len(supervised['y_test']):,} test rows. "
            "Preprocessing was fitted on training features only."
        )
        st.caption(
            "This prepares model inputs but does not train a model or report "
            "evaluation metrics."
        )

    if processed_csv is not None:
        st.download_button(
            "Download processed feature CSV",
            data=processed_csv,
            file_name="processed_dataset.csv",
            mime="text/csv",
            key="download_processed",
        )
    else:
        st.warning("No processed feature export was generated for this dataset.")

with explore_tab:
    st.subheader("Interactive exploration")
    numeric_columns, categorical_columns = experiment1.get_feature_columns(original)
    chart_type = st.radio(
        "Choose a view",
        ("Distribution", "Category counts", "Scatter plot", "Correlations"),
        horizontal=True,
    )

    if chart_type == "Distribution":
        if not numeric_columns:
            st.info("This dataset has no numerical columns to chart.")
        else:
            column = st.selectbox("Numerical column", numeric_columns)
            values = original[column].dropna()
            if values.empty:
                st.info("This column contains no non-missing values.")
            elif values.nunique() > 1:
                bin_count = min(20, max(5, int(len(values) ** 0.5)))
                counts = pd.cut(values, bins=bin_count).value_counts(sort=False)
                chart_data = pd.DataFrame(
                    {"Rows": counts.to_numpy()},
                    index=counts.index.astype(str),
                )
                chart_data.index.name = column
                st.bar_chart(chart_data)
            else:
                st.bar_chart(values.value_counts().rename("Rows"))
    elif chart_type == "Category counts":
        if not categorical_columns:
            st.info("This dataset has no categorical columns to chart.")
        else:
            column = st.selectbox("Categorical column", categorical_columns)
            counts = (
                original[column]
                .astype("string")
                .fillna("(Missing)")
                .value_counts()
                .head(20)
                .sort_values()
                .rename("Rows")
            )
            st.bar_chart(counts)
            if original[column].nunique(dropna=False) > 20:
                st.caption("Showing the 20 most frequent values.")
    elif chart_type == "Scatter plot":
        if len(numeric_columns) < 2:
            st.info("A scatter plot needs at least two numerical columns.")
        else:
            x_column, y_column = st.columns(2)
            x_value = x_column.selectbox("X axis", numeric_columns, index=0)
            y_options = [column for column in numeric_columns if column != x_value]
            y_value = y_column.selectbox("Y axis", y_options, index=0)
            color_value = st.selectbox(
                "Color by (optional)",
                options=[None, *categorical_columns],
                format_func=lambda value: "No grouping" if value is None else str(value),
            )
            st.scatter_chart(
                original,
                x=x_value,
                y=y_value,
                color=color_value,
                use_container_width=True,
            )
    else:
        if len(numeric_columns) < 2:
            st.info("A correlation view needs at least two numerical columns.")
        else:
            st.dataframe(
                original[numeric_columns].corr(numeric_only=True).round(3),
                use_container_width=True,
            )

with questions_tab:
    st.subheader("Ask a question about this dataset")
    st.caption(
        "Offline answers use the experiment's local Pandas calculations. "
        "Supported topics include dimensions, missing values, duplicates, "
        "numerical summaries, and categorical frequencies."
    )
    with st.form("dataset_question_form"):
        question = st.text_input(
            "Your question",
            placeholder="Which columns contain missing values?",
        )
        use_llm = st.checkbox(
            "Opt in to the optional LangChain/OpenAI explanation",
            help=(
                "When enabled and configured, only the computed dataset summary "
                "and your question are sent. Raw rows are not sent."
            ),
        )
        submitted = st.form_submit_button("Get an answer")

    if use_llm:
        st.info(
            "LLM use is opt-in. The existing integration sends computed summary "
            "metadata and the question, not raw dataset rows."
        )
    if submitted:
        if not question.strip():
            st.warning("Enter a question to get an answer.")
        else:
            question_messages: list[str] = []
            with redirect_stdout(StringIO()):
                answer = experiment1.answer_dataset_question(
                    question, original, use_llm, question_messages
                )
            st.markdown("#### Answer")
            st.write(answer)
            for message in question_messages:
                if "not set" in message or "unavailable" in message or "failed" in message:
                    st.warning(message)
                else:
                    st.caption(message)
