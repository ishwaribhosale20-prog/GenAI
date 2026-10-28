"""Optional Streamlit dashboard for the lab demonstrations."""

from contextlib import redirect_stdout
from io import StringIO

import streamlit as st

from modules import (
    genai_demo,
    mathematical_functions,
    numpy_comparison,
    python_basics,
    tensor_arithmetic,
    tensor_creation,
    tensor_reshaping,
)


st.set_page_config(page_title="TensorFlow GenAI Foundations Lab", page_icon="🧮", layout="wide")
st.title("TensorFlow & Python: Generative AI Foundations")
st.caption("Aim: build a practical foundation in Python, NumPy, and TensorFlow 2 tensors.")

modules = {
    "Python basics": python_basics.run_demo,
    "Tensor creation": tensor_creation.run_demo,
    "Tensor arithmetic": tensor_arithmetic.run_demo,
    "Mathematical functions": mathematical_functions.run_demo,
    "Reshaping and dimensions": tensor_reshaping.run_demo,
    "TensorFlow and NumPy": numpy_comparison.run_demo,
    "Generative AI connection": genai_demo.run_demo,
}

st.sidebar.header("Lab modules")
selected_module = st.sidebar.radio("Choose a demonstration", ["Overview", *modules.keys()])

if selected_module == "Overview":
    st.subheader("Welcome")
    st.write(
        "Explore how Python values and arrays become tensors, how tensor operations work, "
        "and why tensors are the numerical building blocks used in deep learning."
    )
    st.info("Choose a module from the sidebar, then select **Run demonstration**.")
    st.markdown(
        """
        **What you will practice**
        - Python values, lists, nested lists, and NumPy arrays
        - Tensor ranks, shapes, data types, and arithmetic
        - Mathematical reductions and dimension transformations
        - Converting between TensorFlow tensors and NumPy arrays
        - A small, illustrative feature-vector transformation
        """
    )
else:
    st.subheader(selected_module)
    st.write("Run the examples to see values and results calculated with real TensorFlow/NumPy operations.")
    if st.button("Run demonstration", type="primary"):
        output = StringIO()
        with redirect_stdout(output):
            modules[selected_module]()
        st.code(output.getvalue(), language="text")
