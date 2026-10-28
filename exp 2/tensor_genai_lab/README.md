# TensorFlow and Python Operations for Generative AI Foundations

## Aim

Understand and implement basic Python operations and TensorFlow 2 tensor operations as a foundation for Generative Artificial Intelligence applications.

## Objectives

- Practice Python variables, data types, arithmetic, lists, nested lists, and NumPy arrays.
- Confirm the TensorFlow installation and inspect its version and available hardware.
- Create tensors of ranks 0 through 4 and inspect values, shape, rank, and dtype.
- Perform tensor arithmetic, mathematical functions, reductions, and dimension transformations.
- Compare TensorFlow tensors with NumPy arrays.
- Connect tensor-based feature transformations to neural networks and Generative AI without suggesting that this lab trains a model.

## Theory

A tensor is a generalization of a number or array: a scalar has rank 0, a vector rank 1, a matrix rank 2, and higher-dimensional arrays have higher ranks. A tensor's **shape** gives the size of each dimension; its **dtype** gives the kind of values it stores. TensorFlow operations can be combined into machine-learning computations, differentiated for training, and run on supported hardware. NumPy is also an array-computing library and is useful for data preparation and scientific computing.

## Prerequisites

- Windows 10/11
- Python 3.12 (the requirements use TensorFlow 2.x releases compatible with Python 3.12)
- Internet access for installing packages
- GPU is optional; the examples run on CPU

## Technologies

- Python 3.12
- TensorFlow 2.x
- NumPy
- Streamlit (optional interactive dashboard; included in `requirements.txt`)

## Installation (Windows PowerShell)

From the folder containing `tensor_genai_lab`:

```powershell
cd .\tensor_genai_lab
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run the project

Run the full command-line lab:

```powershell
python .\main.py
```

Run the optional interactive dashboard:

```powershell
python -m streamlit run .\app.py
```

Run the smoke tests:

```powershell
python -m unittest discover -s .\tests -v
```

The program prints the installed TensorFlow version and available devices. A GPU is never required; when no GPU is available, TensorFlow uses the CPU.

## Modules

1. **Python basics** (`modules/python_basics.py`) — strings, integers, floats, booleans, arithmetic, lists, nested lists, and NumPy array calculations.
2. **TensorFlow setup** (`main.py`) — imports TensorFlow, reports its version, lists visible devices, and reports GPU availability.
3. **Tensor creation** (`modules/tensor_creation.py`) — makes scalar, vector, matrix, 3-D, and 4-D tensors; prints values, shape, rank, and dtype.
4. **Tensor arithmetic** (`modules/tensor_arithmetic.py`) — adds, subtracts, multiplies, and divides element-wise; also shows compatible broadcasting and a readable shape error helper.
5. **Mathematical functions** (`modules/mathematical_functions.py`) — uses `tf.square`, `tf.sqrt`, and `tf.reduce_mean`, `tf.reduce_max`, `tf.reduce_min`, and `tf.reduce_sum`.
6. **Reshaping and dimensions** (`modules/tensor_reshaping.py`) — shows `tf.reshape`, `tf.expand_dims`, `tf.squeeze`, and `tf.transpose`.
7. **TensorFlow and NumPy** (`modules/numpy_comparison.py`) — converts a tensor with `.numpy()`, displays dtypes, and calculates the same result with each library.
8. **Generative AI connection** (`modules/genai_demo.py`) — applies a fixed linear operation to toy features and explains how tensor representations appear in neural networks, Transformers, and LLMs.

The `app.py` dashboard runs the same module demonstrations selected in its sidebar. It captures and displays their real output; the computations remain in the reusable module functions.

## Sample output

Excerpt from the verified Python 3.12.10, TensorFlow 2.21.0 CPU run (other environments may report a different version or device):

```text
TensorFlow and NumPy Generative AI Foundations Lab
==========================================================
TensorFlow version: 2.21.0
Devices available: /physical_device:CPU:0
CPU available: Yes
GPU available: No (CPU is ready)

MODULE 3 - TENSOR CREATION
Scalar (0-D)
 values:
7
  shape: ()
  rank: 0
  dtype: int32
Vector (1-D)
  values:
[1 2 3]
  shape: (3,)
  rank: 1
  dtype: int32

MODULE 4 - TENSOR ARITHMETIC
Addition (element by element):
[[10.  9.]
 [ 6.  3.]]

MODULE 8 - CONNECTION TO GENERATIVE AI
Linear transformation (input @ weights + bias):
[[0.54       0.67999995]]
Output shape: (1, 2)
```

These are values calculated by the program during validation; hardware and version lines can vary between environments.

## Tensors in Deep Learning and Generative AI

Inputs such as images, tabular features, and token embeddings are represented numerically in tensors. Neural networks combine tensor operations to compute intermediate activations and predictions. Transformer models organize token representations into tensors and use attention-related computations to model relationships among tokens. Large Language Models apply many such operations to make next-token predictions. This lab demonstrates only small foundational operations and does not implement, train, or simulate a real LLM or generative model.

## Conclusion

This project provides runnable examples of Python and NumPy basics, TensorFlow tensor creation and manipulation, and a carefully scoped connection between numerical tensors and modern AI. Use the CLI for a complete sequential walkthrough or the dashboard to explore modules individually.
