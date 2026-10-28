# Text Generation Question–Answer System

A beginner-friendly question-answer demo using Google's pre-trained
[`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) model.
It generates answers from the model's learned knowledge; it does not search
your files or guarantee that an answer is correct.

## Aim

Implement a text-generation-based Question–Answer system with a pre-trained
Transformer model.

## Objectives

- Load the Hugging Face FLAN-T5 tokenizer and sequence-to-sequence model.
- Detect CUDA and use it when available, otherwise use the CPU.
- Turn a user's question into a clear instruction prompt and generate an answer.
- Demonstrate the full workflow through a Gradio web interface.
- Observe and discuss the generated outputs and their limitations.

## Theory

### Transformer models

A Transformer is a neural-network architecture for working with sequences of
text. Its attention mechanism helps it relate words to other words in the
context, including words that are far apart. This makes Transformers useful
for tasks such as translation, summarization, and question answering.

### Pre-trained language models

A pre-trained model learns patterns from a large collection of text before it
is used in an application. It can then be adapted to follow instructions or
perform a task. This project uses an existing model; it does not train a model
from scratch.

### FLAN-T5

T5 (Text-to-Text Transfer Transformer) expresses many language tasks as text
input mapped to text output. FLAN-T5 is an instruction-tuned version of T5.
This project uses the `base` checkpoint, `google/flan-t5-base`, from Hugging
Face. The tokenizer and model are downloaded on the first run; internet access
and sufficient disk space and memory may be required. The model is kept in
memory and is not reloaded for each question.

### Tokenization and generation

Tokenization converts a prompt into numerical token IDs that the model can
process. `model.generate()` predicts output token IDs using configurable
generation settings. The tokenizer decodes those IDs back into readable text.
The demo uses beam search (`num_beams=4`), deterministic generation
(`do_sample=False`), and a limit of 128 new tokens to keep answers concise.
These settings are defined near the top of `app.py`.

## Technology stack

- Python 3.12 (recommended)
- PyTorch
- Hugging Face Transformers
- Hugging Face `google/flan-t5-base`
- SentencePiece
- Accelerate
- Gradio
- Optional CUDA GPU
- Google Colab-compatible notebook

## Prerequisites

- Python 3.12 and `pip`
- Internet access for the initial package/model downloads
- Several GB of free storage and available RAM; a GPU is optional
- On Windows, use the PyTorch installation selector if you want a specific
  CUDA-enabled build: <https://pytorch.org/get-started/locally/>

The project can run on CPU, but generation may be slower. A GPU is reported
only when `torch.cuda.is_available()` confirms that CUDA can be used.

## Installation and run (Windows PowerShell)

From the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

If PowerShell blocks virtual-environment activation, either allow it for the
current user according to your institution's policy, or run the environment's
Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open the local URL printed by Gradio in your browser. The initial run downloads
the model from Hugging Face and can take time. If a chosen PyTorch wheel is not
available for your Python/platform combination, install the appropriate build
using the PyTorch website, then install the remaining project dependencies.

## Google Colab

Open `notebooks/text_generation_qa_colab.ipynb` in Google Colab (or upload it
to Colab), then run its cells from top to bottom. In **Runtime → Change runtime
type**, select a GPU if one is available. The notebook installs dependencies,
loads the same model, runs sample questions, and starts Gradio.

The notebook launches Gradio without a public share link by default. If you
choose to enable a temporary share link, anyone with that link may be able to
access the demo while it is active. Do not enter private or sensitive
information.

## Sample questions

- What is Artificial Intelligence?
- Explain how a Transformer works.
- What is Generative AI?
- What is the difference between AI and Machine Learning?
- Explain neural networks in simple words.

Use the interface's Examples list or enter your own question. The sample
responses are generated at runtime; no fixed answer is presented as a model
output. To record response analysis for a practical report, run the notebook's
test cell and copy the actual question/response pairs from that run.

## How the system works

```text
User Question
    → Instruction Prompt
    → Tokenization
    → FLAN-T5 Model
    → Generated Tokens
    → Decoded Answer
    → Gradio Interface
```

`app.py` loads the model once when the Gradio interface is built.
`generate_answer(question)` validates the question, builds the prompt,
tokenizes it, generates output tokens, and decodes the output. The interface
displays the answer and the model/device status. Errors during loading or
generation are logged and shown as an informative message instead of being
silently treated as successful output.

## Limitations

- The answer comes from learned model patterns, not a search of supplied
  documents or a live knowledge source.
- The model may produce incorrect, incomplete, outdated, or differently worded
  responses. Verify important claims independently.
- CPU inference can be slow. Memory requirements vary by operating system,
  PyTorch build, and runtime.
- The model download requires internet access the first time unless the
  checkpoint is already cached locally.
- A temporary Gradio share link is public-facing and should not be used for
  sensitive questions or information.

## Conclusion

This project demonstrates how a pre-trained, instruction-tuned Transformer
can generate a text answer to a user question. It covers model loading,
device selection, tokenization, inference, decoding, and a simple interface.
It is a learning demonstration, not a factuality guarantee.

## Laboratory checklist

| Laboratory objective/procedure | Implementation |
|---|---|
| Load the FLAN-T5 tokenizer and model | `load_model()` in `app.py`; notebook model-loading cell |
| Detect CUDA and otherwise use CPU | `torch.cuda.is_available()` in both implementations |
| Load once rather than per question | Global model/tokenizer and one call during app construction |
| Validate empty questions | `generate_answer()` in `app.py` and notebook |
| Prompt, tokenize, generate, and decode | `generate_answer()` in both implementations |
| Configure concise generation | `MAX_NEW_TOKENS`, `NUM_BEAMS`, and `DO_SAMPLE` constants |
| Build interactive Gradio UI | `build_demo()` and notebook interface cell |
| Demonstrate sample questions and response analysis | Gradio examples and notebook sample test cell |
| Explain workflow and limitations | This README and notebook markdown |
| Use Google Colab | `notebooks/text_generation_qa_colab.ipynb` |
| Handle loading/generation errors | Logging and user-visible messages in `app.py` |

## Viva preparation

**Q: What is a Transformer?**  
A: A neural-network architecture that uses attention to model relationships
between tokens in a sequence.

**Q: What does the tokenizer do?**  
A: It converts text to token IDs the model can process, and converts generated
token IDs back to text.

**Q: What is FLAN-T5?**  
A: An instruction-tuned T5 model that represents tasks as text-to-text input
and output.

**Q: Why use `model.eval()` and `torch.inference_mode()`?**  
A: Evaluation mode disables training-specific behavior, while inference mode
avoids tracking gradients and reduces inference overhead.

**Q: How is the execution device selected?**  
A: The application checks `torch.cuda.is_available()` and selects CUDA only
when it is available; otherwise it selects CPU.

**Q: Is the answer guaranteed to be correct or based on my documents?**  
A: No. It is generated from learned model knowledge and may be inaccurate. It
does not retrieve or cite a supplied document collection.
