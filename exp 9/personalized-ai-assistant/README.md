# Personalized AI Assistant

A beginner-friendly GenAI lab project that combines a base instruction-tuned LLM, retrieval-augmented generation (RAG), and a LoRA/QLoRA fine-tuning workflow in a compact Streamlit application.

## Learning goals

- Understand the difference between a base LLM, retrieval and grounding, and fine-tuning.
- Build a practical assistant that answers questions from uploaded documents.
- Compare the output of three modes:
  - base model without document retrieval
  - RAG-enabled model with document grounding
  - optional fine-tuned model when an adapter is available
- Learn how LoRA/QLoRA works in a simple, classroom-friendly workflow.

## Project structure

```text
personalized-ai-assistant/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── models/
│   ├── __init__.py
│   ├── llm.py
│   └── finetuning.py
├── rag/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   └── vector_store.py
├── evaluation/
│   ├── __init__.py
│   └── compare.py
├── data/
│   └── sample_training.jsonl
├── notebooks/
│   └── lora_finetuning_colab.ipynb
└── .gitignore
```

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## How it works

### 1. Base LLM mode

The app loads a small, instruction-tuned open-source model such as Qwen 2.5 0.5B Instruct and answers the question directly without document retrieval. This acts as a baseline.

### 2. RAG mode

Uploaded PDFs, TXT, and DOCX files are read, split into chunks, converted into embeddings, and stored in a FAISS index. When a question is asked, the most relevant chunks are retrieved and used as context for the LLM response.

### 3. Fine-tuning mode

A small JSONL training dataset can be validated and used to fine-tune a base model with LoRA or QLoRA. The trained adapter is saved separately from the base model so the project can keep using the original foundation model while applying a task-specific adapter at inference time.

## Important notes

- The app keeps uploaded documents locally in the current runtime and does not send them to unrelated external services.
- The base model and RAG pipeline should work on CPU for small demonstrations, while GPU acceleration is used when available.
- Fine-tuning is entirely optional and is not required to use the base or RAG modes.
- Use the sample training file as a placeholder for demonstration and test data only.

## Sample training dataset

The example JSONL file in `data/sample_training.jsonl` is illustrative and intentionally small. It is meant to demonstrate the expected schema instead of representing a real production dataset.

## Fine-tuning in Colab

A notebook template for Google Colab is included in `notebooks/lora_finetuning_colab.ipynb`. It shows how to run the adapter-training workflow on GPU hardware.

## Why chunking, embeddings, and vector search matter

Chunking breaks large documents into manageable units. Embeddings convert each chunk into a dense vector. FAISS performs efficient nearest-neighbor search, identifying the text most relevant to the user question. This improves accuracy when the model must answer from user-provided documents without using the whole document as context.

## Limitations

- Large models may require GPU memory, especially for fine-tuning.
- RAG improves document grounding but depends on the quality of the uploaded files and chunking strategy.
- Fine-tuning and retrieval solve different problems. RAG is useful for knowledge injection from documents, while fine-tuning changes the model's general behavior on a task.

## Useful commands

```bash
# Install project requirements
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Optional syntax validation
python -m compileall .
```
