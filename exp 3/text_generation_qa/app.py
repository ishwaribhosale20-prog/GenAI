"""Interactive question-answer demo powered by Google's FLAN-T5 model."""

from __future__ import annotations

import logging
from typing import Any

MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 128
NUM_BEAMS = 4
DO_SAMPLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

tokenizer: Any | None = None
model: Any | None = None
device: Any | None = None
model_status = "Model has not been loaded."
model_error: str | None = None


def load_model() -> None:
    """Load the tokenizer and model once, choosing CUDA when it is available."""
    global tokenizer, model, device, model_status, model_error

    if model is not None and tokenizer is not None:
        return

    model_status = f"Loading {MODEL_NAME} from Hugging Face..."
    logger.info(model_status)
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        loaded_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        loaded_model.to(device)
        loaded_model.eval()

        tokenizer = loaded_tokenizer
        model = loaded_model
        model_error = None
        model_status = f"Ready — {MODEL_NAME} on {device}."
        logger.info(model_status)
    except Exception as exc:
        model_error = f"{type(exc).__name__}: {exc}"
        model_status = (
            f"Could not load {MODEL_NAME}. Check your internet connection, "
            "installed dependencies, and available memory."
        )
        logger.exception("Model loading failed")


def generate_answer(question: str) -> str:
    """Generate a concise answer to a question using the loaded FLAN-T5 model."""
    if not isinstance(question, str) or not question.strip():
        return "Please enter a question before submitting."

    if model is None or tokenizer is None or device is None:
        detail = f"\nDetails: {model_error}" if model_error else ""
        return f"The model is not available. {model_status}{detail}"

    try:
        import torch

        prompt = (
            "Answer the following question clearly and concisely. "
            "If you are unsure, say so.\n"
            f"Question: {question.strip()}\n"
            "Answer:"
        )

        # Tokenization converts the instruction text into model-readable IDs.
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        # Generation creates answer-token IDs; decoding turns those IDs into text.
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                do_sample=DO_SAMPLE,
            )

        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return answer or "The model returned an empty answer. Please try rephrasing."
    except Exception as exc:
        logger.exception("Answer generation failed")
        return (
            "Sorry, answer generation failed. Please try again or check the "
            f"application logs. ({type(exc).__name__}: {exc})"
        )


def build_demo() -> Any:
    """Create the Gradio app after attempting the one-time model load."""
    import gradio as gr

    load_model()
    with gr.Blocks(title="Text Generation Question–Answer System") as demo:
        gr.Markdown(
            "# Text Generation Using FLAN-T5\n"
            "Ask a question and generate an answer with a pre-trained "
            "sequence-to-sequence Transformer."
        )

        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Your question",
                    placeholder="For example: What is Artificial Intelligence?",
                    lines=3,
                )
                with gr.Row():
                    submit_button = gr.Button("Generate answer", variant="primary")
                    clear_button = gr.Button("Clear")
                answer_output = gr.Textbox(
                    label="Generated answer",
                    lines=6,
                    interactive=False,
                )
            with gr.Column(scale=2):
                gr.Markdown("## Model information")
                status_output = gr.Textbox(
                    label="Loading status",
                    value=model_status,
                    interactive=False,
                )
                device_output = gr.Textbox(
                    label="Execution device",
                    value=str(device) if device is not None else "Unavailable",
                    interactive=False,
                )

        gr.Examples(
            examples=[
                ["What is Artificial Intelligence?"],
                ["Explain how a Transformer works."],
                ["What is Generative AI?"],
                ["What is the difference between AI and Machine Learning?"],
                ["Explain neural networks in simple words."],
            ],
            inputs=question_input,
        )
        gr.Markdown(
            "## How it works\n"
            "Question → instruction prompt → tokenization → FLAN-T5 → "
            "generated tokens → decoded answer. The model uses patterns "
            "learned during pre-training and instruction tuning. It is not "
            "connected to a verified document collection and can be wrong."
        )
        submit_button.click(
            fn=generate_answer,
            inputs=question_input,
            outputs=answer_output,
            show_progress="full",
        )
        question_input.submit(
            fn=generate_answer,
            inputs=question_input,
            outputs=answer_output,
            show_progress="full",
        )
        clear_button.click(
            fn=lambda: ("", ""),
            outputs=[question_input, answer_output],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.queue().launch()
