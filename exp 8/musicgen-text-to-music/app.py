"""Run the beginner-friendly Gradio interface for the MusicGen practical."""

from datetime import datetime
from pathlib import Path

import gradio as gr

from config import (
    DEFAULT_DURATION_SECONDS,
    EXAMPLE_PROMPTS,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
)
from music_generator import MusicGenerator

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
generator = MusicGenerator()


def _generate(
    model_generator: MusicGenerator,
    prompt: str,
    duration_seconds: int,
):
    """Generate one prompt and provide the player, download, and status."""
    output_path = OUTPUT_DIR / (
        f"musicgen_output_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
    )
    try:
        result = model_generator.generate(prompt, duration_seconds, output_path)
    except (OSError, RuntimeError, ValueError) as exc:
        return None, None, f"Generation failed: {exc}"

    status = (
        f'Generated from prompt: "{prompt.strip()}"\n\n'
        f"Duration: {result.duration_seconds:.1f} s | "
        f"Sampling rate: {result.sample_rate:,} Hz | "
        f"Device: {result.device}"
    )
    if result.device == "cpu":
        status += "\n\nNote: CPU inference works, but a Colab GPU is much faster."
    return str(result.path), str(result.path), status


def build_demo(model_generator: MusicGenerator | None = None):
    """Create the interface; an existing generator can be reused by a notebook."""
    active_generator = model_generator or generator

    def generate(prompt: str, duration_seconds: int):
        return _generate(active_generator, prompt, duration_seconds)

    with gr.Blocks(title="MusicGen Text-to-Music", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Text-to-Music with MusicGen\n"
            "Describe a short composition, generate it with the pretrained model, "
            "then listen or download the WAV file."
        )
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    label="Describe the music",
                    placeholder="For example: gentle acoustic guitar music for studying",
                    lines=3,
                )
                example = gr.Dropdown(
                    label="Example prompts",
                    choices=EXAMPLE_PROMPTS,
                    value=None,
                    interactive=True,
                )
                example.change(
                    fn=lambda selected: selected or "",
                    inputs=example,
                    outputs=prompt,
                )
                duration = gr.Slider(
                    minimum=MIN_DURATION_SECONDS,
                    maximum=MAX_DURATION_SECONDS,
                    value=DEFAULT_DURATION_SECONDS,
                    step=1,
                    label="Duration (seconds)",
                )
                generate_button = gr.Button("Generate Music", variant="primary")
            with gr.Column(scale=2):
                audio = gr.Audio(label="Generated composition", type="filepath")
                download = gr.File(label="Download WAV")
        status = gr.Markdown("Ready. Choose an example or enter your own prompt.")
        generate_button.click(
            fn=generate,
            inputs=[prompt, duration],
            outputs=[audio, download, status],
            concurrency_limit=1,
        )
    return demo


if __name__ == "__main__":
    build_demo().queue().launch()
