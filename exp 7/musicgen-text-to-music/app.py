from pathlib import Path

import gradio as gr

from config import DEFAULT_CONFIG, EXAMPLE_PROMPTS
from music_generator import MusicGenerator

generator = MusicGenerator()


def generate_music(prompt: str, duration: float):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt describing the music you want.")
    try:
        result = generator.generate(prompt, duration)
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc
    status = (
        f"Generated {result.duration_seconds:.1f} seconds at {result.sampling_rate:,} Hz. "
        f"Saved as {result.wav_path.name}. Device: {generator.device}."
    )
    return str(result.wav_path), str(result.wav_path), f"Prompt: {result.prompt}\n{status}"


def compare_music_prompts(prompts: str, duration: float):
    prompt_list = [item.strip() for item in prompts.splitlines() if item.strip()]
    if not prompt_list:
        raise gr.Error("Enter at least one prompt, one prompt per line.")
    if len(prompt_list) > 5:
        raise gr.Error("Compare up to five prompts at a time.")
    audio_paths: list[str | None] = [None] * 5
    descriptions = [""] * 5
    for index, item in enumerate(prompt_list):
        try:
            result = generator.generate(item, duration)
        except (ValueError, RuntimeError) as exc:
            raise gr.Error(f"Prompt {index + 1} failed: {exc}") from exc
        audio_paths[index] = str(result.wav_path)
        descriptions[index] = (
            f"Prompt: {result.prompt}\n"
            f"{result.duration_seconds:.1f}s at {result.sampling_rate:,} Hz — {result.wav_path.name}"
        )
    return [value for pair in zip(audio_paths, descriptions) for value in pair]


with gr.Blocks(title="MusicGen Text-to-Music Lab", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# MusicGen Text-to-Music Lab\n"
        "Generate a short music sample from a text description using Meta's pretrained "
        "`facebook/musicgen-small` model. This project runs inference only; it does not "
        "train a model or need a music dataset."
    )
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Describe the music",
                placeholder="For example: gentle acoustic guitar music for studying",
                lines=3,
            )
            examples = gr.Dropdown(
                choices=EXAMPLE_PROMPTS,
                label="Example prompt",
                value=None,
                interactive=True,
            )
            duration = gr.Slider(
                minimum=DEFAULT_CONFIG.min_duration_seconds,
                maximum=DEFAULT_CONFIG.max_duration_seconds,
                value=DEFAULT_CONFIG.default_duration_seconds,
                step=1,
                label="Duration (seconds; maximum 30)",
            )
            generate_button = gr.Button("Generate Music", variant="primary")
        with gr.Column(scale=1):
            status = gr.Textbox(label="Generation status", lines=4, interactive=False)
    audio_player = gr.Audio(label="Generated composition", type="filepath")
    download = gr.File(label="Download WAV")

    examples.change(lambda value: value or "", inputs=examples, outputs=prompt)
    generate_button.click(generate_music, inputs=[prompt, duration], outputs=[audio_player, download, status])
    with gr.Tab("Compare prompts"):
        gr.Markdown("Enter up to five music descriptions, one per line, to generate and compare.")
        prompt_list = gr.Textbox(
            label="Prompts to compare",
            placeholder="\n".join(EXAMPLE_PROMPTS[:3]),
            lines=5,
        )
        compare_duration = gr.Slider(
            minimum=DEFAULT_CONFIG.min_duration_seconds,
            maximum=DEFAULT_CONFIG.max_duration_seconds,
            value=5,
            step=1,
            label="Duration for each sample (seconds)",
        )
        compare_button = gr.Button("Generate comparison", variant="primary")
        compare_outputs = []
        for index in range(5):
            with gr.Row():
                compare_outputs.extend(
                    [
                        gr.Audio(label=f"Prompt {index + 1}", type="filepath"),
                        gr.Textbox(label="Prompt and audio details", interactive=False),
                    ]
                )
        compare_button.click(
            compare_music_prompts,
            inputs=[prompt_list, compare_duration],
            outputs=compare_outputs,
        )
    gr.Markdown(
        "Prompts that mention genre, instruments, mood, and tempo can help guide generation. "
        "Different random samples may still vary. The first request downloads and loads model "
        "weights; later requests reuse the loaded model."
    )


def main() -> None:
    demo.queue().launch()


if __name__ == "__main__":
    main()
