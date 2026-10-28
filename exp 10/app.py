"""Beginner-friendly Gradio UI for pretrained text-to-video generation."""

from __future__ import annotations

import math

from config import EXAMPLE_PROMPTS, MODEL_ID, MODEL_LICENSE, GenerationSettings
from video_generator import VideoGenerationError, VideoGenerator, validate_prompt

import gradio as gr


generator = VideoGenerator(cpu_offload=True)


def select_prompt(prompt: str) -> str:
    return prompt


def _whole_number(value: int | float, label: str) -> int:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a whole number.") from exc
    if not math.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(f"{label} must be a whole number.")
    return int(numeric_value)


def generate_video(
    prompt: str,
    num_frames: int,
    fps: int,
    steps: int,
    guidance: float,
    seed: int,
    history: list[dict] | None,
    progress=gr.Progress(track_tqdm=True),
):
    try:
        prompt = validate_prompt(prompt)
        settings = GenerationSettings(
            num_frames=_whole_number(num_frames, "Number of frames"),
            fps=_whole_number(fps, "FPS"),
            num_inference_steps=_whole_number(steps, "Inference steps"),
            guidance_scale=float(guidance),
            seed=_whole_number(seed, "Seed"),
        ).validate()
        result = generator.generate(prompt, settings, progress=progress)
    except (ValueError, VideoGenerationError) as exc:
        return None, f"Generation failed: {exc}", history or [], history or []

    entry = {
        "prompt": result.prompt,
        "duration_seconds": round(result.duration_seconds, 2),
        "fps": result.settings.fps,
        "frames": result.settings.num_frames,
        "steps": result.settings.num_inference_steps,
        "guidance": result.settings.guidance_scale,
        "seed": result.settings.seed,
        "filename": result.video_path.name,
    }
    updated_history = [entry, *(history or [])][:20]
    status = (
        f"Saved **{result.video_path.name}** -- "
        f"{result.settings.num_frames} frames at {result.settings.fps} FPS "
        f"({result.duration_seconds:.2f} seconds), seed "
        f"{result.settings.seed}."
    )
    return str(result.video_path), status, updated_history, updated_history


with gr.Blocks(title="Text-to-Video Generation Lab") as demo:
    gr.Markdown(
        "# Text-to-Video Generation Lab\n"
        "Generate a short clip with pretrained **ZeroScope v2 576w**. "
        "First model download and generation can take several minutes. "
        "A CUDA GPU with about 8 GB or more of memory is recommended."
    )
    gr.Markdown(
        f"Model: [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) | "
        f"License: **{MODEL_LICENSE} (non-commercial use only)**"
    )
    history_state = gr.State([])
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Describe your video",
                placeholder="Describe the scene, subjects, action, camera, lighting, and style...",
                lines=4,
                max_lines=8,
            )
            example = gr.Dropdown(
                choices=EXAMPLE_PROMPTS,
                label="Choose an example prompt",
                value=None,
            )
            example.change(select_prompt, inputs=example, outputs=prompt)
            with gr.Accordion("Generation settings", open=True):
                num_frames = gr.Slider(8, 32, value=24, step=1, label="Number of frames")
                fps = gr.Slider(1, 24, value=8, step=1, label="Playback FPS")
                steps = gr.Slider(10, 50, value=25, step=1, label="Inference steps")
                guidance = gr.Slider(
                    1.0, 15.0, value=9.0, step=0.5, label="Guidance scale"
                )
                seed = gr.Number(value=42, precision=0, label="Random seed (-1 = random)")
            generate_button = gr.Button("Generate video", variant="primary")
            gr.Markdown(
                "More frames make a longer clip and use more memory. More inference "
                "steps can improve detail but take longer. Guidance controls how "
                "strongly the result follows the prompt. A fixed seed helps compare prompts."
            )
        with gr.Column(scale=1):
            preview = gr.Video(label="Generated video", format="mp4", autoplay=False)
            download = gr.File(label="Download MP4")
            status = gr.Markdown("Your generated video and download link will appear here.")

    generate_button.click(
        generate_video,
        inputs=[prompt, num_frames, fps, steps, guidance, seed, history_state],
        outputs=[preview, status, history_state, download],
    )
    gr.Markdown("## Session history")
    history_table = gr.Dataframe(
        headers=[
            "prompt",
            "duration_seconds",
            "fps",
            "frames",
            "steps",
            "guidance",
            "seed",
            "filename",
        ],
        datatype=["str", "number", "number", "number", "number", "number", "number", "str"],
        interactive=False,
        label="Most recent generations first",
    )
    history_state.change(
        lambda items: [
            [
                item[key]
                for key in (
                    "prompt",
                    "duration_seconds",
                    "fps",
                    "frames",
                    "steps",
                    "guidance",
                    "seed",
                    "filename",
                )
            ]
            for item in items
        ],
        inputs=history_state,
        outputs=history_table,
    )


if __name__ == "__main__":
    demo.queue().launch()
