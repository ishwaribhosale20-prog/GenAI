"""Inference and MP4 export for the pretrained ZeroScope video model."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from config import (
    MIN_GPU_MEMORY_GB,
    MODEL_ID,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    GenerationSettings,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ProgressCallback = Callable[[float, str], None]


class VideoGenerationError(RuntimeError):
    """An actionable error while loading, generating, or exporting a video."""


class InsufficientGPUError(VideoGenerationError):
    """Raised when the available CUDA device cannot safely run this model."""


@dataclass(frozen=True)
class GenerationResult:
    video_path: Path
    prompt: str
    settings: GenerationSettings

    @property
    def duration_seconds(self) -> float:
        return self.settings.num_frames / self.settings.fps


def validate_prompt(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a prompt describing the video you want to generate.")
    prompt = prompt.strip()
    if len(prompt) > 1000:
        raise ValueError("Prompt must be 1000 characters or fewer.")
    return prompt


def _safe_prompt_slug(prompt: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", prompt.lower())
    return "_".join(words[:5]) or "video"


def save_video(
    frames: Sequence[object],
    fps: int,
    output_dir: Path = OUTPUT_DIR,
    prompt: str = "video",
) -> Path:
    """Encode RGB frames to an H.264 MP4 file and verify the saved output."""
    if len(frames) == 0:
        raise VideoGenerationError("The model returned no frames; no video was saved.")
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 24:
        raise ValueError("FPS must be a whole number between 1 and 24.")

    try:
        import imageio.v2 as imageio
        import numpy as np
    except ImportError as exc:
        raise VideoGenerationError(
            "Video export dependencies are missing. Install requirements.txt and retry."
        ) from exc

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VideoGenerationError(
            f"Cannot create the output folder '{output_dir}'. Check folder permissions."
        ) from exc
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"{_safe_prompt_slug(prompt)}_{timestamp}.mp4"
    try:
        with imageio.get_writer(
            str(output_path),
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        ) as writer:
            expected_shape = None
            for index, frame in enumerate(frames):
                array = np.asarray(frame)
                if array.ndim != 3 or array.shape[2] not in (3, 4):
                    raise VideoGenerationError(
                        f"Frame {index} must be an RGB or RGBA image."
                    )
                if expected_shape is None:
                    expected_shape = array.shape[:2]
                elif array.shape[:2] != expected_shape:
                    raise VideoGenerationError(
                        "All video frames must have the same size."
                    )
                if array.shape[2] == 4:
                    array = array[:, :, :3]
                if np.issubdtype(array.dtype, np.floating):
                    if not np.isfinite(array).all():
                        raise VideoGenerationError(
                            f"Frame {index} contains non-finite pixel values."
                        )
                    minimum = float(array.min())
                    maximum = float(array.max())
                    if minimum >= 0.0 and maximum <= 1.0:
                        array = array * 255.0
                    elif minimum >= -1.0 and maximum <= 1.0:
                        array = (array + 1.0) * 127.5
                if array.dtype != np.uint8:
                    array = np.clip(array, 0, 255).astype(np.uint8)
                writer.append_data(array)
    except (OSError, RuntimeError, ValueError) as exc:
        output_path.unlink(missing_ok=True)
        if isinstance(exc, VideoGenerationError):
            raise
        raise VideoGenerationError(
            "Could not encode the MP4. Check that imageio-ffmpeg is installed and "
            "that the output folder is writable."
        ) from exc

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise VideoGenerationError("MP4 export did not create a non-empty file.")
    return output_path


class VideoGenerator:
    """Loads ZeroScope once and reuses its pipeline for subsequent prompts."""

    def __init__(self, cpu_offload: bool = True) -> None:
        self.cpu_offload = cpu_offload
        self._pipeline = None
        self._torch = None
        self._device = None

    def _load_pipeline(self, progress: ProgressCallback | None = None) -> None:
        if self._pipeline is not None:
            return
        if progress:
            progress(0.05, "Loading the pretrained model (first run may take several minutes)...")

        try:
            import torch
            from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline
        except ImportError as exc:
            raise VideoGenerationError(
                "PyTorch or Diffusers is not installed. Install requirements.txt, "
                "then follow the PyTorch CUDA installation instructions in README.md."
            ) from exc

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self._device.type == "cuda" else torch.float32
        if self._device.type == "cuda":
            memory_gb = (
                torch.cuda.get_device_properties(self._device).total_memory
                / 1_000_000_000
            )
            if memory_gb < MIN_GPU_MEMORY_GB:
                raise InsufficientGPUError(
                    f"Detected {memory_gb:.1f} GB of GPU memory; this model needs "
                    f"at least about {MIN_GPU_MEMORY_GB} GB. Use a Colab GPU with "
                    "more memory, or close other GPU workloads and try again."
                )

        try:
            pipeline = DiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                use_safetensors=False,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise VideoGenerationError(
                f"Could not load {MODEL_ID}. Check your internet connection, available "
                "disk/RAM, and Hugging Face model access."
            ) from exc

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        if self._device.type == "cuda":
            if self.cpu_offload:
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to(self._device)
            pipeline.enable_attention_slicing("auto")
        else:
            pipeline.to(self._device)
        self._pipeline = pipeline

    def generate(
        self,
        prompt: str,
        settings: GenerationSettings | None = None,
        progress: ProgressCallback | None = None,
    ) -> GenerationResult:
        prompt = validate_prompt(prompt)
        settings = (settings or GenerationSettings()).validate()
        self._load_pipeline(progress)
        torch = self._torch

        seed = settings.seed
        if seed == -1:
            seed = int(torch.randint(0, 4_294_967_296, (1,)).item())
        actual_settings = replace(settings, seed=seed)
        generator = torch.Generator(device=self._device).manual_seed(seed)

        def on_step(step_index, _timestep, _latents):
            if progress:
                fraction = (step_index + 1) / settings.num_inference_steps
                progress(
                    0.12 + fraction * 0.82,
                    f"Generating frame sequence: step "
                    f"{step_index + 1}/{settings.num_inference_steps}",
                )

        try:
            with torch.inference_mode():
                result = self._pipeline(
                    prompt,
                    height=VIDEO_HEIGHT,
                    width=VIDEO_WIDTH,
                    num_frames=settings.num_frames,
                    num_inference_steps=settings.num_inference_steps,
                    guidance_scale=float(settings.guidance_scale),
                    generator=generator,
                    callback=on_step,
                    callback_steps=1,
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if self._device.type == "cuda":
                    torch.cuda.empty_cache()
                raise InsufficientGPUError(
                    "GPU ran out of memory during generation. Try fewer frames, "
                    "fewer inference steps, or a Colab GPU with more memory."
                ) from exc
            raise VideoGenerationError(f"Video generation failed: {exc}") from exc

        frames = result.frames[0]
        if len(frames) == 0:
            raise VideoGenerationError("The model completed without returning any frames.")
        if len(frames) != settings.num_frames:
            raise VideoGenerationError(
                f"The model returned {len(frames)} frames; expected "
                f"{settings.num_frames}."
            )
        frame_shape = getattr(frames[0], "shape", None)
        if frame_shape is not None:
            frame_height, frame_width = frame_shape[:2]
        else:
            frame_width, frame_height = frames[0].size
        if (frame_height, frame_width) != (VIDEO_HEIGHT, VIDEO_WIDTH):
            raise VideoGenerationError(
                f"The model returned {frame_width}x{frame_height} frames; expected "
                f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}."
            )
        if progress:
            progress(0.96, "Encoding the generated frames into MP4...")
        video_path = save_video(frames, settings.fps, prompt=prompt)
        if progress:
            progress(1.0, "Video is ready.")
        return GenerationResult(video_path, prompt, actual_settings)
