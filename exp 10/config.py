"""Project settings and validation for the ZeroScope text-to-video demo."""

from dataclasses import dataclass


MODEL_ID = "cerspense/zeroscope_v2_576w"
MODEL_LICENSE = "CC-BY-NC-4.0"
VIDEO_WIDTH = 576
VIDEO_HEIGHT = 320
MIN_GPU_MEMORY_GB = 8
EXAMPLE_PROMPTS = [
    "A small boat sailing across a calm lake at sunrise, gentle water movement, cinematic wide shot.",
    "A butterfly flying through a colorful garden, flowers moving gently in the breeze.",
    "A futuristic city at night with flying vehicles, glowing neon lights, and slow camera movement.",
    "Waves gently hitting a beach at sunset, warm lighting and a peaceful atmosphere.",
    "A small robot walking through a green forest, cinematic lighting and smooth motion.",
]


@dataclass(frozen=True)
class GenerationSettings:
    """Inference controls supported by the selected Diffusers pipeline."""

    num_frames: int = 24
    fps: int = 8
    num_inference_steps: int = 25
    guidance_scale: float = 9.0
    seed: int = 42

    def validate(self) -> "GenerationSettings":
        if isinstance(self.num_frames, bool) or not isinstance(self.num_frames, int):
            raise ValueError("Number of frames must be a whole number.")
        if not 8 <= self.num_frames <= 32:
            raise ValueError("Number of frames must be between 8 and 32.")
        if isinstance(self.fps, bool) or not isinstance(self.fps, int):
            raise ValueError("FPS must be a whole number.")
        if not 1 <= self.fps <= 24:
            raise ValueError("FPS must be between 1 and 24.")
        if isinstance(self.num_inference_steps, bool) or not isinstance(
            self.num_inference_steps, int
        ):
            raise ValueError("Inference steps must be a whole number.")
        if not 10 <= self.num_inference_steps <= 50:
            raise ValueError("Inference steps must be between 10 and 50.")
        if isinstance(self.guidance_scale, bool) or not isinstance(
            self.guidance_scale, (int, float)
        ):
            raise ValueError("Guidance scale must be a number.")
        if not 1.0 <= self.guidance_scale <= 15.0:
            raise ValueError("Guidance scale must be between 1 and 15.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("Seed must be a whole number.")
        if not -1 <= self.seed <= 4_294_967_295:
            raise ValueError("Seed must be -1 (random) or between 0 and 4294967295.")
        return self
