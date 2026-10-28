from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MusicGenConfig:
    model_id: str = "facebook/musicgen-small"
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 30.0
    default_duration_seconds: float = 8.0
    output_dir: Path = Path("outputs")
    guidance_scale: float = 3.0
    sampling_rate_fallback: int = 32_000

    def validate_duration(self, duration_seconds: float) -> float:
        value = float(duration_seconds)
        if not self.min_duration_seconds <= value <= self.max_duration_seconds:
            raise ValueError(
                f"Duration must be between {self.min_duration_seconds:g} and "
                f"{self.max_duration_seconds:g} seconds."
            )
        return value


DEFAULT_CONFIG = MusicGenConfig()
EXAMPLE_PROMPTS = [
    "Peaceful instrumental piano with soft flute and a relaxing atmosphere.",
    "Energetic electronic music with a fast tempo and powerful synthesizers.",
    "Indian classical instrumental music featuring sitar and tabla.",
    "Cinematic orchestral music with emotional strings and dramatic percussion.",
    "Gentle acoustic guitar music for studying.",
]
