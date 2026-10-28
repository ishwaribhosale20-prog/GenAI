"""Shared settings for the MusicGen text-to-music practical."""

MODEL_ID = "facebook/musicgen-small"
DEFAULT_DURATION_SECONDS = 5
MIN_DURATION_SECONDS = 1
MAX_DURATION_SECONDS = 20
TOKENS_PER_SECOND = 50

EXAMPLE_PROMPTS = [
    "Peaceful instrumental piano with soft flute and a relaxing atmosphere.",
    "Energetic electronic music with a fast tempo and powerful synthesizers.",
    "Indian classical instrumental music featuring sitar and tabla.",
    "Cinematic orchestral music with emotional strings and dramatic percussion.",
    "Gentle acoustic guitar music for studying.",
]
