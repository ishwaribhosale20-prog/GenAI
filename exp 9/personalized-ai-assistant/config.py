from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 4
DEFAULT_MAX_NEW_TOKENS = 180
APP_TITLE = "Personalized AI Assistant"
DEMO_TRAINING_DATA = PROJECT_ROOT / "data" / "sample_training.jsonl"
