"""Load MusicGen once, generate text-conditioned audio, and save it as WAV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import wave

from config import (
    DEFAULT_DURATION_SECONDS,
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    MODEL_ID,
    TOKENS_PER_SECOND,
)


@dataclass(frozen=True)
class GenerationResult:
    """Details about one successfully generated WAV file."""

    path: Path
    sample_rate: int
    channels: int
    duration_seconds: float
    device: str


def validate_generation_request(prompt: str, duration_seconds: int | float) -> None:
    """Reject an empty prompt or a duration outside the supported range."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a music prompt before generating audio.")
    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float)):
        raise ValueError("Duration must be a whole number of seconds.")
    if not MIN_DURATION_SECONDS <= duration_seconds <= MAX_DURATION_SECONDS:
        raise ValueError(
            f"Duration must be between {MIN_DURATION_SECONDS} and "
            f"{MAX_DURATION_SECONDS} seconds."
        )
    if duration_seconds != int(duration_seconds):
        raise ValueError("Duration must be a whole number of seconds.")


def save_waveform_as_wav(waveform, sample_rate: int, output_path: str | Path) -> Path:
    """Write a mono or stereo floating-point waveform as 16-bit PCM WAV."""
    import numpy as np

    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 1:
        channels = 1
    elif audio.ndim == 2 and audio.shape[1] in (1, 2):
        channels = audio.shape[1]
        if channels == 1:
            audio = audio[:, 0]
    else:
        raise ValueError("Audio must be a non-empty mono or stereo waveform.")

    if audio.shape[0] == 0 or sample_rate <= 0:
        raise ValueError("Audio waveform and sampling rate must be non-empty and valid.")

    pcm_audio = np.rint(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm_audio.tobytes())

    if not destination.is_file() or destination.stat().st_size < 44:
        raise OSError(f"WAV output was not written correctly: {destination}")
    with wave.open(str(destination), "rb") as wav_file:
        if (
            wav_file.getframerate() != sample_rate
            or wav_file.getnchannels() != channels
            or wav_file.getnframes() == 0
        ):
            raise OSError(f"WAV output failed its audio-header check: {destination}")
    return destination


class MusicGenerator:
    """A reusable inference-only MusicGen model wrapper."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._processor = None
        self._model = None
        self._torch = None
        self._device = None
        self._lock = threading.RLock()

    @property
    def device_name(self) -> str:
        """Return the selected device, loading PyTorch only when needed."""
        self._load_model()
        return str(self._device)

    def _load_model(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoProcessor, MusicgenForConditionalGeneration
            except ImportError as exc:
                raise RuntimeError(
                    "MusicGen dependencies are missing. Install the packages from "
                    "requirements.txt and install a PyTorch build for your device."
                ) from exc

            self._torch = torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if self._device.type == "cuda" else torch.float32
            try:
                processor = AutoProcessor.from_pretrained(self.model_id)
                model = MusicgenForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                )
                model.to(self._device)
                model.eval()
            except OSError as exc:
                raise RuntimeError(
                    f"Could not download or read {self.model_id}. Check the internet "
                    "connection, Hugging Face access, and available disk space, then retry."
                ) from exc
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        "There is not enough memory to load MusicGen. Use a Colab GPU "
                        "runtime, restart the runtime to free memory, or close other "
                        "memory-heavy applications."
                    ) from exc
                raise

            self._processor = processor
            self._model = model

    def generate(
        self,
        prompt: str,
        duration_seconds: int | float = DEFAULT_DURATION_SECONDS,
        output_path: str | Path = "musicgen_output.wav",
    ) -> GenerationResult:
        """Generate one sample and persist it as a playable WAV file."""
        validate_generation_request(prompt, duration_seconds)
        with self._lock:
            self._load_model()
            torch = self._torch
            try:
                inputs = self._processor(
                    text=[prompt.strip()],
                    padding=True,
                    return_tensors="pt",
                ).to(self._device)
                with torch.inference_mode():
                    audio_values = self._model.generate(
                        **inputs,
                        do_sample=True,
                        guidance_scale=3.0,
                        max_new_tokens=int(duration_seconds * TOKENS_PER_SECOND),
                    )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if self._device.type == "cuda":
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        "There is not enough memory to generate this sample. Try a "
                        "shorter duration or use a Colab GPU with more available memory."
                    ) from exc
                raise

            if audio_values.ndim != 3 or audio_values.shape[0] < 1:
                raise RuntimeError("MusicGen returned audio with an unexpected tensor shape.")
            waveform = audio_values[0].detach().to(device="cpu", dtype=torch.float32).numpy()
            if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
                raise RuntimeError("MusicGen returned an unsupported audio channel layout.")
            if waveform.shape[1] == 0:
                raise RuntimeError("MusicGen returned an empty audio waveform.")

            if waveform.shape[0] == 1:
                playable_waveform = waveform[0]
            else:
                playable_waveform = waveform.T

            sample_rate = int(self._model.config.audio_encoder.sampling_rate)
            saved_path = save_waveform_as_wav(
                playable_waveform,
                sample_rate,
                output_path,
            )
            return GenerationResult(
                path=saved_path,
                sample_rate=sample_rate,
                channels=int(waveform.shape[0]),
                duration_seconds=float(waveform.shape[1] / sample_rate),
                device=str(self._device),
            )
