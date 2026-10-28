from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from config import DEFAULT_CONFIG, MusicGenConfig


@dataclass(frozen=True)
class GeneratedAudio:
    prompt: str
    waveform: np.ndarray
    sampling_rate: int
    wav_path: Path
    duration_seconds: float


class MusicGenerator:
    """Load Meta MusicGen once and use it for inference-only text-to-music generation."""

    def __init__(self, config: MusicGenConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self._load_lock = threading.Lock()
        self._generation_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            try:
                from transformers import AutoProcessor, MusicgenForConditionalGeneration

                self.processor = AutoProcessor.from_pretrained(self.config.model_id)
                model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
                self.model = MusicgenForConditionalGeneration.from_pretrained(
                    self.config.model_id,
                    torch_dtype=model_dtype,
                    low_cpu_mem_usage=True,
                )
                self.model.to(self.device)
                self.model.eval()
            except ImportError as exc:
                raise RuntimeError(
                    "MusicGen requires compatible PyTorch and Hugging Face Transformers. "
                    "Install the project dependencies with `pip install -r requirements.txt`."
                ) from exc
            except (OSError, ValueError, RuntimeError) as exc:
                detail = str(exc)
                if self.device.type == "cuda" and (
                    "out of memory" in detail.lower() or "cuda" in detail.lower()
                ):
                    advice = (
                        " The GPU may not have enough memory; restart the runtime, select a GPU "
                        "with more memory, or use CPU inference (which is slower)."
                    )
                elif isinstance(exc, OSError):
                    advice = (
                        " Check network access to Hugging Face and verify that the model ID "
                        "facebook/musicgen-small is reachable."
                    )
                else:
                    advice = (
                        " Verify the installed PyTorch/Transformers versions and use the "
                        "provided requirements.txt in a fresh Python 3.10-3.12 environment."
                    )
                raise RuntimeError(f"Could not load {self.config.model_id}.{advice} Details: {detail}") from exc

    def generate(
        self,
        prompt: str,
        duration_seconds: float = DEFAULT_CONFIG.default_duration_seconds,
        output_dir: str | Path | None = None,
        filename: str | None = None,
    ) -> GeneratedAudio:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            raise ValueError("Enter a text prompt describing the music to generate.")
        duration = self.config.validate_duration(duration_seconds)
        self.load()

        output_folder = Path(output_dir or self.config.output_dir).expanduser()
        output_folder.mkdir(parents=True, exist_ok=True)
        safe_name = filename or self._prompt_filename(clean_prompt)
        if not safe_name.lower().endswith(".wav"):
            safe_name += ".wav"
        destination = output_folder / Path(safe_name).name

        assert self.processor is not None and self.model is not None
        frame_rate = getattr(self.model.config, "frame_rate", None)
        if frame_rate is None:
            frame_rate = getattr(self.model.config.decoder, "frame_rate", 50)
        max_new_tokens = max(1, math.ceil(duration * float(frame_rate)))
        try:
            inputs = self.processor(
                text=[clean_prompt],
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self._generation_lock, torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=self.config.guidance_scale,
                    max_new_tokens=max_new_tokens,
                )
        except torch.cuda.OutOfMemoryError as exc:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(
                "The GPU ran out of memory while generating audio. Try a shorter duration "
                "or restart the Colab runtime to clear GPU memory."
            ) from exc
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Music generation failed: {exc}") from exc

        sample_rate = int(self.model.config.audio_encoder.sampling_rate)
        waveform = self._to_mono_waveform(generated)
        if waveform.size == 0:
            raise RuntimeError("MusicGen returned an empty waveform; try another prompt or duration.")
        self.save_wav(destination, waveform, sample_rate)
        actual_duration = waveform.shape[0] / sample_rate
        return GeneratedAudio(clean_prompt, waveform, sample_rate, destination, actual_duration)

    @staticmethod
    def _to_mono_waveform(audio: torch.Tensor) -> np.ndarray:
        if not isinstance(audio, torch.Tensor):
            raise TypeError("The model output must be a PyTorch tensor.")
        audio = audio.detach().float().cpu()
        if audio.ndim == 3:
            audio = audio[0]
        if audio.ndim == 2:
            if audio.shape[0] in (1, 2):
                audio = audio.mean(dim=0)
            elif audio.shape[1] in (1, 2):
                audio = audio.mean(dim=1)
            else:
                raise ValueError(f"Cannot identify audio channels in output shape {tuple(audio.shape)}.")
        if audio.ndim != 1:
            raise ValueError(f"Expected a one-dimensional waveform, got {tuple(audio.shape)}.")
        return audio.clamp(-1, 1).numpy().astype(np.float32, copy=False)

    @staticmethod
    def save_wav(path: str | Path, waveform: np.ndarray, sampling_rate: int) -> Path:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be a positive integer.")
        samples = np.asarray(waveform, dtype=np.float32)
        if samples.ndim != 1 or samples.size == 0:
            raise ValueError("waveform must be a non-empty mono array.")
        if not np.isfinite(samples).all():
            raise ValueError("waveform contains NaN or infinite values.")
        samples = np.clip(samples, -1.0, 1.0)
        pcm = np.round(samples * 32767).astype(np.int16)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(destination, int(sampling_rate), pcm)
        return destination

    @staticmethod
    def _prompt_filename(prompt: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.lower()).strip("_")
        return f"{slug[:48] or 'musicgen_output'}.wav"


def create_generator(config: MusicGenConfig = DEFAULT_CONFIG) -> MusicGenerator:
    return MusicGenerator(config)
