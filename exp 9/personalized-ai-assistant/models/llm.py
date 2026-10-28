from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import DEFAULT_MODEL


class LLMManager:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def load(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
        except Exception as exc:  # pragma: no cover - runtime safety
            raise RuntimeError(f"Unable to load model '{self.model_name}': {exc}") from exc

    def generate(self, prompt: str, max_new_tokens: int = 180, temperature: float = 0.7):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return decoded.strip()

    def unload(self):
        self.model = None
        self.tokenizer = None
