from __future__ import annotations

import json
from pathlib import Path

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


class FineTuningSetup:
    def __init__(self, base_model_name: str, output_dir: str = "./adapter-output"):
        self.base_model_name = base_model_name
        self.output_dir = Path(output_dir)
        self.config = None

    def validate_dataset(self, dataset_path: str | Path):
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Training dataset not found: {path}")
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "instruction" not in item or "response" not in item:
                    raise ValueError("Each JSONL row must include 'instruction' and 'response'.")
                rows.append(item)
        if not rows:
            raise ValueError("Training dataset is empty.")
        return rows

    def build_lora_config(self, rank: int = 8, alpha: int = 16, dropout: float = 0.05):
        self.config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        return self.config

    def prepare_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        model = get_peft_model(model, self.config)
        return tokenizer, model

    def train(self, dataset_path: str | Path):
        rows = self.validate_dataset(dataset_path)
        tokenizer, model = self.prepare_model()
        train_texts = [f"Instruction: {row['instruction']}\nResponse: {row['response']}" for row in rows]
        tokenized = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")

        class SimpleDataset:
            def __init__(self, features):
                self.features = features

            def __len__(self):
                return len(self.features["input_ids"])

            def __getitem__(self, idx):
                return {key: value[idx] for key, value in self.features.items()}

        train_dataset = SimpleDataset(tokenized)
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=1,
            num_train_epochs=1,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
        trainer.train()
        model.save_pretrained(str(self.output_dir))
        return trainer
