from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from .io_utils import read_jsonl


IGNORE_INDEX = -100


def build_training_text(prompt: str, response: str) -> tuple[str, str]:
    prompt_text = prompt.rstrip() + "\n\nFixed code:\n"
    response_text = response.strip()
    return prompt_text, response_text


class SftJsonlDataset(Dataset):
    def __init__(self, path: Path, tokenizer: AutoTokenizer, max_seq_length: int):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        prompt, response = build_training_text(str(row["prompt"]), str(row["response"]))
        full_text = prompt + response + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class DataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in features]
        attention_mask = [item["attention_mask"] for item in features]
        labels = [item["labels"] for item in features]

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        padded_attention = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        return {
            "input_ids": padded_inputs,
            "attention_mask": padded_attention,
            "labels": padded_labels,
        }


def train_qlora(
    train_path: Path,
    output_dir: Path,
    model_name: str,
    load_in_4bit: bool,
    max_seq_length: int,
    learning_rate: float,
    epochs: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    max_steps: int,
) -> None:
    if not train_path.exists():
        raise FileNotFoundError(
            f"training file not found: {train_path}. "
            "Run src.filter_candidates first to create SFT rows."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = SftJsonlDataset(train_path, tokenizer, max_seq_length)
    if len(dataset) == 0:
        raise ValueError(f"no training rows found in {train_path}")

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-seq-length", default=1024, type=int)
    parser.add_argument("--learning-rate", default=2e-4, type=float)
    parser.add_argument("--epochs", default=2.0, type=float)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=8, type=int)
    parser.add_argument("--lora-rank", default=8, type=int)
    parser.add_argument("--lora-alpha", default=16, type=int)
    parser.add_argument("--lora-dropout", default=0.05, type=float)
    parser.add_argument("--max-steps", default=-1, type=int)
    args = parser.parse_args()

    train_qlora(
        train_path=args.train,
        output_dir=args.output_dir,
        model_name=args.model,
        load_in_4bit=not args.no_4bit,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
