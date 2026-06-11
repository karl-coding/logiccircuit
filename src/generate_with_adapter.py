from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .generate_candidates import extract_code, format_chat_prompt
from .io_utils import read_jsonl, write_jsonl
from .task_schema import CodeRepairTask


def load_adapter_model(
    base_model: str,
    adapter_dir: Path,
) -> tuple[AutoTokenizer, PeftModel]:
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tokenizer, model


def generate_with_adapter(
    tasks_path: Path,
    output_path: Path,
    base_model: str,
    adapter_dir: Path,
    candidates_per_task: int,
    max_tasks: int | None,
    splits: set[str] | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> int:
    if not tasks_path.exists():
        raise FileNotFoundError(
            f"task file not found: {tasks_path}. "
            "Run src.make_tasks first or provide an existing JSONL task file."
        )
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"adapter directory not found: {adapter_dir}. Run src.train_qlora first."
        )

    tasks = [CodeRepairTask.from_json(row) for row in read_jsonl(tasks_path)]
    if splits is not None:
        tasks = [task for task in tasks if task.split in splits]
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    if not tasks:
        raise ValueError(f"no tasks selected from {tasks_path}")

    tokenizer, model = load_adapter_model(base_model, adapter_dir)

    rows: list[dict[str, object]] = []
    for index, task in enumerate(tasks, start=1):
        prompt = format_chat_prompt(tokenizer, task)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = int(inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                num_return_sequences=candidates_per_task,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for rank, output_ids in enumerate(outputs, start=1):
            generated = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
            rows.append(
                {
                    "task_id": task.id,
                    "rank": rank,
                    "code": extract_code(generated),
                    "raw_response": generated.strip(),
                    "base_model": base_model,
                    "adapter_dir": str(adapter_dir),
                }
            )
        print(f"generated {candidates_per_task} adapter candidates for {task.id} ({index}/{len(tasks)})")

    write_jsonl(output_path, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--candidates-per-task", default=8, type=int)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--max-new-tokens", default=512, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    args = parser.parse_args()

    count = generate_with_adapter(
        tasks_path=args.tasks,
        output_path=args.output,
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        candidates_per_task=args.candidates_per_task,
        max_tasks=args.max_tasks,
        splits=set(args.splits) if args.splits else None,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"wrote {count} adapter candidates to {args.output}")


if __name__ == "__main__":
    main()
