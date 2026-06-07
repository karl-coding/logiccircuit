from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .io_utils import read_jsonl, write_jsonl
from .make_prompts import build_prompt
from .task_schema import CodeRepairTask


CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    match = CODE_BLOCK_PATTERN.search(text)
    if match:
        return match.group(1).strip()

    lines = text.strip().splitlines()
    code_lines: list[str] = []
    started = False
    for line in lines:
        if line.strip().startswith(("def ", "class ", "import ", "from ")):
            started = True
        if started:
            code_lines.append(line)
    return "\n".join(code_lines).strip() if code_lines else text.strip()


def format_chat_prompt(tokenizer: AutoTokenizer, task: CodeRepairTask) -> str:
    prompt = build_prompt(task)
    messages = [
        {
            "role": "system",
            "content": "You fix Python functions. Return only valid Python code, with no explanation.",
        },
        {"role": "user", "content": prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + "\n\nFixed code:\n"


def load_model(model_name: str, load_in_4bit: bool) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
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
    model.eval()
    return tokenizer, model


def generate_candidates(
    tasks_path: Path,
    output_path: Path,
    model_name: str,
    candidates_per_task: int,
    max_tasks: int | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    load_in_4bit: bool,
) -> int:
    tokenizer, model = load_model(model_name, load_in_4bit)
    tasks = [CodeRepairTask.from_json(row) for row in read_jsonl(tasks_path)]
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

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
                    "model": model_name,
                }
            )

        print(f"generated {candidates_per_task} candidates for {task.id} ({index}/{len(tasks)})")

    write_jsonl(output_path, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--candidates-per-task", default=8, type=int)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-new-tokens", default=512, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    count = generate_candidates(
        tasks_path=args.tasks,
        output_path=args.output,
        model_name=args.model,
        candidates_per_task=args.candidates_per_task,
        max_tasks=args.max_tasks,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        load_in_4bit=not args.no_4bit,
    )
    print(f"wrote {count} candidates to {args.output}")


if __name__ == "__main__":
    main()

