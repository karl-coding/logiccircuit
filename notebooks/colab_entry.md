# Colab Entry

This file is a lightweight notebook outline. Convert it to a Colab notebook or
copy the cells into Colab.

## 1. Install

```bash
pip install -q -r requirements.txt
```

## 2. Clone or Upload Project

Upload this project directory or clone it into Colab.

## 3. Baseline

Create a starter task file:

```bash
python -m src.make_tasks --output data/tasks.jsonl
```

Generate candidate solutions for `data/sample_tasks.jsonl`, then evaluate:

```bash
python -m src.evaluate \
  --tasks data/sample_tasks.jsonl \
  --candidates data/sample_candidates.jsonl \
  --output runs/sample_eval.jsonl
```

## 4. Candidate Generation

Use the base model to produce 2 candidates per task on T4. Save them as:

```json
{"task_id":"task_id","rank":1,"code":"def fixed(...): ..."}
```

T4 smoke command:

```bash
python -m src.generate_candidates \
  --tasks data/tasks.jsonl \
  --output runs/base_candidates_smoke.jsonl \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --candidates-per-task 2 \
  --splits train \
  --max-tasks 32 \
  --max-new-tokens 256
```

## 5. Verifier Filtering

Keep candidates that pass public and hidden tests. Train QLoRA on the passing
solutions and compare against a random-SFT control.

```bash
python -m src.filter_candidates \
  --tasks data/tasks.jsonl \
  --candidates runs/base_candidates_smoke.jsonl \
  --output runs/sft_train_smoke.jsonl \
  --splits train

python -m src.train_qlora \
  --train runs/sft_train_smoke.jsonl \
  --output-dir runs/deepseek_r1_qwen_1p5b_smoke_adapter \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --max-seq-length 1024 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --max-steps 35
```

## 6. Final Validation

Run:

```text
base model
verifier-guided QLoRA
random-SFT control
```

Compare pass@1, pass@3, pass@8, hidden-test pass rate, variable-renaming tests,
and hard/transfer/adversarial splits.
