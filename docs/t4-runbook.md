# T4 10-Hour Runbook

## Goal

Run a low-cost code-repair circuit-strengthening experiment on a Google Colab
T4 runtime and verify whether the improvement is real.

## Fixed Hardware Assumption

This project now uses one profile:

```text
Google Colab T4, about 16 GB VRAM
```

Do not use the 3B A100 commands on T4. Use the 1.5B model and smaller candidate
counts first.

## T4 Profile

Use:

```text
configs/t4_10h.yaml
```

Default limits:

```text
model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
fallback: Qwen/Qwen2.5-1.5B-Instruct
sequence length: 1024
LoRA rank: 8
candidates per task: 4
training rows: 300-800
expected runtime: 4-8 GPU hours
hard stop: 10 GPU hours
```

## First Commands

Check runtime:

```bash
python -m src.runtime_check --config configs/t4_10h.yaml
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -U "bitsandbytes>=0.46.1"
```

Mount Google Drive before long runs:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Create tasks:

```bash
python -m src.make_tasks --output data/tasks.jsonl --repeat 3
```

## Smoke Loop

Generate train candidates:

```bash
python -m src.generate_candidates \
  --tasks data/tasks.jsonl \
  --output runs/base_candidates_smoke.jsonl \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --candidates-per-task 4 \
  --splits train \
  --max-tasks 20
```

Evaluate and filter:

```bash
python -m src.evaluate \
  --tasks data/tasks.jsonl \
  --candidates runs/base_candidates_smoke.jsonl \
  --output runs/base_eval_smoke.jsonl \
  --splits train \
  --only-with-candidates

python -m src.filter_candidates \
  --tasks data/tasks.jsonl \
  --candidates runs/base_candidates_smoke.jsonl \
  --output runs/sft_train_smoke.jsonl \
  --splits train
```

Train a short adapter:

```bash
python -m src.train_qlora \
  --train runs/sft_train_smoke.jsonl \
  --output-dir runs/deepseek_r1_qwen_1p5b_smoke_adapter \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --max-seq-length 1024 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --max-steps 20
```

Generate adapter candidates and compare:

```bash
python -m src.generate_with_adapter \
  --tasks data/tasks.jsonl \
  --output runs/adapter_candidates_smoke.jsonl \
  --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter-dir runs/deepseek_r1_qwen_1p5b_smoke_adapter \
  --candidates-per-task 4 \
  --splits train \
  --max-tasks 20

python -m src.evaluate \
  --tasks data/tasks.jsonl \
  --candidates runs/adapter_candidates_smoke.jsonl \
  --output runs/adapter_eval_smoke.jsonl \
  --splits train \
  --only-with-candidates

python -m src.compare_eval \
  --baseline runs/base_eval_smoke.jsonl \
  --adapter runs/adapter_eval_smoke.jsonl
```

Back up artifacts:

```bash
python -m src.backup_to_drive \
  --project-root /content/logiccircuit \
  --drive-root /content/drive/MyDrive/logiccircuit
```

