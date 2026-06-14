# T4 Budget Runbook

## Goal

Run a low-cost code-repair circuit-strengthening experiment on a Google Colab
T4 runtime and get a useful signal before GPU time expires.

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

Default budget limits:

```text
model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
fallback: Qwen/Qwen2.5-1.5B-Instruct
sequence length: 1024
LoRA rank: 8
candidates per task: 2
training rows: 40-120
training steps: 30-40
expected runtime: 45-120 minutes
hard stop: 2.5 GPU hours
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

Verify 4-bit support before running generation:

```bash
python - <<'PY'
import bitsandbytes as bnb
print(bnb.__version__)
PY
```

If `bitsandbytes` keeps failing on T4, use `--no-4bit` for the 1.5B model:

```bash
python -m src.generate_candidates \
  --tasks data/tasks.jsonl \
  --output runs/base_candidates_smoke.jsonl \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --candidates-per-task 2 \
  --splits train \
  --max-tasks 32 \
  --max-new-tokens 256 \
  --no-4bit
```

Mount Google Drive before long runs:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Create tasks:

```bash
python -m src.make_tasks --output data/tasks.jsonl --repeat 2
```

## Budget Loop

Generate train candidates:

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
  --max-steps 35
```

If 4-bit training fails for dependency reasons, retry the same command with:

```bash
  --no-4bit
```

Generate adapter candidates and compare:

```bash
python -m src.generate_with_adapter \
  --tasks data/tasks.jsonl \
  --output runs/adapter_candidates_smoke.jsonl \
  --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter-dir runs/deepseek_r1_qwen_1p5b_smoke_adapter \
  --candidates-per-task 2 \
  --splits train \
  --max-tasks 32 \
  --max-new-tokens 256

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

## Held-Out Mini Check

Run this before attempting a larger experiment:

```bash
python -m src.generate_candidates \
  --tasks data/tasks.jsonl \
  --output runs/base_candidates_heldout_t4_mini.jsonl \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --candidates-per-task 2 \
  --splits test-similar test-hard test-transfer \
  --max-tasks 48 \
  --max-new-tokens 256

python -m src.generate_with_adapter \
  --tasks data/tasks.jsonl \
  --output runs/adapter_candidates_heldout_t4_mini.jsonl \
  --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter-dir runs/deepseek_r1_qwen_1p5b_smoke_adapter \
  --candidates-per-task 2 \
  --splits test-similar test-hard test-transfer \
  --max-tasks 48 \
  --max-new-tokens 256

python -m src.evaluate \
  --tasks data/tasks.jsonl \
  --candidates runs/base_candidates_heldout_t4_mini.jsonl \
  --output runs/base_eval_heldout_t4_mini.jsonl \
  --splits test-similar test-hard test-transfer \
  --only-with-candidates \
  --k 1 2

python -m src.evaluate \
  --tasks data/tasks.jsonl \
  --candidates runs/adapter_candidates_heldout_t4_mini.jsonl \
  --output runs/adapter_eval_heldout_t4_mini.jsonl \
  --splits test-similar test-hard test-transfer \
  --only-with-candidates \
  --k 1 2

python -m src.compare_eval \
  --baseline runs/base_eval_heldout_t4_mini.jsonl \
  --adapter runs/adapter_eval_heldout_t4_mini.jsonl
```

## Stop Rules

Stop and back up when any one of these is true:

```text
loss is below 0.05 for two logs
training reaches 35 steps
candidate generation exceeds 45 minutes
filtered SFT rows are below 10
held-out pass@1 does not improve on the mini check
```
