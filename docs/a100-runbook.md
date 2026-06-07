# A100 10-Hour Runbook

## Goal

Run a timely code-repair circuit-strengthening experiment on a Google Colab
A100 runtime and verify whether the improvement is real.

## Fixed Hardware Assumption

This project uses one profile only:

```text
Google Colab A100, about 40 GB VRAM
```

Do not branch to T4 settings. If the runtime check does not report A100-ready,
switch the Colab runtime before training.

## A100 Profile

Use:

```text
configs/a100_10h.yaml
```

Default limits:

```text
model: Qwen/Qwen2.5-Coder-3B-Instruct
fallback: DeepSeek-R1-Distill-Qwen-1.5B
sequence length: 1536
LoRA rank: 16
candidates per task: 8
training rows: 1000-3000
expected runtime: 4-8 GPU hours
hard stop: 10 GPU hours
```

## First Commands

Check runtime:

```bash
python -m src.runtime_check --config configs/a100_10h.yaml
```

Run verifier smoke test:

```bash
python -m src.evaluate \
  --tasks data/sample_tasks.jsonl \
  --candidates data/sample_candidates.jsonl \
  --output runs/sample_eval.jsonl
```

Build verified SFT rows:

```bash
python -m src.filter_candidates \
  --tasks data/sample_tasks.jsonl \
  --candidates data/sample_candidates.jsonl \
  --output runs/sft_train.jsonl
```

## Real Model Loop

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate base-model candidates. Start with a 20-task smoke run:

```bash
python -m src.generate_candidates \
  --tasks data/tasks.jsonl \
  --output runs/base_candidates_smoke.jsonl \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --candidates-per-task 8 \
  --max-tasks 20
```

Evaluate base candidates:

```bash
python -m src.evaluate \
  --tasks data/tasks.jsonl \
  --candidates runs/base_candidates_smoke.jsonl \
  --output runs/base_eval_smoke.jsonl
```

Filter passing candidates into supervised rows:

```bash
python -m src.filter_candidates \
  --tasks data/tasks.jsonl \
  --candidates runs/base_candidates_smoke.jsonl \
  --output runs/sft_train_smoke.jsonl
```

Train a short QLoRA smoke adapter:

```bash
python -m src.train_qlora \
  --train runs/sft_train_smoke.jsonl \
  --output-dir runs/qwen_coder_3b_smoke_adapter \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --max-steps 20
```

Generate candidates from the adapter:

```bash
python -m src.generate_with_adapter \
  --tasks data/tasks.jsonl \
  --output runs/adapter_candidates_smoke.jsonl \
  --base-model Qwen/Qwen2.5-Coder-3B-Instruct \
  --adapter-dir runs/qwen_coder_3b_smoke_adapter \
  --candidates-per-task 8 \
  --max-tasks 20
```

Evaluate adapter candidates:

```bash
python -m src.evaluate \
  --tasks data/tasks.jsonl \
  --candidates runs/adapter_candidates_smoke.jsonl \
  --output runs/adapter_eval_smoke.jsonl
```

Compare baseline and adapter:

```bash
python -m src.compare_eval \
  --baseline runs/base_eval_smoke.jsonl \
  --adapter runs/adapter_eval_smoke.jsonl
```

If the smoke run completes, remove `--max-tasks 20` and `--max-steps 20` for
the full run. Keep the 10-hour hard stop.

## Timely Validation Gates

Stop early if:

```text
runtime_check says a100_ready: False
baseline evaluation takes more than 45 minutes
candidate generation takes more than 3 hours
candidate verification takes more than 90 minutes
filtered data has fewer than 1000 passing rows
post-train eval takes more than 90 minutes
```

## Success Criteria

Minimum useful result:

```text
test-similar pass@1 improves by >= 10%
test-hard pass@1 improves by >= 8%
test-transfer pass@1 improves by >= 5%
hidden-test pass rate improves with public-test pass rate
variable-renaming degradation <= 5%
verifier-guided QLoRA beats random-SFT control
```

## Interpretation

If only training or similar tasks improve, the run likely learned a task format.
If similar, hard, transfer, and hidden tests improve together, the result is
evidence of local logic-circuit strengthening.
