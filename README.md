# Logic Circuit Fast-Insight Training

This project captures a low-cost training and validation plan for inducing local
"aha" behavior in small language models by strengthening weak logic circuits.

## Core Thesis

Model "insight" is not a whole-model awakening. It is usually a local circuit
crossing a usability threshold:

```text
weak, scattered feature activations
-> stable attention/MLP routing
-> reusable computation path
-> improved pass@1 and transfer behavior
```

The practical goal is to find a weak circuit that already exists, activate it
dense enough, reward it accurately, and prevent shortcut circuits from taking
the reward.

## Minimum-Cost Direction

With only about 10 hours of Google Colab T4 time, do not pretrain a model.
Use a 1.5B open model and QLoRA to strengthen one local capability. On T4, keep
the first run under 2 GPU hours.

Recommended models:

- `DeepSeek-R1-Distill-Qwen-1.5B`
- `Qwen/Qwen2.5-1.5B-Instruct`

Recommended first target:

```text
code repair + unit-test verifier
```

This gives the clearest reward signal and the highest chance of observing a
real local circuit improvement within a short T4 budget.

## Key Documents

- `docs/theory.md`: What changes inside the model during insight.
- `docs/training-plan.md`: Lowest-cost training method.
- `docs/validation-plan.md`: How to prove the improvement is real.
- `docs/cost-model.md`: Cost logic and why small-model QLoRA is the right path.
- `docs/t4-runbook.md`: T4 runtime and validation plan.

## Runnable Scaffold

The project includes a minimal code-repair verifier and evaluator:

```bash
python -m src.evaluate \
  --tasks data/sample_tasks.jsonl \
  --candidates data/sample_candidates.jsonl \
  --output runs/sample_eval.jsonl
```

Generate model prompts:

```bash
python -m src.make_prompts \
  --tasks data/sample_tasks.jsonl \
  --output runs/sample_prompts.jsonl
```

Filter passing candidates into SFT/QLoRA rows:

```bash
python -m src.filter_candidates \
  --tasks data/sample_tasks.jsonl \
  --candidates data/sample_candidates.jsonl \
  --output runs/sft_train.jsonl
```

Generate real model candidates on T4:

```bash
python -m src.make_tasks --output data/tasks.jsonl

python -m src.generate_candidates \
  --tasks data/tasks.jsonl \
  --output runs/base_candidates.jsonl \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --candidates-per-task 2 \
  --splits train \
  --max-tasks 32 \
  --max-new-tokens 256
```

Train a QLoRA adapter:

```bash
python -m src.train_qlora \
  --train runs/sft_train.jsonl \
  --output-dir runs/deepseek_r1_qwen_1p5b_adapter \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --max-seq-length 1024 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --max-steps 35
```

Evaluate the trained adapter by generating candidates and reusing the verifier:

```bash
python -m src.generate_with_adapter \
  --tasks data/tasks.jsonl \
  --output runs/adapter_candidates.jsonl \
  --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --adapter-dir runs/deepseek_r1_qwen_1p5b_adapter \
  --candidates-per-task 2 \
  --max-new-tokens 256
```

The scripts use standard Python plus the lightweight packages in
`requirements.txt`. Model training dependencies are intentionally left for the
Colab runtime described in `notebooks/colab_entry.md`.

This project is now standardized on T4. Check the active Colab runtime before
training:

```bash
python -m src.runtime_check --config configs/t4_10h.yaml
```

Back up important Colab artifacts to Google Drive with the same folder layout:

```bash
python -m src.backup_to_drive \
  --project-root /content/logiccircuit \
  --drive-root /content/drive/MyDrive/logiccircuit
```

## Success Criteria

A run is useful only if held-out behavior improves, not just training loss.

Minimum acceptance:

```text
test-similar pass@1 improvement >= 10%
test-hard pass@1 improvement >= 8%
variable-renaming degradation <= 5%
hidden tests improve with public tests
verifier-guided QLoRA beats random-SFT control
```
