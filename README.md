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

With only about 10 hours of Google Colab GPU time, do not pretrain a model.
Use a 1.5B-3B open model and QLoRA to strengthen one local capability.

Recommended models:

- `DeepSeek-R1-Distill-Qwen-1.5B`
- `Qwen2.5-3B-Instruct`
- `Qwen2.5-Coder-3B`

Recommended first target:

```text
code repair + unit-test verifier
```

This gives the clearest reward signal and the highest chance of observing a
real local circuit improvement in 10 GPU hours.

## Key Documents

- `docs/theory.md`: What changes inside the model during insight.
- `docs/training-plan.md`: Lowest-cost training method.
- `docs/validation-plan.md`: How to prove the improvement is real.
- `docs/cost-model.md`: Cost logic and why small-model QLoRA is the right path.
- `docs/a100-runbook.md`: A100-only runtime and validation plan.

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

Generate real model candidates on A100:

```bash
python -m src.generate_candidates \
  --tasks data/tasks.jsonl \
  --output runs/base_candidates.jsonl \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --candidates-per-task 8
```

Train a QLoRA adapter:

```bash
python -m src.train_qlora \
  --train runs/sft_train.jsonl \
  --output-dir runs/qwen_coder_3b_adapter \
  --model Qwen/Qwen2.5-Coder-3B-Instruct
```

Evaluate the trained adapter by generating candidates and reusing the verifier:

```bash
python -m src.generate_with_adapter \
  --tasks data/tasks.jsonl \
  --output runs/adapter_candidates.jsonl \
  --base-model Qwen/Qwen2.5-Coder-3B-Instruct \
  --adapter-dir runs/qwen_coder_3b_adapter \
  --candidates-per-task 8
```

The scripts use standard Python plus the lightweight packages in
`requirements.txt`. Model training dependencies are intentionally left for the
Colab runtime described in `notebooks/colab_entry.md`.

This project is now standardized on A100. Check the active Colab runtime before
training:

```bash
python -m src.runtime_check --config configs/a100_10h.yaml
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
