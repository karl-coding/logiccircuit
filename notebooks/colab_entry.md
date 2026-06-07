# Colab Entry

This file is a lightweight notebook outline. Convert it to a Colab notebook or
copy the cells into Colab.

## 1. Install

```bash
pip install -q transformers accelerate peft bitsandbytes trl datasets pyyaml
```

## 2. Clone or Upload Project

Upload this project directory or clone it into Colab.

## 3. Baseline

Generate candidate solutions for `data/sample_tasks.jsonl`, then evaluate:

```bash
python -m src.evaluate \
  --tasks data/sample_tasks.jsonl \
  --candidates data/sample_candidates.jsonl \
  --output runs/sample_eval.jsonl
```

## 4. Candidate Generation

Use the base model to produce 4-8 candidates per task. Save them as:

```json
{"task_id":"task_id","rank":1,"code":"def fixed(...): ..."}
```

## 5. Verifier Filtering

Keep candidates that pass public and hidden tests. Train QLoRA on the passing
solutions and compare against a random-SFT control.

## 6. Final Validation

Run:

```text
base model
verifier-guided QLoRA
random-SFT control
```

Compare pass@1, pass@3, pass@8, hidden-test pass rate, variable-renaming tests,
and hard/transfer/adversarial splits.

