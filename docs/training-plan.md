# Training Plan: 10-Hour Colab Circuit Strengthening

## Goal

Strengthen one local logic circuit in a small model within about 10 GPU hours.

Recommended target:

```text
code repair with unit-test verification
```

This is preferred because unit tests provide cheap, precise rewards.

## Model

Preferred on T4:

```text
DeepSeek-R1-Distill-Qwen-1.5B
```

Alternatives:

```text
Qwen/Qwen2.5-1.5B-Instruct
Llama-3.2-3B
```

Use 3B models only on larger GPUs or when runtime budget is less constrained.

Use 4-bit QLoRA.

## Training Recipe

```text
1. Build 800-1500 train tasks and 300-500 held-out test tasks.
2. Run baseline evaluation on held-out sets.
3. For each train task, sample 4-8 candidate answers.
4. Run unit tests as verifier.
5. Keep only passing or diagnostically useful trajectories.
6. Train QLoRA on selected correct trajectories.
7. Add hard negatives and counterfactual examples for repeated failures.
8. Run final evaluation against baseline and random-SFT control.
```

## Time Budget

```text
0-1h: environment, model load, baseline eval
1-3h: candidate generation and verifier filtering
3-7h: QLoRA training
7-8h: first eval
8-9h: hard-negative patch set
9-10h: short second training run and final eval
```

## Suggested QLoRA Settings

```text
quantization: 4-bit
LoRA rank: 16 or 32
learning rate: 1e-4 to 2e-4
max length: 1024 to 2048
epochs: 1 to 3
batch size: small batch + gradient accumulation
```

## Data Format

Code repair example:

```json
{
  "instruction": "Fix the Python function so it passes the tests.",
  "input": "def add_digits(n): ...",
  "tests": "assert add_digits(123) == 6",
  "output": "def add_digits(n):\n    return sum(int(c) for c in str(abs(n)))"
}
```

## Why This Is Low Cost

```text
small model -> low GPU requirement
QLoRA -> low memory
unit tests -> cheap verifier
multi-sampling -> finds weak existing circuits
filtered training -> strengthens correct paths
hard negatives -> blocks shortcut circuits
```
