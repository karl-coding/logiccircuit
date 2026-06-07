# Validation Plan

## Objective

Prove that a local logic circuit strengthened, rather than merely improving
training loss, memorizing examples, or learning output formatting.

## Evaluation Splits

```text
train: 800-1500 tasks
dev: 100-200 tasks
test-similar: 200 tasks
test-hard: 100 tasks
test-transfer: 100 tasks
test-adversarial: 50 tasks
```

No test item may appear in training, candidate filtering, or prompt tuning.

## Core Metrics

For code repair:

```text
pass@1
pass@3
pass@8
hidden-test pass rate
error-type distribution
```

The most important metric is pass@1. pass@8 can improve through exploration
alone; pass@1 indicates a more stable circuit.

## Controls

Use at least three models:

```text
base model
verifier-guided QLoRA model
random-SFT control model
```

The random-SFT control should use the same number of examples and training
steps, but without verifier-selected correct trajectories.

## Anti-Shortcut Checks

### Format Removal

Train with explanations if useful, then test with code-only output. If the gain
disappears, the model may have learned format rather than logic.

### Variable Renaming

Randomize names:

```text
nums, target, result
-> a, q, z
```

Large degradation means the model is relying on surface patterns.

### Minimal Counterfactuals

Change one condition:

```text
empty list returns 0
-> empty list returns -1
```

The model must update behavior accordingly.

### Hidden Tests

Add tests for:

```text
empty input
boundary values
duplicates
negative numbers
large numbers
ordering changes
type edge cases
```

## Acceptance Criteria

```text
test-similar pass@1 improvement >= 10%
test-hard pass@1 improvement >= 8%
test-transfer pass@1 improvement >= 5%-8%
variable-renaming degradation <= 5%
hidden tests improve with public tests
verifier-guided QLoRA beats random-SFT
```

## Result Interpretation

| Observation | Interpretation |
| --- | --- |
| train improves, test does not | memorization or overfitting |
| similar improves, hard does not | task-pattern learning |
| hard improves, transfer does not | useful local repair |
| similar/hard/transfer improve | circuit strengthening likely |
| adversarial drops sharply | shortcut circuit likely |
| pass@8 improves, pass@1 does not | exploration improved, stability did not |
| pass@1 and hidden tests improve | strong evidence of real gain |

