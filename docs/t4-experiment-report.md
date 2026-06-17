# T4 Experiment Report

## Summary

The best checkpoint from the T4 budget experiments is:

```text
runs/qwen_coder_1p5b_t4_patch_adapter
```

This adapter gave the strongest broad held-out performance while also improving
the adversarial split.

## Hardware And Model

```text
GPU: Google Colab T4
Base model: Qwen/Qwen2.5-Coder-1.5B-Instruct
Training mode: LoRA, no 4-bit
LoRA rank: 8
LoRA alpha: 16
Max sequence length: 1024
Candidates per task: 2
```

The `--no-4bit` path was used because Colab repeatedly had `bitsandbytes` and
`torchao` package conflicts.

## Baseline

Evaluation file:

```text
runs/base_eval_adversarial_t4.jsonl
```

Baseline metrics:

```text
test-similar      pass@1 0.5625 | pass@2 0.7500
test-hard         pass@1 0.8125 | pass@2 0.8125
test-transfer     pass@1 0.8125 | pass@2 0.8750
test-adversarial  pass@1 0.6875 | pass@2 0.7500
```

## Best Adapter: Hard Patch

Adapter:

```text
runs/qwen_coder_1p5b_t4_patch_adapter
```

Evaluation file:

```text
runs/adapter_eval_adversarial_t4_patch.jsonl
```

Metrics:

```text
test-similar      pass@1 1.0000 | pass@2 1.0000
test-hard         pass@1 1.0000 | pass@2 1.0000
test-transfer     pass@1 0.9375 | pass@2 1.0000
test-adversarial  pass@1 0.8125 | pass@2 0.8125
```

Delta vs baseline:

```text
test-similar      pass@1 +0.4375 | pass@2 +0.2500
test-hard         pass@1 +0.1875 | pass@2 +0.1875
test-transfer     pass@1 +0.1250 | pass@2 +0.1250
test-adversarial  pass@1 +0.1250 | pass@2 +0.0625
```

Decision:

```text
Adopt as current best checkpoint.
```

## Remaining-Focus Micro Patch

Adapter:

```text
runs/qwen_coder_1p5b_t4_remaining_patch_adapter
```

Metrics:

```text
test-similar      pass@1 0.8750 | pass@2 0.9375
test-hard         pass@1 1.0000 | pass@2 1.0000
test-transfer     pass@1 1.0000 | pass@2 1.0000
test-adversarial  pass@1 0.8125 | pass@2 0.8125
```

Decision:

```text
Do not adopt over the hard-patch adapter.
```

Reason: it improves transfer but regresses `test-similar` compared with the
hard-patch adapter, while adversarial performance does not improve.

## State Curriculum Adapter

Adapter:

```text
runs/qwen_coder_1p5b_t4_state_curriculum_adapter
```

Metrics:

```text
test-similar      pass@1 0.8125 | pass@2 0.9375
test-hard         pass@1 0.8750 | pass@2 1.0000
test-transfer     pass@1 0.8750 | pass@2 0.9375
test-adversarial  pass@1 0.8125 | pass@2 0.8750
```

Decision:

```text
Do not adopt over the hard-patch adapter.
```

Reason: it partially improves `state_update_previous_value`, but broad
performance is weaker than the hard-patch adapter.

## Failure Modes

The hard-patch and curriculum runs revealed these remaining weak areas:

```text
reversed_bounds
state_update_previous_value
boundary_negative_number on some similar/transfer variants
```

The most persistent issue is `state_update_previous_value`: one adversarial
running-difference task improved, but the second variant remained unresolved.

## Interpretation

The T4 experiment gives evidence of local circuit strengthening:

```text
1. pass@1 improved on all held-out splits
2. adversarial pass@1 improved
3. training completed within a short T4 budget
4. bug-type analysis showed targeted gains rather than only train-set fitting
```

The result is not a general reasoning breakthrough. It is a localized code
repair improvement around boundary handling, state updates, and selected
anti-shortcut cases.

## Next Step

Do not add more generic SFT rows. The next useful experiment is a redesigned
state-machine task family with held-out tasks that vary:

```text
previous value update timing
initial previous value
None vs 0 initial states
tuple/list state variables
multi-variable state updates
```

Acceptance condition for the next run:

```text
state_update_previous_value pass@1 improves without lowering
test-similar or test-transfer pass@1 below the hard-patch adapter.
```

