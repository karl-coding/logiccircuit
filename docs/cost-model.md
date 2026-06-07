# Cost Model

## Principle

With only 10 hours of Google Colab GPU, the correct goal is not pretraining.
The correct goal is local circuit strengthening.

## Why Not Pretrain

Pretraining even a small competitive model requires far more compute than 10
GPU hours. It also spends tokens on broad language modeling rather than the
target circuit.

## Lowest-Cost Useful Path

```text
open 1.5B-3B model
-> 4-bit QLoRA
-> verifier-guided data selection
-> short fine-tuning
-> held-out verification
```

## Cost Drivers

The practical cost is driven by:

```text
model size
sequence length
number of candidate generations
number of QLoRA steps
verifier runtime
evaluation size
```

Keep sequence length at 1024-2048 unless the target circuit requires longer
context.

## Expected Outcome

For a 10-hour run:

```text
single-domain improvement: possible, often 10%-40%
general intelligence improvement: unlikely
best domains: code repair, math microtasks, tool calls
```

The result should be judged by held-out and transfer behavior, not training
loss.

