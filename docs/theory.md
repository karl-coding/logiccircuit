# Theory: What Changes During Model Insight

## Essence

"Insight" is a local circuit transition, not a global intelligence jump.

```text
feature activation pattern
+ attention routing
+ MLP transformations
+ multi-layer circuit connectivity
-> stable reusable computation
```

The external symptom is sudden benchmark or task improvement. The internal
change is more gradual: weak structure becomes strong enough to be called
reliably.

## What Specifically Changes

### 1. Representation Geometry

Before insight, examples are often organized by surface similarity.

After insight, they become organized by mechanism:

```text
surface-similar clustering
-> mechanism-similar clustering
```

For example, math, code variables, pronoun tracking, and table references can
start sharing a variable-binding circuit.

### 2. Features Become Cleaner

Weak models often mix roles:

```text
number = year + quantity + index + math operand + code line
```

Training can separate these:

```text
year
quantity
math operand
list index
code index
```

This reduces wrong circuit activation.

### 3. Attention Routes Stabilize

Weak behavior:

```text
current token -> nearby tokens
```

Strong behavior:

```text
answer position -> relevant premise
variable use -> variable definition
closing bracket -> opening bracket
conclusion -> constraints
```

### 4. MLP Blocks Act Like Reusable Operators

The model shifts from pattern completion to state transformation:

```text
input pattern -> common output
```

becomes:

```text
input state -> operation -> updated state
```

### 5. Multi-Layer Circuit Connectivity

A useful logic circuit can look like:

```text
recognize entities
-> bind variables
-> propagate constraints
-> update state
-> verify consistency
-> emit answer
```

Insight occurs when this path becomes stable and reusable.

## Local, Not Global

Insight normally affects a local circuit:

```text
addition carry
variable binding
state tracking
constraint propagation
tool-call planning
error checking
```

If the circuit is foundational, the local gain can transfer across tasks and
look like broader intelligence improvement.

## Main Acceleration Factors

Circuit strengthening speed depends on:

```text
activation density
* verifier quality
* exploration success
* credit assignment accuracy
* transfer diversity
/ shortcut competition
```

The most important practical factors:

1. A weak seed circuit already exists.
2. Task difficulty is in the critical zone, around 20%-60% base success.
3. The reward is verifiable.
4. Training data densely activates the target circuit.
5. Shortcut circuits are blocked with adversarial and counterfactual examples.

