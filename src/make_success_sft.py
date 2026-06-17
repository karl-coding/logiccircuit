from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import write_jsonl


def sft_row(task_id: str, bug_type: str, buggy_code: str, tests: list[str], response: str) -> dict[str, object]:
    prompt = f"""\
Fix the Python function so it passes the tests. Return only code.

Buggy code:
```python
{buggy_code}
```

Tests:
```python
{chr(10).join(tests)}
```
"""
    return {
        "task_id": task_id,
        "split": "train-success-curriculum",
        "bug_type": bug_type,
        "prompt": prompt,
        "response": response.strip(),
        "source_rank": 0,
    }


def state_contrast_rows() -> list[dict[str, object]]:
    return [
        sft_row(
            "success_state_deltas_after_use",
            "state_update_previous_value",
            "def deltas(nums):\n    out = []\n    prev = 0\n    for n in nums:\n        out.append(n - prev)\n    return out",
            ["assert deltas([3, 5, 9]) == [3, 2, 4]", "assert deltas([-1, -3, 2]) == [-1, -2, 5]"],
            "def deltas(nums):\n    out = []\n    prev = 0\n    for n in nums:\n        out.append(n - prev)\n        prev = n\n    return out",
        ),
        sft_row(
            "success_state_pairs_after_use",
            "state_update_previous_value",
            "def previous_pairs(items):\n    pairs = []\n    previous = None\n    for item in items:\n        pairs.append((previous, item))\n    return pairs",
            [
                "assert previous_pairs(['a', 'b', 'c']) == [(None, 'a'), ('a', 'b'), ('b', 'c')]",
                "assert previous_pairs([1]) == [(None, 1)]",
            ],
            "def previous_pairs(items):\n    pairs = []\n    previous = None\n    for item in items:\n        pairs.append((previous, item))\n        previous = item\n    return pairs",
        ),
        sft_row(
            "success_state_flags_after_use",
            "state_update_previous_value",
            "def same_as_previous(items):\n    flags = []\n    prev = None\n    for item in items:\n        flags.append(item == prev)\n    return flags",
            ["assert same_as_previous(['x', 'x', 'y', 'y']) == [False, True, False, True]"],
            "def same_as_previous(items):\n    flags = []\n    prev = None\n    for item in items:\n        flags.append(item == prev)\n        prev = item\n    return flags",
        ),
        sft_row(
            "success_state_gaps_after_use",
            "state_update_previous_value",
            "def gaps(values):\n    out = []\n    last = 0\n    for value in values:\n        out.append(value - last)\n    return out",
            ["assert gaps([2, 2, 5]) == [2, 0, 3]", "assert gaps([0, -2]) == [0, -2]"],
            "def gaps(values):\n    out = []\n    last = 0\n    for value in values:\n        out.append(value - last)\n        last = value\n    return out",
        ),
        sft_row(
            "success_state_two_vars",
            "state_update_previous_value",
            "def step_ratios(nums):\n    out = []\n    prev = 1\n    for n in nums:\n        out.append(n / prev)\n    return out",
            ["assert step_ratios([2, 4, 8]) == [2, 2, 2]", "assert step_ratios([1]) == [1]"],
            "def step_ratios(nums):\n    out = []\n    prev = 1\n    for n in nums:\n        out.append(n / prev)\n        prev = n\n    return out",
        ),
        sft_row(
            "success_state_initial_none",
            "state_update_previous_value",
            "def changes_or_none(values):\n    out = []\n    previous = None\n    for value in values:\n        if previous is None:\n            out.append(None)\n        else:\n            out.append(value - previous)\n    return out",
            ["assert changes_or_none([5, 8, 3]) == [None, 3, -5]"],
            "def changes_or_none(values):\n    out = []\n    previous = None\n    for value in values:\n        if previous is None:\n            out.append(None)\n        else:\n            out.append(value - previous)\n        previous = value\n    return out",
        ),
    ]


def protection_rows() -> list[dict[str, object]]:
    return [
        sft_row(
            "protect_reversed_bounds",
            "reversed_bounds",
            "def clamp_any_order(x, a, b):\n    if x < a:\n        return a\n    if x > b:\n        return b\n    return x",
            ["assert clamp_any_order(12, 10, 0) == 10", "assert clamp_any_order(-2, 10, 0) == 0"],
            "def clamp_any_order(x, a, b):\n    lo = min(a, b)\n    hi = max(a, b)\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x",
        ),
        sft_row(
            "protect_stable_sort",
            "stable_sort_constraint",
            "def sort_by_score(pairs):\n    return sorted(pairs)",
            [
                "assert sort_by_score([('a', 2), ('b', 1)]) == [('b', 1), ('a', 2)]",
                "assert sort_by_score([('a', 1), ('b', 1)]) == [('a', 1), ('b', 1)]",
            ],
            "def sort_by_score(pairs):\n    return sorted(pairs, key=lambda pair: pair[1])",
        ),
        sft_row(
            "protect_none_falsey",
            "none_vs_falsey",
            "def replace_none(value, default):\n    if not value:\n        return default\n    return value",
            ["assert replace_none(None, 5) == 5", "assert replace_none(0, 5) == 0"],
            "def replace_none(value, default):\n    if value is None:\n        return default\n    return value",
        ),
        sft_row(
            "protect_boundary_negative",
            "boundary_negative_number",
            "def digit_sum(n):\n    total = 0\n    for ch in str(n):\n        total += int(ch)\n    return total",
            ["assert digit_sum(-45) == 9", "assert digit_sum(10001) == 2"],
            "def digit_sum(n):\n    return sum(int(ch) for ch in str(abs(n)))",
        ),
        sft_row(
            "protect_boundary_zero",
            "boundary_zero",
            "def safe_divide(a, b):\n    return a / b",
            ["assert safe_divide(6, 3) == 2", "assert safe_divide(5, 0) is None"],
            "def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a / b",
        ),
        sft_row(
            "protect_mutation",
            "mutation_trap",
            "def sorted_copy(items):\n    items.sort()\n    return items",
            [
                "xs = [3, 1, 2]\nys = sorted_copy(xs)\nassert ys == [1, 2, 3]",
                "xs = [3, 1, 2]\nys = sorted_copy(xs)\nassert xs == [3, 1, 2]",
            ],
            "def sorted_copy(items):\n    return sorted(items)",
        ),
    ]


def build_rows() -> list[dict[str, object]]:
    return state_contrast_rows() + protection_rows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Path("runs/success_curriculum_sft.jsonl"), type=Path)
    parser.add_argument("--repeat", default=2, type=int)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    base_rows = build_rows()
    for repeat_index in range(args.repeat):
        for item in base_rows:
            copied = dict(item)
            copied["task_id"] = f"{item['task_id']}_r{repeat_index:02d}"
            rows.append(copied)

    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} success-curriculum SFT rows to {args.output}")


if __name__ == "__main__":
    main()

