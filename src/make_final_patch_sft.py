from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import write_jsonl


def row(task_id: str, bug_type: str, buggy_code: str, tests: list[str], response: str) -> dict[str, object]:
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
        "split": "train-final-patch",
        "bug_type": bug_type,
        "prompt": prompt,
        "response": response.strip(),
        "source_rank": 0,
    }


def build_rows() -> list[dict[str, object]]:
    return [
        row(
            "final_running_difference_prev_after",
            "state_update_previous_value",
            "def running_difference(nums):\n    out = []\n    prev = 0\n    for n in nums:\n        prev = n\n        out.append(n - prev)\n    return out",
            [
                "assert running_difference([3, 5, 9]) == [3, 2, 4]",
                "assert running_difference([-1, -3, 2]) == [-1, -2, 5]",
            ],
            "def running_difference(nums):\n    out = []\n    prev = 0\n    for n in nums:\n        out.append(n - prev)\n        prev = n\n    return out",
        ),
        row(
            "final_running_difference_missing_update",
            "state_update_previous_value",
            "def running_difference(nums):\n    out = []\n    previous = 0\n    for value in nums:\n        out.append(value - previous)\n    return out",
            [
                "assert running_difference([10, 7, 8]) == [10, -3, 1]",
                "assert running_difference([]) == []",
            ],
            "def running_difference(nums):\n    out = []\n    previous = 0\n    for value in nums:\n        out.append(value - previous)\n        previous = value\n    return out",
        ),
        row(
            "final_clamp_reversed_bounds_sorted",
            "reversed_bounds",
            "def clamp(value, lower, upper):\n    if value < lower:\n        return lower\n    if value > upper:\n        return upper\n    return value",
            [
                "assert clamp(5, 10, 0) == 5",
                "assert clamp(-1, 10, 0) == 0",
                "assert clamp(12, 10, 0) == 10",
            ],
            "def clamp(value, lower, upper):\n    lo = min(lower, upper)\n    hi = max(lower, upper)\n    if value < lo:\n        return lo\n    if value > hi:\n        return hi\n    return value",
        ),
        row(
            "final_clamp_reversed_bounds_expression",
            "reversed_bounds",
            "def clamp_score(score, a, b):\n    return max(a, min(score, b))",
            [
                "assert clamp_score(7, 10, 0) == 7",
                "assert clamp_score(-5, 10, 0) == 0",
                "assert clamp_score(15, 10, 0) == 10",
            ],
            "def clamp_score(score, a, b):\n    lo = min(a, b)\n    hi = max(a, b)\n    return max(lo, min(score, hi))",
        ),
        row(
            "final_digit_sum_negative_abs",
            "boundary_negative_number",
            "def digit_sum(n):\n    total = 0\n    for ch in str(n):\n        total += int(ch)\n    return total",
            [
                "assert digit_sum(-45) == 9",
                "assert digit_sum(-1002) == 3",
                "assert digit_sum(0) == 0",
            ],
            "def digit_sum(n):\n    total = 0\n    for ch in str(abs(n)):\n        total += int(ch)\n    return total",
        ),
        row(
            "final_digit_sum_string_negative",
            "type_role_disambiguation",
            "def digit_sum_string(text):\n    total = 0\n    for ch in text:\n        total += int(ch)\n    return total",
            [
                "assert digit_sum_string('-45') == 9",
                "assert digit_sum_string('a1b2') == 3",
                "assert digit_sum_string('') == 0",
            ],
            "def digit_sum_string(text):\n    total = 0\n    for ch in text:\n        if ch.isdigit():\n            total += int(ch)\n    return total",
        ),
        row(
            "protect_running_total_append_after_update",
            "state_update_order",
            "def running_total(nums):\n    out = []\n    total = 0\n    for n in nums:\n        out.append(total)\n        total += n\n    return out",
            [
                "assert running_total([1, 2, 3]) == [1, 3, 6]",
                "assert running_total([-1, 2]) == [-1, 1]",
            ],
            "def running_total(nums):\n    out = []\n    total = 0\n    for n in nums:\n        total += n\n        out.append(total)\n    return out",
        ),
        row(
            "protect_running_total_not_difference",
            "state_update_order",
            "def cumulative(values):\n    out = []\n    previous = 0\n    for value in values:\n        out.append(value - previous)\n        previous = value\n    return out",
            [
                "assert cumulative([2, 5, 1]) == [2, 7, 8]",
                "assert cumulative([]) == []",
            ],
            "def cumulative(values):\n    out = []\n    total = 0\n    for value in values:\n        total += value\n        out.append(total)\n    return out",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Path("runs/final_patch_sft.jsonl"), type=Path)
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
    print(f"wrote {len(rows)} final-patch SFT rows to {args.output}")


if __name__ == "__main__":
    main()
