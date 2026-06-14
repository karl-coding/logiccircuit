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
        "split": "train-hard-patch",
        "bug_type": bug_type,
        "prompt": prompt,
        "response": response.strip(),
        "source_rank": 0,
    }


def build_rows() -> list[dict[str, object]]:
    return [
        row(
            "hard_patch_deltas_001",
            "state_update_previous_value",
            "def deltas(nums):\n    out = []\n    prev = 0\n    for n in nums:\n        out.append(n - prev)\n    return out",
            ["assert deltas([3, 5, 9]) == [3, 2, 4]", "assert deltas([-1, -3, 2]) == [-1, -2, 5]"],
            "def deltas(nums):\n    out = []\n    prev = 0\n    for n in nums:\n        out.append(n - prev)\n        prev = n\n    return out",
        ),
        row(
            "hard_patch_deltas_002",
            "state_update_previous_value",
            "def changes(values):\n    result = []\n    last = 0\n    for value in values:\n        result.append(value - last)\n    return result",
            ["assert changes([10, 7, 8]) == [10, -3, 1]", "assert changes([]) == []"],
            "def changes(values):\n    result = []\n    last = 0\n    for value in values:\n        result.append(value - last)\n        last = value\n    return result",
        ),
        row(
            "hard_patch_deltas_003",
            "state_update_previous_value",
            "def step_diffs(nums):\n    diffs = []\n    previous = None\n    for n in nums:\n        if previous is None:\n            diffs.append(n)\n        else:\n            diffs.append(n - previous)\n    return diffs",
            ["assert step_diffs([4, 6, 1]) == [4, 2, -5]", "assert step_diffs([0]) == [0]"],
            "def step_diffs(nums):\n    diffs = []\n    previous = None\n    for n in nums:\n        if previous is None:\n            diffs.append(n)\n        else:\n            diffs.append(n - previous)\n        previous = n\n    return diffs",
        ),
        row(
            "hard_patch_no_mutation_001",
            "mutation_trap",
            "def sorted_copy(items):\n    items.sort()\n    return items",
            [
                "xs = [3, 1, 2]\nys = sorted_copy(xs)\nassert ys == [1, 2, 3]",
                "xs = [3, 1, 2]\nys = sorted_copy(xs)\nassert xs == [3, 1, 2]",
            ],
            "def sorted_copy(items):\n    return sorted(items)",
        ),
        row(
            "hard_patch_no_mutation_002",
            "mutation_trap",
            "def reversed_copy(items):\n    items.reverse()\n    return items",
            [
                "xs = [1, 2, 3]\nys = reversed_copy(xs)\nassert ys == [3, 2, 1]",
                "xs = [1, 2, 3]\nys = reversed_copy(xs)\nassert xs == [1, 2, 3]",
            ],
            "def reversed_copy(items):\n    return list(reversed(items))",
        ),
        row(
            "hard_patch_no_mutation_003",
            "mutation_trap",
            "def append_copy(items, value):\n    items.append(value)\n    return items",
            [
                "xs = [1, 2]\nys = append_copy(xs, 3)\nassert ys == [1, 2, 3]",
                "xs = [1, 2]\nys = append_copy(xs, 3)\nassert xs == [1, 2]",
            ],
            "def append_copy(items, value):\n    return list(items) + [value]",
        ),
        row(
            "hard_patch_boundary_zero_001",
            "boundary_zero",
            "def safe_divide(a, b):\n    return a / b",
            ["assert safe_divide(6, 3) == 2", "assert safe_divide(5, 0) is None"],
            "def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a / b",
        ),
        row(
            "hard_patch_boundary_zero_002",
            "boundary_zero",
            "def ratio_or_none(top, bottom):\n    if not bottom:\n        return None\n    return top / bottom",
            ["assert ratio_or_none(0, 5) == 0", "assert ratio_or_none(5, 0) is None"],
            "def ratio_or_none(top, bottom):\n    if bottom == 0:\n        return None\n    return top / bottom",
        ),
        row(
            "hard_patch_boundary_zero_003",
            "boundary_zero",
            "def inverse(x):\n    return 1 / x",
            ["assert inverse(2) == 0.5", "assert inverse(0) is None"],
            "def inverse(x):\n    if x == 0:\n        return None\n    return 1 / x",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Path("runs/hard_patch_sft.jsonl"), type=Path)
    parser.add_argument("--repeat", default=2, type=int)
    args = parser.parse_args()

    base_rows = build_rows()
    rows: list[dict[str, object]] = []
    for repeat_index in range(args.repeat):
        for item in base_rows:
            copied = dict(item)
            copied["task_id"] = f"{item['task_id']}_r{repeat_index:02d}"
            rows.append(copied)

    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} hard-patch SFT rows to {args.output}")


if __name__ == "__main__":
    main()

