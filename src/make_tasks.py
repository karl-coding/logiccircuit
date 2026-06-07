from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import write_jsonl


def task(
    task_id: str,
    split: str,
    instruction: str,
    buggy_code: str,
    public_tests: list[str],
    hidden_tests: list[str],
    bug_type: str,
) -> dict[str, object]:
    return {
        "id": task_id,
        "split": split,
        "instruction": instruction,
        "buggy_code": buggy_code,
        "public_tests": public_tests,
        "hidden_tests": hidden_tests,
        "bug_type": bug_type,
    }


def build_tasks() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    splits = ["train", "train", "train", "test-similar", "test-hard", "test-transfer"]

    for index, split in enumerate(splits):
        suffix = f"{index:03d}"
        rows.extend(
            [
                task(
                    f"digit_sum_{suffix}",
                    split,
                    "Fix the Python function so it passes the tests. Return only code.",
                    "def digit_sum(n):\n    total = 0\n    for ch in str(n):\n        total += int(ch)\n    return total",
                    ["assert digit_sum(123) == 6", "assert digit_sum(0) == 0"],
                    ["assert digit_sum(-45) == 9", "assert digit_sum(10001) == 2"],
                    "boundary_negative_number",
                ),
                task(
                    f"unique_order_{suffix}",
                    split,
                    "Fix the function so it removes duplicates while preserving first occurrence order. Return only code.",
                    "def unique_items(items):\n    return list(set(items))",
                    ["assert unique_items([1, 2, 1, 3]) == [1, 2, 3]"],
                    ["assert unique_items(['b', 'a', 'b']) == ['b', 'a']", "assert unique_items([]) == []"],
                    "state_update_order",
                ),
                task(
                    f"clamp_{suffix}",
                    split,
                    "Fix the Python function so it applies both lower and upper bounds. Return only code.",
                    "def clamp(x, lo, hi):\n    if x < lo:\n        return lo\n    return x",
                    ["assert clamp(2, 0, 5) == 2", "assert clamp(-1, 0, 5) == 0"],
                    ["assert clamp(9, 0, 5) == 5", "assert clamp(5, 0, 5) == 5"],
                    "constraint_propagation",
                ),
                task(
                    f"count_matches_{suffix}",
                    split,
                    "Fix the function so it counts values equal to the target. Return only code.",
                    "def count_matches(items, target):\n    count = 0\n    for target in items:\n        if target == target:\n            count += 1\n    return count",
                    ["assert count_matches([1, 2, 1], 1) == 2", "assert count_matches([3, 4], 2) == 0"],
                    ["assert count_matches([], 1) == 0", "assert count_matches(['a', 'b', 'a'], 'a') == 2"],
                    "variable_shadowing",
                ),
                task(
                    f"running_total_{suffix}",
                    split,
                    "Fix the function so it returns running totals. Return only code.",
                    "def running_total(nums):\n    out = []\n    total = 0\n    for n in nums:\n        out.append(total)\n        total += n\n    return out",
                    ["assert running_total([1, 2, 3]) == [1, 3, 6]"],
                    ["assert running_total([]) == []", "assert running_total([-1, 2]) == [-1, 1]"],
                    "state_update_order",
                ),
                task(
                    f"safe_divide_{suffix}",
                    split,
                    "Fix the function so division by zero returns None. Return only code.",
                    "def safe_divide(a, b):\n    return a / b",
                    ["assert safe_divide(6, 3) == 2", "assert safe_divide(5, 0) is None"],
                    ["assert safe_divide(0, 2) == 0", "assert safe_divide(-6, 3) == -2"],
                    "boundary_zero",
                ),
                task(
                    f"get_nested_{suffix}",
                    split,
                    "Fix the function so it safely reads a nested dictionary key. Return only code.",
                    "def get_nested(data, outer, inner, default=None):\n    return data[outer][inner]",
                    [
                        "assert get_nested({'a': {'b': 3}}, 'a', 'b') == 3",
                        "assert get_nested({}, 'a', 'b', 0) == 0",
                    ],
                    [
                        "assert get_nested({'a': {}}, 'a', 'b', 'x') == 'x'",
                        "assert get_nested({'a': {'b': None}}, 'a', 'b', 'x') is None",
                    ],
                    "nested_constraint",
                ),
                task(
                    f"flatten_one_{suffix}",
                    split,
                    "Fix the function so it flattens one level of nested lists. Return only code.",
                    "def flatten_one(items):\n    out = []\n    for item in items:\n        out.append(item)\n    return out",
                    ["assert flatten_one([[1, 2], [3]]) == [1, 2, 3]"],
                    ["assert flatten_one([]) == []", "assert flatten_one([[], [1]]) == [1]"],
                    "loop_state_update",
                ),
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Path("data/tasks.jsonl"), type=Path)
    parser.add_argument("--repeat", default=1, type=int)
    args = parser.parse_args()

    base_rows = build_tasks()
    rows: list[dict[str, object]] = []
    for repeat_index in range(args.repeat):
        for row in base_rows:
            copied = dict(row)
            copied["id"] = f"{row['id']}_r{repeat_index:02d}"
            rows.append(copied)

    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} tasks to {args.output}")


if __name__ == "__main__":
    main()

