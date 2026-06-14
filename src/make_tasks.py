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
    rows.extend(build_adversarial_tasks())
    return rows


def build_adversarial_tasks() -> list[dict[str, object]]:
    split = "test-adversarial"
    return [
        task(
            "adversarial_none_zero",
            split,
            "Fix the function so it returns the default only when the value is None. Return only code.",
            "def replace_none(value, default):\n    if not value:\n        return default\n    return value",
            ["assert replace_none(None, 5) == 5", "assert replace_none(7, 5) == 7"],
            ["assert replace_none(0, 5) == 0", "assert replace_none('', 'x') == ''"],
            "none_vs_falsey",
        ),
        task(
            "adversarial_index_or_missing",
            split,
            "Fix the function so it returns the index of target or -1 if missing. Return only code.",
            "def find_index(items, target):\n    if target in items:\n        return target\n    return -1",
            ["assert find_index(['a', 'b'], 'b') == 1", "assert find_index([3, 4], 2) == -1"],
            ["assert find_index([0, 1, 0], 0) == 0", "assert find_index(['x'], 'x') == 0"],
            "value_vs_index",
        ),
        task(
            "adversarial_no_mutation",
            split,
            "Fix the function so it returns a sorted copy without mutating the input list. Return only code.",
            "def sorted_copy(items):\n    items.sort()\n    return items",
            [
                "xs = [3, 1, 2]\nys = sorted_copy(xs)\nassert ys == [1, 2, 3]",
                "xs = [3, 1, 2]\nys = sorted_copy(xs)\nassert xs == [3, 1, 2]",
            ],
            [
                "xs = []\nys = sorted_copy(xs)\nassert ys == [] and xs == []",
                "xs = ['b', 'a']\nys = sorted_copy(xs)\nassert ys == ['a', 'b'] and xs == ['b', 'a']",
            ],
            "mutation_trap",
        ),
        task(
            "adversarial_stable_sort_key",
            split,
            "Fix the function so it sorts pairs by the second value while preserving order for ties. Return only code.",
            "def sort_by_score(pairs):\n    return sorted(pairs)",
            [
                "assert sort_by_score([('a', 2), ('b', 1)]) == [('b', 1), ('a', 2)]",
                "assert sort_by_score([('a', 1), ('b', 1)]) == [('a', 1), ('b', 1)]",
            ],
            [
                "assert sort_by_score([('z', 0), ('a', 0), ('m', -1)]) == [('m', -1), ('z', 0), ('a', 0)]",
            ],
            "stable_sort_constraint",
        ),
        task(
            "adversarial_clamp_reversed_bounds",
            split,
            "Fix the function so it clamps x even when bounds are provided in either order. Return only code.",
            "def clamp_any_order(x, a, b):\n    if x < a:\n        return a\n    if x > b:\n        return b\n    return x",
            ["assert clamp_any_order(5, 0, 10) == 5", "assert clamp_any_order(-2, 0, 10) == 0"],
            ["assert clamp_any_order(12, 10, 0) == 10", "assert clamp_any_order(-2, 10, 0) == 0"],
            "reversed_bounds",
        ),
        task(
            "adversarial_shadow_nested",
            split,
            "Fix the function so it counts items equal to target across nested lists. Return only code.",
            "def count_nested(groups, target):\n    count = 0\n    for target in groups:\n        for item in target:\n            if item == target:\n                count += 1\n    return count",
            ["assert count_nested([[1, 2], [1]], 1) == 2", "assert count_nested([[3], []], 2) == 0"],
            ["assert count_nested([['a'], ['a', 'b']], 'a') == 2", "assert count_nested([], 1) == 0"],
            "nested_variable_shadowing",
        ),
        task(
            "adversarial_digit_sum_string",
            split,
            "Fix the function so it sums digit characters in a string and ignores non-digits. Return only code.",
            "def sum_digits_text(text):\n    return sum(int(ch) for ch in text)",
            ["assert sum_digits_text('a1b2') == 3", "assert sum_digits_text('') == 0"],
            ["assert sum_digits_text('-12.3') == 6", "assert sum_digits_text('no digits') == 0"],
            "type_role_disambiguation",
        ),
        task(
            "adversarial_running_difference",
            split,
            "Fix the function so it returns differences between each item and the previous item. Return only code.",
            "def deltas(nums):\n    out = []\n    prev = 0\n    for n in nums:\n        out.append(n - prev)\n    return out",
            ["assert deltas([3, 5, 9]) == [3, 2, 4]"],
            ["assert deltas([]) == []", "assert deltas([-1, -3, 2]) == [-1, -2, 5]"],
            "state_update_previous_value",
        ),
    ]


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
