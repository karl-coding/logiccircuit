from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import write_jsonl


def row(task_id: str, buggy_code: str, tests: list[str], response: str) -> dict[str, object]:
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
        "split": "train-state-curriculum",
        "bug_type": "state_update_previous_value",
        "prompt": prompt,
        "response": response.strip(),
        "source_rank": 0,
    }


def build_rows() -> list[dict[str, object]]:
    return [
        row(
            "state_curriculum_deltas_basic",
            "def deltas(nums):\n    out = []\n    prev = 0\n    for n in nums:\n        out.append(n - prev)\n    return out",
            ["assert deltas([3, 5, 9]) == [3, 2, 4]", "assert deltas([]) == []"],
            "def deltas(nums):\n    out = []\n    prev = 0\n    for n in nums:\n        out.append(n - prev)\n        prev = n\n    return out",
        ),
        row(
            "state_curriculum_price_changes",
            "def price_changes(prices):\n    changes = []\n    previous = 0\n    for price in prices:\n        changes.append(price - previous)\n    return changes",
            ["assert price_changes([100, 95, 105]) == [100, -5, 10]", "assert price_changes([0, 2]) == [0, 2]"],
            "def price_changes(prices):\n    changes = []\n    previous = 0\n    for price in prices:\n        changes.append(price - previous)\n        previous = price\n    return changes",
        ),
        row(
            "state_curriculum_time_gaps",
            "def time_gaps(times):\n    gaps = []\n    last = 0\n    for t in times:\n        gaps.append(t - last)\n    return gaps",
            ["assert time_gaps([5, 8, 20]) == [5, 3, 12]", "assert time_gaps([]) == []"],
            "def time_gaps(times):\n    gaps = []\n    last = 0\n    for t in times:\n        gaps.append(t - last)\n        last = t\n    return gaps",
        ),
        row(
            "state_curriculum_pairwise_distances",
            "def pair_distances(points):\n    distances = []\n    prev = 0\n    for point in points:\n        distances.append(abs(point - prev))\n    return distances",
            ["assert pair_distances([2, 5, 1]) == [2, 3, 4]", "assert pair_distances([-2, -5]) == [2, 3]"],
            "def pair_distances(points):\n    distances = []\n    prev = 0\n    for point in points:\n        distances.append(abs(point - prev))\n        prev = point\n    return distances",
        ),
        row(
            "state_curriculum_previous_pairs",
            "def previous_pairs(items):\n    pairs = []\n    previous = None\n    for item in items:\n        pairs.append((previous, item))\n    return pairs",
            [
                "assert previous_pairs(['a', 'b', 'c']) == [(None, 'a'), ('a', 'b'), ('b', 'c')]",
                "assert previous_pairs([]) == []",
            ],
            "def previous_pairs(items):\n    pairs = []\n    previous = None\n    for item in items:\n        pairs.append((previous, item))\n        previous = item\n    return pairs",
        ),
        row(
            "state_curriculum_sign_changes",
            "def sign_changes(nums):\n    out = []\n    prev = 0\n    for n in nums:\n        out.append((prev < 0 and n >= 0) or (prev >= 0 and n < 0))\n    return out",
            ["assert sign_changes([1, -1, -2, 3]) == [False, True, False, True]"],
            "def sign_changes(nums):\n    out = []\n    prev = 0\n    for n in nums:\n        out.append((prev < 0 and n >= 0) or (prev >= 0 and n < 0))\n        prev = n\n    return out",
        ),
        row(
            "state_curriculum_streak_lengths",
            "def same_as_previous_flags(items):\n    flags = []\n    previous = None\n    for item in items:\n        flags.append(item == previous)\n    return flags",
            ["assert same_as_previous_flags(['a', 'a', 'b', 'b']) == [False, True, False, True]"],
            "def same_as_previous_flags(items):\n    flags = []\n    previous = None\n    for item in items:\n        flags.append(item == previous)\n        previous = item\n    return flags",
        ),
        row(
            "state_curriculum_cumulative_delta_check",
            "def sum_of_steps(nums):\n    total = 0\n    previous = 0\n    for n in nums:\n        total += n - previous\n    return total",
            ["assert sum_of_steps([3, 5, 9]) == 9", "assert sum_of_steps([-1, -3, 2]) == 2"],
            "def sum_of_steps(nums):\n    total = 0\n    previous = 0\n    for n in nums:\n        total += n - previous\n        previous = n\n    return total",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Path("runs/state_curriculum_sft.jsonl"), type=Path)
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
    print(f"wrote {len(rows)} state-curriculum SFT rows to {args.output}")


if __name__ == "__main__":
    main()

