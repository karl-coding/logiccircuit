from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .eval_utils import ensure_eval_file
from .io_utils import read_jsonl


def load_passes(
    path: Path,
    tasks_path: Path,
    splits: set[str] | None,
    k: int,
) -> dict[str, dict[str, object]]:
    ensure_eval_file(path, tasks_path, splits, ks=[1, 2, 3, 8, k])
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[str(row["task_id"])].append(row)

    out: dict[str, dict[str, object]] = {}
    for task_id, rows in grouped.items():
        ranked = sorted(rows, key=lambda row: int(row["rank"]))
        out[task_id] = {
            "split": str(ranked[0]["split"]),
            "bug_type": str(ranked[0]["bug_type"]),
            "passed": any(bool(row["passed_all"]) for row in ranked[:k]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--tasks", default=Path("data/tasks.jsonl"), type=Path)
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--k", default=1, type=int)
    args = parser.parse_args()

    splits_filter = set(args.splits) if args.splits else None
    baseline = load_passes(args.baseline, args.tasks, splits_filter, args.k)
    adapter = load_passes(args.adapter, args.tasks, splits_filter, args.k)
    task_ids = sorted(set(baseline) | set(adapter))

    print("category,split,bug_type,task_id")
    for task_id in task_ids:
        base = baseline.get(task_id)
        tuned = adapter.get(task_id)
        if base is None or tuned is None:
            continue
        if bool(base["passed"]) and not bool(tuned["passed"]):
            category = "regression"
        elif not bool(base["passed"]) and not bool(tuned["passed"]):
            category = "unresolved"
        elif not bool(base["passed"]) and bool(tuned["passed"]):
            category = "improved"
        else:
            continue
        print(f"{category},{tuned['split']},{tuned['bug_type']},{task_id}")


if __name__ == "__main__":
    main()
