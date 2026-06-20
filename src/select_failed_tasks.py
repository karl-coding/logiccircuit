from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, write_jsonl


def failed_task_ids(eval_path: Path, k: int) -> set[str]:
    if not eval_path.exists():
        raise FileNotFoundError(f"evaluation file not found: {eval_path}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(eval_path):
        grouped[str(row["task_id"])].append(row)

    failed: set[str] = set()
    for task_id, rows in grouped.items():
        ranked = sorted(rows, key=lambda row: int(row["rank"]))
        if not any(bool(row["passed_all"]) for row in ranked[:k]):
            failed.add(task_id)
    return failed


def select_failed_tasks(tasks_path: Path, eval_path: Path, output_path: Path, k: int) -> int:
    if not tasks_path.exists():
        raise FileNotFoundError(f"task file not found: {tasks_path}")

    failed = failed_task_ids(eval_path, k)
    rows = [row for row in read_jsonl(tasks_path) if str(row["id"]) in failed]
    write_jsonl(output_path, rows)

    print("split,bug_type,task_id")
    for row in rows:
        print(f"{row['split']},{row['bug_type']},{row['id']}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", default=1, type=int)
    args = parser.parse_args()

    count = select_failed_tasks(args.tasks, args.eval, args.output, args.k)
    print(f"wrote {count} failed tasks to {args.output}")


if __name__ == "__main__":
    main()
