from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import read_jsonl, write_jsonl


def select_tasks(tasks_path: Path, output_path: Path, task_ids: set[str]) -> int:
    if not tasks_path.exists():
        raise FileNotFoundError(f"task file not found: {tasks_path}")
    rows = read_jsonl(tasks_path)
    selected = [row for row in rows if str(row["id"]) in task_ids]
    found = {str(row["id"]) for row in selected}
    missing = sorted(task_ids - found)
    if missing:
        raise ValueError(f"task ids not found in {tasks_path}: {', '.join(missing)}")
    write_jsonl(output_path, selected)
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task-ids", nargs="+", required=True)
    args = parser.parse_args()

    count = select_tasks(args.tasks, args.output, set(args.task_ids))
    print(f"wrote {count} selected tasks to {args.output}")


if __name__ == "__main__":
    main()
