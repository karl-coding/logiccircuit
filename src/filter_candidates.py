from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .io_utils import read_jsonl, write_jsonl
from .make_prompts import build_prompt
from .task_schema import Candidate, CodeRepairTask
from .verifier import verify_candidate


def filter_candidates(
    tasks_path: Path,
    candidates_path: Path,
    output_path: Path,
    max_per_task: int,
    splits: set[str] | None,
) -> int:
    if not tasks_path.exists():
        raise FileNotFoundError(
            f"task file not found: {tasks_path}. "
            "Run src.make_tasks first or provide an existing JSONL task file."
        )
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"candidate file not found: {candidates_path}. Run src.generate_candidates first."
        )

    tasks = [CodeRepairTask.from_json(row) for row in read_jsonl(tasks_path)]
    if splits is not None:
        tasks = [task for task in tasks if task.split in splits]
    candidates = [Candidate.from_json(row) for row in read_jsonl(candidates_path)]

    candidates_by_task: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_task[candidate.task_id].append(candidate)

    rows: list[dict[str, object]] = []
    for task in tasks:
        kept = 0
        ranked = sorted(candidates_by_task.get(task.id, []), key=lambda item: item.rank)
        for candidate in ranked:
            if kept >= max_per_task:
                break
            result = verify_candidate(candidate.code, task.public_tests, task.hidden_tests)
            if not result.passed_all:
                continue
            rows.append(
                {
                    "task_id": task.id,
                    "split": task.split,
                    "bug_type": task.bug_type,
                    "prompt": build_prompt(task),
                    "response": candidate.code,
                    "source_rank": candidate.rank,
                }
            )
            kept += 1

    write_jsonl(output_path, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-per-task", default=1, type=int)
    parser.add_argument("--splits", nargs="+")
    args = parser.parse_args()

    count = filter_candidates(
        args.tasks,
        args.candidates,
        args.output,
        args.max_per_task,
        set(args.splits) if args.splits else None,
    )
    print(f"wrote {count} training rows to {args.output}")


if __name__ == "__main__":
    main()
