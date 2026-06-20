from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, write_jsonl
from .task_schema import CodeRepairTask
from .verifier import run_tests, verify_candidate


def score_candidate(
    code: str,
    task: CodeRepairTask,
    mode: str,
) -> tuple[int, bool, bool, str | None]:
    if mode == "public":
        passed_public, error = run_tests(code, task.public_tests)
        return int(passed_public), passed_public, False, error

    result = verify_candidate(code, task.public_tests, task.hidden_tests)
    return int(result.passed_all), result.passed_public, result.passed_hidden, result.error


def rerank_candidates(
    tasks_path: Path,
    candidates_path: Path,
    output_path: Path,
    splits: set[str] | None,
    mode: str,
    prefer_shorter: bool,
) -> int:
    if not tasks_path.exists():
        raise FileNotFoundError(f"task file not found: {tasks_path}")
    if not candidates_path.exists():
        raise FileNotFoundError(f"candidate file not found: {candidates_path}")

    tasks = [CodeRepairTask.from_json(row) for row in read_jsonl(tasks_path)]
    if splits is not None:
        tasks = [task for task in tasks if task.split in splits]
    task_by_id = {task.id: task for task in tasks}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(candidates_path):
        task_id = str(row["task_id"])
        if task_id in task_by_id:
            grouped[task_id].append(row)

    output_rows: list[dict[str, Any]] = []
    selected_public_passes = 0
    selected_oracle_passes = 0
    selected_total = 0

    for task in tasks:
        rows = sorted(grouped.get(task.id, []), key=lambda item: int(item["rank"]))
        if not rows:
            continue

        scored: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
        for row in rows:
            original_rank = int(row["rank"])
            code = str(row["code"])
            score, passed_public, passed_hidden, error = score_candidate(code, task, mode)
            length_penalty = len(code) if prefer_shorter else 0
            sort_key = (-score, length_penalty, original_rank)
            enriched = dict(row)
            enriched["original_rank"] = original_rank
            enriched["selector_mode"] = mode
            enriched["selector_score"] = score
            enriched["selector_passed_public"] = passed_public
            enriched["selector_passed_hidden"] = passed_hidden
            enriched["selector_error"] = error
            scored.append((sort_key, enriched))

        reranked = [row for _, row in sorted(scored, key=lambda item: item[0])]
        selected = reranked[0]
        selected_public_passes += int(bool(selected["selector_passed_public"]))
        selected_oracle_passes += int(bool(selected["selector_passed_hidden"]))
        selected_total += 1

        for new_rank, row in enumerate(reranked, start=1):
            row["rank"] = new_rank
            output_rows.append(row)

    write_jsonl(output_path, output_rows)
    if mode == "oracle":
        print(
            f"selected public-pass {selected_public_passes}/{selected_total}; "
            f"selected oracle-pass {selected_oracle_passes}/{selected_total}"
        )
    else:
        print(f"selected public-pass {selected_public_passes}/{selected_total}")
    return len(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--splits", nargs="+")
    parser.add_argument(
        "--mode",
        default="public",
        choices=["public", "oracle"],
        help="public uses only public tests; oracle uses hidden tests for upper-bound analysis.",
    )
    parser.add_argument("--prefer-shorter", action="store_true")
    args = parser.parse_args()

    count = rerank_candidates(
        tasks_path=args.tasks,
        candidates_path=args.candidates,
        output_path=args.output,
        splits=set(args.splits) if args.splits else None,
        mode=args.mode,
        prefer_shorter=args.prefer_shorter,
    )
    print(f"wrote {count} reranked candidates to {args.output}")


if __name__ == "__main__":
    main()
