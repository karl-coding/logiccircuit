from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .io_utils import read_jsonl, write_jsonl
from .task_schema import Candidate, CodeRepairTask
from .verifier import verify_candidate


def pass_at_k(results: list[bool], k: int) -> bool:
    return any(results[:k])


def evaluate(tasks_path: Path, candidates_path: Path, output_path: Path, ks: list[int]) -> dict[str, object]:
    tasks = [CodeRepairTask.from_json(row) for row in read_jsonl(tasks_path)]
    candidates = [Candidate.from_json(row) for row in read_jsonl(candidates_path)]

    task_by_id = {task.id: task for task in tasks}
    candidates_by_task: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_task[candidate.task_id].append(candidate)

    detail_rows: list[dict[str, object]] = []
    aggregate: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for task in tasks:
        ranked = sorted(candidates_by_task.get(task.id, []), key=lambda item: item.rank)
        candidate_results: list[bool] = []

        for candidate in ranked:
            result = verify_candidate(candidate.code, task.public_tests, task.hidden_tests)
            candidate_results.append(result.passed_all)
            detail_rows.append(
                {
                    "task_id": task.id,
                    "split": task.split,
                    "bug_type": task.bug_type,
                    "rank": candidate.rank,
                    "passed_public": result.passed_public,
                    "passed_hidden": result.passed_hidden,
                    "passed_all": result.passed_all,
                    "error": result.error,
                }
            )

        for k in ks:
            key = f"pass@{k}"
            aggregate[task.split][key] += int(pass_at_k(candidate_results, k))
        aggregate[task.split]["total"] += 1

    summary: dict[str, object] = {}
    for split, counts in aggregate.items():
        total = counts["total"]
        summary[split] = {
            key: value / total
            for key, value in counts.items()
            if key != "total"
        } | {"total": total}

    write_jsonl(output_path, detail_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--k", nargs="+", default=[1, 3, 8], type=int)
    args = parser.parse_args()

    summary = evaluate(args.tasks, args.candidates, args.output, args.k)
    for split, metrics in summary.items():
        print(split, metrics)


if __name__ == "__main__":
    main()

