from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .eval_utils import ensure_eval_file
from .io_utils import read_jsonl


def summarize(eval_path: Path, tasks_path: Path, splits: set[str] | None) -> dict[str, dict[str, float]]:
    ensure_eval_file(eval_path, tasks_path, splits, ks=[1, 2, 3, 8])
    rows = read_jsonl(eval_path)
    split_task_results: dict[str, dict[str, list[tuple[int, bool]]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        split = str(row["split"])
        task_id = str(row["task_id"])
        split_task_results[split][task_id].append((int(row["rank"]), bool(row["passed_all"])))

    summary: dict[str, dict[str, float]] = {}
    for split, task_results in split_task_results.items():
        total = len(task_results)
        summary[split] = {"total": float(total)}
        max_rank = max(rank for results in task_results.values() for rank, _ in results)
        candidate_ks = sorted({1, 2, 3, 8, max_rank})
        for k in candidate_ks:
            if k > max_rank:
                continue
            passed = 0
            for results in task_results.values():
                ranked = [passed_all for _, passed_all in sorted(results, key=lambda item: item[0])]
                passed += int(any(ranked[:k]))
            summary[split][f"pass@{k}"] = passed / total if total else 0.0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--tasks", default=Path("data/tasks.jsonl"), type=Path)
    parser.add_argument("--splits", nargs="+")
    args = parser.parse_args()

    splits_filter = set(args.splits) if args.splits else None
    baseline = summarize(args.baseline, args.tasks, splits_filter)
    adapter = summarize(args.adapter, args.tasks, splits_filter)
    splits = sorted(set(baseline) | set(adapter))

    print("split,total,metric,baseline,adapter,delta")
    for split in splits:
        metrics = sorted((set(baseline.get(split, {})) | set(adapter.get(split, {}))) - {"total"})
        total = int(adapter.get(split, baseline.get(split, {})).get("total", 0))
        for metric in metrics:
            base_value = baseline.get(split, {}).get(metric, 0.0)
            adapter_value = adapter.get(split, {}).get(metric, 0.0)
            print(
                f"{split},{total},{metric},"
                f"{base_value:.4f},{adapter_value:.4f},{adapter_value - base_value:.4f}"
            )


if __name__ == "__main__":
    main()
