from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .io_utils import read_jsonl


TaskKey = tuple[str, str, str]


def load_results(path: Path) -> dict[TaskKey, list[tuple[int, bool]]]:
    if not path.exists():
        raise FileNotFoundError(f"evaluation file not found: {path}")

    grouped: dict[TaskKey, list[tuple[int, bool]]] = defaultdict(list)
    for row in read_jsonl(path):
        key = (str(row["split"]), str(row["bug_type"]), str(row["task_id"]))
        grouped[key].append((int(row["rank"]), bool(row["passed_all"])))
    return grouped


def pass_at(results: list[tuple[int, bool]], k: int) -> bool:
    ranked = [passed for _, passed in sorted(results, key=lambda item: item[0])]
    return any(ranked[:k])


def summarize_by_bug_type(
    results: dict[TaskKey, list[tuple[int, bool]]],
    ks: list[int],
) -> dict[tuple[str, str], dict[str, float]]:
    buckets: dict[tuple[str, str], list[list[tuple[int, bool]]]] = defaultdict(list)
    for split, bug_type, _task_id in results:
        buckets[(split, bug_type)].append(results[(split, bug_type, _task_id)])

    summary: dict[tuple[str, str], dict[str, float]] = {}
    for key, task_results in buckets.items():
        metrics: dict[str, float] = {"total": float(len(task_results))}
        for k in ks:
            metrics[f"pass@{k}"] = sum(pass_at(result, k) for result in task_results) / len(task_results)
        summary[key] = metrics
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--k", nargs="+", default=[1, 2], type=int)
    args = parser.parse_args()

    baseline = summarize_by_bug_type(load_results(args.baseline), args.k)
    adapter = summarize_by_bug_type(load_results(args.adapter), args.k)
    keys = sorted(set(baseline) | set(adapter))

    print("split,bug_type,total,metric,baseline,adapter,delta")
    for split, bug_type in keys:
        base_metrics = baseline.get((split, bug_type), {})
        adapter_metrics = adapter.get((split, bug_type), {})
        total = int(adapter_metrics.get("total", base_metrics.get("total", 0)))
        for k in args.k:
            metric = f"pass@{k}"
            base_value = float(base_metrics.get(metric, 0.0))
            adapter_value = float(adapter_metrics.get(metric, 0.0))
            print(
                f"{split},{bug_type},{total},{metric},"
                f"{base_value:.4f},{adapter_value:.4f},{adapter_value - base_value:.4f}"
            )


if __name__ == "__main__":
    main()

