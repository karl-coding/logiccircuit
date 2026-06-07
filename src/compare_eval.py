from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from .io_utils import read_jsonl


def summarize(eval_path: Path) -> dict[str, dict[str, float]]:
    rows = read_jsonl(eval_path)
    split_task_results: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        split = str(row["split"])
        task_id = str(row["task_id"])
        split_task_results[split][task_id].append(bool(row["passed_all"]))

    summary: dict[str, dict[str, float]] = {}
    for split, task_results in split_task_results.items():
        total = len(task_results)
        summary[split] = {"total": float(total)}
        for k in (1, 3, 8):
            passed = sum(any(results[:k]) for results in task_results.values())
            summary[split][f"pass@{k}"] = passed / total if total else 0.0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    args = parser.parse_args()

    baseline = summarize(args.baseline)
    adapter = summarize(args.adapter)
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

