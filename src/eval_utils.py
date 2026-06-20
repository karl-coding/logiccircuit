from __future__ import annotations

from pathlib import Path

from .evaluate import evaluate


def infer_candidates_path(eval_path: Path) -> Path:
    name = eval_path.name
    if "_eval_" not in name:
        raise FileNotFoundError(
            f"evaluation file not found: {eval_path}. "
            "Run src.evaluate first, or use an eval filename containing '_eval_' "
            "so the matching candidate filename can be inferred."
        )
    return eval_path.with_name(name.replace("_eval_", "_candidates_", 1))


def ensure_eval_file(
    eval_path: Path,
    tasks_path: Path,
    splits: set[str] | None,
    ks: list[int],
) -> None:
    if eval_path.exists():
        return

    candidates_path = infer_candidates_path(eval_path)
    if not tasks_path.exists():
        raise FileNotFoundError(
            f"evaluation file not found: {eval_path}, and task file not found: {tasks_path}. "
            "Run src.make_tasks first or restore data/tasks.jsonl."
        )
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"evaluation file not found: {eval_path}, and candidate file not found: {candidates_path}. "
            "Regenerate candidates first or restore the matching runs/*candidates*.jsonl file."
        )

    print(f"warning: rebuilt missing evaluation file {eval_path} from {candidates_path}")
    evaluate(
        tasks_path=tasks_path,
        candidates_path=candidates_path,
        output_path=eval_path,
        ks=ks,
        splits=splits,
        only_with_candidates=True,
    )
