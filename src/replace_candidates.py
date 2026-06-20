from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl, write_jsonl


def replace_candidates(base_path: Path, patch_path: Path, output_path: Path) -> int:
    if not base_path.exists():
        raise FileNotFoundError(f"base candidate file not found: {base_path}")
    if not patch_path.exists():
        raise FileNotFoundError(f"patch candidate file not found: {patch_path}")

    base_rows = read_jsonl(base_path)
    patch_rows = read_jsonl(patch_path)
    patch_task_ids = {str(row["task_id"]) for row in patch_rows}

    rows: list[dict[str, Any]] = [
        row for row in base_rows if str(row["task_id"]) not in patch_task_ids
    ]
    rows.extend(patch_rows)
    write_jsonl(output_path, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--patch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    count = replace_candidates(args.base, args.patch, args.output)
    print(f"wrote {count} merged candidates to {args.output}")


if __name__ == "__main__":
    main()
