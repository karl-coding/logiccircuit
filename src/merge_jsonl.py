from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in args.inputs:
        rows.extend(read_jsonl(path))

    write_jsonl(args.output, rows)
    print(f"merged {len(rows)} rows into {args.output}")


if __name__ == "__main__":
    main()

