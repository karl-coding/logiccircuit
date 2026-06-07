from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import read_jsonl, write_jsonl
from .task_schema import CodeRepairTask


def build_prompt(task: CodeRepairTask) -> str:
    public_tests = "\n".join(task.public_tests)
    return f"""\
{task.instruction}

Buggy code:
```python
{task.buggy_code}
```

Tests:
```python
{public_tests}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for row in read_jsonl(args.tasks):
        task = CodeRepairTask.from_json(row)
        rows.append({"task_id": task.id, "split": task.split, "prompt": build_prompt(task)})
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()

