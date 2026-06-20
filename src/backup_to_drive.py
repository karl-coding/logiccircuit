from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_PATTERNS = [
    "data/*.jsonl",
    "runs/*.jsonl",
    "runs/*.txt",
    "runs/*adapter*",
    "configs/*.yaml",
]


def should_copy(path: Path) -> bool:
    if path.is_dir():
        return True
    if path.suffix in {".jsonl", ".json", ".yaml", ".yml", ".md", ".txt"}:
        return True
    adapter_names = {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json"}
    return path.name in adapter_names


def copy_path(source: Path, project_root: Path, drive_root: Path) -> int:
    relative = source.relative_to(project_root)
    target = drive_root / relative

    if source.is_dir():
        copied = 0
        for child in source.rglob("*"):
            if child.is_file() and should_copy(child):
                child_target = drive_root / child.relative_to(project_root)
                child_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, child_target)
                copied += 1
        return copied

    if source.is_file() and should_copy(source):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return 1
    return 0


def backup(project_root: Path, drive_root: Path, patterns: list[str]) -> int:
    project_root = project_root.resolve()
    drive_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    seen: set[Path] = set()
    for pattern in patterns:
        for source in project_root.glob(pattern):
            source = source.resolve()
            if source in seen:
                continue
            seen.add(source)
            copied += copy_path(source, project_root, drive_root)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=Path("."), type=Path)
    parser.add_argument(
        "--drive-root",
        default=Path("/content/drive/MyDrive/logiccircuit"),
        type=Path,
    )
    parser.add_argument("--patterns", nargs="+", default=DEFAULT_PATTERNS)
    args = parser.parse_args()

    copied = backup(args.project_root, args.drive_root, args.patterns)
    print(f"copied {copied} files to {args.drive_root}")


if __name__ == "__main__":
    main()
