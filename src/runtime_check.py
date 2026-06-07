from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:
    yaml = None


@dataclass(frozen=True)
class RuntimeReport:
    python: str
    platform: str
    torch_available: bool
    cuda_available: bool
    gpu_name: str | None
    gpu_memory_gb: float | None
    a100_ready: bool
    warnings: list[str]


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None or yaml is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def inspect_runtime(config_path: Path | None = None) -> RuntimeReport:
    warnings: list[str] = []
    _load_config(config_path)

    try:
        import torch

        torch_available = True
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_memory_gb = round(props.total_memory / (1024**3), 2)
        else:
            gpu_name = None
            gpu_memory_gb = None
            warnings.append("CUDA is not available. Select an A100 GPU runtime in Colab.")
    except Exception as exc:
        torch_available = False
        cuda_available = False
        gpu_name = None
        gpu_memory_gb = None
        warnings.append(f"torch unavailable or failed to load: {exc}")

    a100_ready = bool(cuda_available and gpu_memory_gb is not None and gpu_memory_gb >= 32)
    if cuda_available and not a100_ready:
        warnings.append("Detected GPU memory is below 32 GB. This project is configured for A100.")
    if gpu_name and "A100" not in gpu_name and a100_ready:
        warnings.append("GPU has enough memory, but name is not A100. Results may differ from the A100 profile.")

    return RuntimeReport(
        python=platform.python_version(),
        platform=platform.platform(),
        torch_available=torch_available,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        a100_ready=a100_ready,
        warnings=warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/a100_10h.yaml"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = inspect_runtime(args.config)
    if args.json:
        print(json.dumps(asdict(report), indent=2, ensure_ascii=True))
        return

    print(f"python: {report.python}")
    print(f"platform: {report.platform}")
    print(f"torch_available: {report.torch_available}")
    print(f"cuda_available: {report.cuda_available}")
    print(f"gpu_name: {report.gpu_name}")
    print(f"gpu_memory_gb: {report.gpu_memory_gb}")
    print(f"a100_ready: {report.a100_ready}")
    for warning in report.warnings:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()

