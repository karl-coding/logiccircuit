from __future__ import annotations

from importlib import metadata
import re


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", version)
    return tuple(int(part) for part in parts[:3])


def disable_incompatible_torchao() -> None:
    """Prevent old Colab torchao packages from breaking PEFT LoRA injection."""
    try:
        version = metadata.version("torchao")
    except metadata.PackageNotFoundError:
        return

    if _version_tuple(version) >= (0, 16, 0):
        return

    def torchao_unavailable() -> bool:
        return False

    try:
        import peft.import_utils as peft_import_utils

        peft_import_utils.is_torchao_available = torchao_unavailable
    except Exception:
        pass

    try:
        import peft.tuners.lora.torchao as peft_lora_torchao

        peft_lora_torchao.is_torchao_available = torchao_unavailable
    except Exception:
        pass

    print(f"warning: disabled incompatible torchao {version}; PEFT will use standard LoRA modules")
