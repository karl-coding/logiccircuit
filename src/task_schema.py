from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CodeRepairTask:
    id: str
    split: str
    instruction: str
    buggy_code: str
    public_tests: list[str]
    hidden_tests: list[str]
    bug_type: str

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "CodeRepairTask":
        return cls(
            id=str(payload["id"]),
            split=str(payload["split"]),
            instruction=str(payload["instruction"]),
            buggy_code=str(payload["buggy_code"]),
            public_tests=list(payload.get("public_tests", [])),
            hidden_tests=list(payload.get("hidden_tests", [])),
            bug_type=str(payload.get("bug_type", "unknown")),
        )


@dataclass(frozen=True)
class Candidate:
    task_id: str
    rank: int
    code: str

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Candidate":
        return cls(
            task_id=str(payload["task_id"]),
            rank=int(payload["rank"]),
            code=str(payload["code"]),
        )

