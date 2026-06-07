from __future__ import annotations

import multiprocessing as mp
import textwrap
import traceback
from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationResult:
    passed_public: bool
    passed_hidden: bool
    error: str | None = None

    @property
    def passed_all(self) -> bool:
        return self.passed_public and self.passed_hidden


def _run_code(code: str, tests: list[str], queue: mp.Queue) -> None:
    namespace: dict[str, object] = {}
    try:
        exec(textwrap.dedent(code), namespace)
        for test in tests:
            exec(test, namespace)
        queue.put((True, None))
    except Exception:
        queue.put((False, traceback.format_exc(limit=2)))


def run_tests(code: str, tests: list[str], timeout_seconds: float = 2.0) -> tuple[bool, str | None]:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_run_code, args=(code, tests, queue))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return False, "timeout"

    if queue.empty():
        return False, "no verifier result"

    passed, error = queue.get()
    return bool(passed), error


def verify_candidate(
    code: str,
    public_tests: list[str],
    hidden_tests: list[str],
    timeout_seconds: float = 2.0,
) -> VerificationResult:
    passed_public, public_error = run_tests(code, public_tests, timeout_seconds)
    if not passed_public:
        return VerificationResult(False, False, public_error)

    passed_hidden, hidden_error = run_tests(code, hidden_tests, timeout_seconds)
    return VerificationResult(True, passed_hidden, hidden_error)

