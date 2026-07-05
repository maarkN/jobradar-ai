"""In-memory registry of ingest runs (single-process MVP)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Run:
    run_id: str
    state: str = "running"  # running | done | error
    result: dict[str, int] | None = None
    error: str | None = None


class RunRegistry:
    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._counter = 0

    def create(self) -> Run:
        self._counter += 1
        run = Run(run_id=f"run-{self._counter}")
        self._runs[run.run_id] = run
        return run

    def complete(self, run_id: str, result: dict[str, int]) -> None:
        self._runs[run_id].state = "done"
        self._runs[run_id].result = result

    def fail(self, run_id: str, error: str) -> None:
        self._runs[run_id].state = "error"
        self._runs[run_id].error = error

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)
