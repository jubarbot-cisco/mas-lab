"""Data contract between benchmark readers and the debate engine.

Problems carry the problem, never the protocol: distribution, decisions,
and topology live in ``debate.config``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["DebateProblem", "load_problems"]

_MAX_INPUT_BYTES = 20 * 1024 * 1024

class DebateProblem(BaseModel):
    """One immutable benchmark question validated from untrusted JSON.

    Attributes:
        index: Position in the problem list; must match for reproducible
            seeding.
        task_id: Unique problem identifier.
        question: The question text.
        options: Candidate answer strings.
        correct: Ground truth answer, never shown to agents.
        shared_facts: Facts visible to all agents. Empty when the benchmark
            has no hidden-profile structure.
        hidden_facts: Facts partitioned across agents. Empty when the
            benchmark has no hidden-profile structure.
        meta: Untyped benchmark-specific metadata.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0)
    task_id: str
    question: str
    options: tuple[str, ...]
    correct: str
    shared_facts: tuple[str, ...] = ()
    hidden_facts: tuple[str, ...] = ()
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> DebateProblem:
        if not self.task_id.strip():
            raise ValueError("problem task_id must be non-empty")
        if not self.question.strip():
            raise ValueError(f"problem {self.task_id!r}: question must be non-empty")
        if len(self.options) < 2:
            raise ValueError(f"problem {self.task_id!r}: need at least 2 candidate answers")
        if any(not option.strip() for option in self.options):
            raise ValueError(f"problem {self.task_id!r}: options must not be blank")
        if self.correct not in self.options:
            raise ValueError(f"problem {self.task_id!r}: correct answer must be one of the options")
        return self


RowAdapter = Callable[[int, dict[str, Any]], DebateProblem]


# ------------------------------------------------------------ adapters
#
# Each adapter maps one raw benchmark row to a DebateProblem.
# Signature: (index, row_dict) -> DebateProblem


def _mmlu_adapter(index: int, row: dict[str, Any]) -> DebateProblem:
    """Map one MMLU-Pro row to a ``DebateProblem``."""
    options = row.get("options") or ()
    answer_index = row.get("answer_index")
    correct = str(row.get("answer", "")).strip()
    if isinstance(answer_index, int) and 0 <= answer_index < len(options):
        correct = options[answer_index]
    return DebateProblem(
        index=index,
        task_id=str(row.get("question_id", index)),
        question=row.get("question", ""),
        options=options,
        correct=correct,
        meta={"category": row.get("category"), "src": row.get("src")},
    )


def _hiddenbench_adapter(index: int, row: dict[str, Any]) -> DebateProblem:
    """Map one HiddenBench scenario to a ``DebateProblem``."""
    return DebateProblem(
        index=index,
        task_id=str(row["id"]),
        question=row["description"],
        options=row["possible_answers"],
        correct=row["correct_answer"],
        shared_facts=row["shared_information"],
        hidden_facts=row["hidden_information"],
        meta={"scenario": row.get("name")},
    )


_ADAPTERS: dict[str, RowAdapter] = {
    ".jsonl": _mmlu_adapter,
    ".json": _hiddenbench_adapter,
}


# ------------------------------------------------------------- loader


def load_problems(input_path: str | Path, *, limit: int | None = None) -> list[DebateProblem]:
    """Read a benchmark file into validated problems.

    Handles ``.jsonl`` (MMLU-Pro) and ``.json`` (HiddenBench).

    Args:
        input_path: Path to the benchmark file.
        limit: Keep only the first N problems, or ``None`` for all.

    Returns:
        Validated and optionally truncated problem list.

    Raises:
        FileNotFoundError: If the benchmark file does not exist.
        ValueError: If the suffix is unsupported or the content is invalid.
    """
    path = Path(input_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"benchmark not found: {path}")
    if path.stat().st_size > _MAX_INPUT_BYTES:
        raise ValueError(f"{path}: input exceeds {_MAX_INPUT_BYTES} bytes")

    adapter = _ADAPTERS.get(path.suffix)
    if adapter is None:
        raise ValueError(f"{path}: unsupported suffix {path.suffix!r}")

    rows = _load_rows(path)
    problems = [adapter(i, row) for i, row in enumerate(rows)]

    # Validate index alignment and unique task_ids.
    seen: set[str] = set()
    for position, problem in enumerate(problems):
        if problem.index != position:
            raise ValueError(
                f"problem {problem.task_id!r} carries index {problem.index} "
                f"but sits at position {position}"
            )
        if problem.task_id in seen:
            raise ValueError(f"duplicate task_id {problem.task_id!r}")
        seen.add(problem.task_id)

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive or None")
        problems = [
            p.model_copy(update={"index": i})
            for i, p in enumerate(problems[:limit])
        ]

    return problems


# --------------------------------------------------------------- I/O


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Load raw dicts from ``.jsonl`` (one per line) or ``.json`` (a list)."""
    with path.open(encoding="utf-8") as fh:
        if path.suffix == ".jsonl":
            rows: list[dict[str, Any]] = []
            for line_number, line in enumerate(fh, start=1):
                if line := line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
                    task = row.get("raw_task", row)
                    if not isinstance(task, dict):
                        raise ValueError(f"{path}:{line_number}: raw_task must be an object")
                    rows.append(task)
            return rows
        rows = json.load(fh)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"{path}: JSON benchmark must be a list of objects")
        return rows
