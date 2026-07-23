"""Deterministic JSON, JSONL, and CSV helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class DataError(ValueError):
    """Raised for malformed or missing study artifacts."""


def stable_numbers(value: object) -> object:
    """Round finite floats for byte-stable artifacts across numerical runtimes."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("machine artifacts require finite floats")
        return float(f"{value:.12g}")
    if isinstance(value, Mapping):
        return {str(key): stable_numbers(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [stable_numbers(item) for item in value]
    return value


def canonical_json(value: object) -> str:
    return json.dumps(
        stable_numbers(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read JSON {path}: {exc}") from exc


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DataError(f"cannot read JSONL {path}: {exc}") from exc
    records: list[T] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except ValueError as exc:
            raise DataError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise DataError(f"{path} contains no records")
    return records


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            stable_numbers(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: list[str],
) -> None:
    if not rows:
        raise ValueError("CSV output must contain at least one row")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [
                {key: stable_numbers(value) for key, value in row.items()}
                for row in rows
            ]
        )
