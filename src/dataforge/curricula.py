"""Curriculum registration and split composition with governance gates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dataforge.guards import count_pii_matches, leakage_report
from dataforge.rows import normalize_text

CurriculumFunc = Callable[[str], Sequence[dict[str, Any]]]


@dataclass(frozen=True)
class Curriculum:
    name: str
    splits: tuple[str, ...]
    func: CurriculumFunc


class Registry:
    """A collection of curricula, each producing rows for a subset of splits."""

    def __init__(self) -> None:
        self._curricula: list[Curriculum] = []

    def register(
        self, name: str, splits: Sequence[str]
    ) -> Callable[[CurriculumFunc], CurriculumFunc]:
        def decorator(func: CurriculumFunc) -> CurriculumFunc:
            if any(existing.name == name for existing in self._curricula):
                raise ValueError(f"curriculum {name!r} is already registered")
            self._curricula.append(Curriculum(name=name, splits=tuple(splits), func=func))
            return func

        return decorator

    @property
    def curricula(self) -> tuple[Curriculum, ...]:
        return tuple(self._curricula)

    def build(self, split: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for curriculum in self._curricula:
            if split in curriculum.splits:
                rows.extend(curriculum.func(split))
        return rows


_DEFAULT_REGISTRY = Registry()


def curriculum(name: str, splits: Sequence[str]) -> Callable[[CurriculumFunc], CurriculumFunc]:
    """Register a curriculum function on the module-level default registry."""
    return _DEFAULT_REGISTRY.register(name, splits)


def default_registry() -> Registry:
    return _DEFAULT_REGISTRY


def _dedup_across_splits(
    splits: Mapping[str, list[dict[str, Any]]],
    priority: Sequence[str],
    text_field: str,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    if set(priority) != set(splits.keys()):
        raise ValueError("dedup_priority must be a permutation of the split names")
    seen: set[str] = set()
    kept: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    removed = 0
    for split in priority:
        local: set[str] = set()
        for row in splits[split]:
            key = normalize_text(str(row[text_field]))
            if key in seen or key in local:
                removed += 1
                continue
            kept[split].append(row)
            local.add(key)
        seen.update(local)
    return kept, removed


def compose(
    seed_splits: Mapping[str, Sequence[Mapping[str, Any]]],
    registry: Registry,
    *,
    split_order: Sequence[str] = ("train", "validation", "test"),
    dedup_priority: Sequence[str] | None = None,
    text_field: str = "text",
    group_field: str = "group_id",
    trajectory_field: str = "trajectory_id",
    pair_field: str | None = "pair_id",
    count_fields: Sequence[str] = ("example_kind",),
    held_out_texts: Iterable[str] = (),
    ngram_size: int = 4,
    heldout_excluded_splits: Sequence[str] = ("test",),
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Run every registered curriculum, enforce non-leakage, dedup, and report.

    Rows are accumulated split by split (seed rows, then curriculum rows).
    Any ``group_id``/``trajectory_id``/``pair_id`` seen under more than one
    split raises immediately (fail-fast, matching the order examples were
    produced in). The accumulated rows are then deduplicated by normalized
    text across splits: ``dedup_priority`` (default: eval splits before
    train, i.e. ``reversed(split_order)``) decides which split keeps a row
    when the same text appears in more than one. Finally a governance report
    (per-field counts, cross-split duplicate count, PII matches, and a full
    leakage report) is produced over the deduplicated result.
    """
    splits: dict[str, list[dict[str, Any]]] = {
        split: [dict(row) for row in seed_splits.get(split, ())] for split in split_order
    }
    group_to_split: dict[str, str] = {}
    trajectory_to_split: dict[str, str] = {}
    pair_to_split: dict[str, str] = {}

    def _track(row: Mapping[str, Any], split: str) -> None:
        group = str(row[group_field])
        previous = group_to_split.setdefault(group, split)
        if previous != split:
            raise ValueError(f"group {group!r} appears in both {previous} and {split}")
        trajectory = str(row.get(trajectory_field, group))
        previous = trajectory_to_split.setdefault(trajectory, split)
        if previous != split:
            raise ValueError(f"trajectory {trajectory!r} appears in both {previous} and {split}")
        if pair_field:
            pair_value = row.get(pair_field)
            if pair_value:
                previous = pair_to_split.setdefault(str(pair_value), split)
                if previous != split:
                    raise ValueError(
                        f"pair {pair_value!r} appears in both {previous} and {split}"
                    )

    for split in split_order:
        for row in splits[split]:
            _track(row, split)

    for split in split_order:
        for row in registry.build(split):
            _track(row, split)
            splits[split].append(row)

    priority = tuple(dedup_priority) if dedup_priority is not None else tuple(reversed(split_order))
    deduplicated, duplicates_removed = _dedup_across_splits(splits, priority, text_field)

    counts = {
        field: {
            split: dict(Counter(str(row.get(field)) for row in rows))
            for split, rows in deduplicated.items()
        }
        for field in count_fields
    }
    report = {
        "contract": "dataforge-curriculum-report",
        "split_counts": {split: len(rows) for split, rows in deduplicated.items()},
        "counts": counts,
        "cross_split_duplicates_removed": duplicates_removed,
        "pii_matches": count_pii_matches(
            str(row[text_field]) for rows in deduplicated.values() for row in rows
        ),
        "leakage": leakage_report(
            deduplicated,
            group_field=group_field,
            trajectory_field=trajectory_field,
            pair_field=pair_field,
            text_field=text_field,
            held_out_texts=held_out_texts,
            ngram_size=ngram_size,
            heldout_excluded_splits=heldout_excluded_splits,
        ),
    }
    return deduplicated, report
