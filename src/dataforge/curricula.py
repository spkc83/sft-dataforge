"""Curriculum registration and split composition with governance gates."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dataforge.guards import count_pii_matches, leakage_report
from dataforge.rows import canonical_json_bytes, normalize_text

CurriculumFunc = Callable[[str], Sequence[dict[str, Any]]]

#: Default split iteration/emission order. Purely about ordering -- it does
#: NOT determine which split wins deduplication; see DEFAULT_DEDUP_PRIORITY.
DEFAULT_SPLIT_ORDER: tuple[str, ...] = ("train", "validation", "test")

#: Default cross-split dedup priority: eval splits win over train. This is a
#: fixed constant, deliberately NOT derived from split_order, so reordering
#: split_order (a cosmetic/emission-order knob) can never silently flip which
#: split keeps a duplicated example.
DEFAULT_DEDUP_PRIORITY: tuple[str, ...] = ("test", "validation", "train")

#: The report contract produced by build_report()/compose(). write_dataset()
#: requires a matching contract by default so it never gates release on a
#: report it can't interpret.
REPORT_CONTRACT = "dataforge-curriculum-report"

LeakCheck = Callable[[Mapping[str, Sequence[Mapping[str, Any]]]], Mapping[str, Any]]


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


def _iter_strings(value: Any) -> Iterable[str]:
    """Recursively yield every string leaf inside ``value``.

    Used for PII scanning so a field that happens to be a nested dict or
    list (e.g. ``prior_state``, ``provenance``, a ``history`` list of
    ``{"role": ..., "content": ...}`` turns) is scanned as thoroughly as a
    flat string field -- PII does not respect a row's schema.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list | tuple | set | frozenset):
        for item in value:
            yield from _iter_strings(item)


def splits_fingerprint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    """A content fingerprint over every field of every row in ``splits``.

    :func:`build_report` records this as ``report["splits_fingerprint"]``,
    and :func:`dataforge.emit.write_dataset` recomputes it over the exact
    ``splits`` it is about to write and refuses to gate/emit on a mismatch.
    This is what makes a stale report (e.g. one built before a teacher
    realization pass edited row content) a hard error instead of a silent
    gate bypass -- the report must describe the bytes actually emitted.
    """
    payload = {split: list(rows) for split, rows in splits.items()}
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def _merge_leakage(
    leakage: dict[str, Any], result: Mapping[str, Any], *, source: str
) -> dict[str, Any]:
    """Merge one leak check's result into ``leakage``, refusing collisions.

    Shared by ``build_report``'s ``extra_leak_checks`` and ``compose``'s
    ``pre_dedup_checks`` so both obey one rule: a check may never overwrite a
    built-in leakage key or another check's key -- silently shadowing a
    ``*_leaks``/``*_leak_count`` key would disarm the gate that reads it.
    """
    overlap = set(result) & set(leakage)
    if overlap:
        raise ValueError(f"{source} produced colliding report keys: {sorted(overlap)}")
    leakage.update(result)
    return leakage


def _failing_leak_keys(leakage: Mapping[str, Any]) -> list[str]:
    """Keys that :func:`dataforge.emit.default_gates` would fail the build on."""
    return sorted(
        key
        for key, value in leakage.items()
        if (key.endswith("_leak_count") or key.endswith("_leaks")) and value
    )


def _dedup_across_splits(
    splits: Mapping[str, list[dict[str, Any]]],
    priority: Sequence[str],
    text_field: str,
) -> tuple[dict[str, list[dict[str, Any]]], int, int]:
    """Drop duplicate ``text_field`` values, ``priority``'s earlier splits winning.

    Returns ``(kept, removed, within_split_removed)``. ``removed`` counts every
    dropped row, cross-split and within-split alike -- its historical meaning,
    kept unchanged so no caller's ``cross_split_duplicates_removed`` shifts
    value. ``within_split_removed`` is a sub-count of it: rows dropped because
    the same text already appeared **earlier in their own split**. Neither
    number gates anything; they are report metadata.
    """
    if set(priority) != set(splits.keys()):
        raise ValueError("dedup_priority must be a permutation of the split names")
    seen: set[str] = set()
    kept: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}
    removed = 0
    within_split_removed = 0
    for split in priority:
        local: set[str] = set()
        for row in splits[split]:
            key = normalize_text(str(row[text_field]))
            if key in seen:
                removed += 1
                continue
            if key in local:
                removed += 1
                within_split_removed += 1
                continue
            kept[split].append(row)
            local.add(key)
        seen.update(local)
    return kept, removed, within_split_removed


def build_report(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    text_field: str = "text",
    group_field: str = "group_id",
    trajectory_field: str = "trajectory_id",
    pair_field: str | None = "pair_id",
    count_fields: Sequence[str] = ("example_kind",),
    held_out_texts: Iterable[str] = (),
    ngram_size: int = 4,
    heldout_excluded_splits: Sequence[str] = ("test",),
    secondary_leak_fields: Sequence[str] = ("current_text",),
    secondary_leak_min_tokens: int = 3,
    secondary_leak_row_predicate: Callable[[Mapping[str, Any]], bool] | None = None,
    extra_leak_checks: Sequence[LeakCheck] = (),
    pii_fields: Sequence[str] | None = None,
    cross_split_duplicates_removed: int = 0,
    within_split_duplicates_removed: int = 0,
) -> dict[str, Any]:
    """Build the governance report for ``splits`` as they stand right now.

    This is what :func:`compose` calls internally after curricula + dedup,
    but it is also meant to be called again -- on the same config -- by any
    caller that mutates row content after ``compose`` (most commonly, a
    teacher-realization pass rewriting wording fields). The manifest's
    governance report must describe the bytes actually emitted, so gating on
    a report computed before such a mutation is a bug: rebuild the report
    on the post-mutation ``splits`` before calling
    :func:`dataforge.emit.write_dataset`.

    ``pii_fields``: ``None`` (the default) recursively scans every string
    leaf reachable from every field of every row for PII -- including
    strings nested inside dict/list-valued fields like ``prior_state``,
    ``provenance``, or ``history`` -- not just ``text_field``. A teacher
    editing e.g. an ``assistant_response`` field, or metadata carrying a
    free-text note, can introduce PII just as easily as a curriculum
    editing ``text``. Pass an explicit field list to scan only those
    fields (each is still scanned recursively, so a listed dict/list field
    is fully covered, not just its top level).

    ``secondary_leak_fields``: in addition to identifier-based leakage
    (group/trajectory/pair), also flag any of these fields whose normalized
    value appears in more than one split -- even under different
    ``group_id``s. Defaults to ``("current_text",)``, restoring hello-SLM's
    state-conditioned-utterance leak coverage generically (no hardcoded
    example-kind names or group-id prefixes required). ``secondary_leak_min_tokens``
    (default ``3``) and ``secondary_leak_row_predicate`` scope this check --
    see :func:`dataforge.guards.secondary_field_leaks` -- so short, widely
    and legitimately reused text (e.g. a bare ``"yes"`` clarification
    answer) doesn't trip it.

    ``extra_leak_checks``: additional callables, each given ``splits`` and
    returning a dict merged into ``report["leakage"]``. Lets a domain team
    add its own leak checks (e.g. a paraphrase-family check) without
    mutating this module or the returned report's other keys; any key
    ending in ``_leak_count``/``_leaks`` gates automatically via
    :func:`dataforge.emit.default_gates`. Raises if a check's keys collide
    with the built-in leakage keys or with another check's keys.

    ``cross_split_duplicates_removed``/``within_split_duplicates_removed`` are
    reported verbatim (``build_report`` cannot recompute them -- only
    :func:`compose` sees the pre-dedup rows). The first counts every row dedup
    dropped, the second is its within-split sub-count. Neither gates.
    """
    counts = {
        field: {
            split: dict(Counter(str(row.get(field)) for row in rows))
            for split, rows in splits.items()
        }
        for field in count_fields
    }

    leakage = dict(
        leakage_report(
            splits,
            group_field=group_field,
            trajectory_field=trajectory_field,
            pair_field=pair_field,
            text_field=text_field,
            held_out_texts=held_out_texts,
            ngram_size=ngram_size,
            heldout_excluded_splits=heldout_excluded_splits,
            secondary_leak_fields=secondary_leak_fields,
            secondary_leak_min_tokens=secondary_leak_min_tokens,
            secondary_leak_row_predicate=secondary_leak_row_predicate,
        )
    )
    for check in extra_leak_checks:
        _merge_leakage(leakage, check(splits), source="extra_leak_checks")

    if pii_fields is None:
        pii_texts: Iterable[str] = (
            text
            for rows in splits.values()
            for row in rows
            for text in _iter_strings(row)
        )
    else:
        pii_texts = (
            text
            for rows in splits.values()
            for row in rows
            for field in pii_fields
            if field in row
            for text in _iter_strings(row[field])
        )

    return {
        "contract": REPORT_CONTRACT,
        "splits_fingerprint": splits_fingerprint(splits),
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "counts": counts,
        "cross_split_duplicates_removed": cross_split_duplicates_removed,
        "within_split_duplicates_removed": within_split_duplicates_removed,
        "pii_matches": count_pii_matches(pii_texts),
        "leakage": leakage,
    }


def compose(
    seed_splits: Mapping[str, Sequence[Mapping[str, Any]]],
    registry: Registry,
    *,
    split_order: Sequence[str] = DEFAULT_SPLIT_ORDER,
    dedup_priority: Sequence[str] | None = None,
    text_field: str = "text",
    group_field: str = "group_id",
    trajectory_field: str = "trajectory_id",
    pair_field: str | None = "pair_id",
    count_fields: Sequence[str] = ("example_kind",),
    held_out_texts: Iterable[str] = (),
    ngram_size: int = 4,
    heldout_excluded_splits: Sequence[str] = ("test",),
    secondary_leak_fields: Sequence[str] = ("current_text",),
    secondary_leak_min_tokens: int = 3,
    secondary_leak_row_predicate: Callable[[Mapping[str, Any]], bool] | None = None,
    extra_leak_checks: Sequence[LeakCheck] = (),
    pre_dedup_checks: Sequence[LeakCheck] = (),
    pii_fields: Sequence[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Run every registered curriculum, enforce non-leakage, dedup, and report.

    Every key in ``seed_splits`` and every split any registered curriculum
    targets must be in ``split_order``; anything else raises immediately
    rather than being silently dropped.

    Rows are accumulated split by split (seed rows, then curriculum rows).
    Any ``group_id``/``trajectory_id``/``pair_id`` seen under more than one
    split raises immediately (fail-fast, matching the order examples were
    produced in). The accumulated rows are then deduplicated by normalized
    text across splits: ``dedup_priority`` decides which split keeps a row
    when the same text appears in more than one, and defaults to
    :data:`DEFAULT_DEDUP_PRIORITY` (eval splits before train) -- a fixed
    constant, independent of ``split_order``'s permutation, so reordering
    ``split_order`` can never silently change which split wins. When
    ``split_order`` uses non-default split names, pass ``dedup_priority``
    explicitly; otherwise ``compose`` raises rather than guess.

    ``pre_dedup_checks`` run on the raw accumulated splits -- after the
    leak-tracking pass, **before** deduplication -- and fail the build fast:
    if any of their merged keys ending in ``_leak_count`` is truthy or ending
    in ``_leaks`` is non-empty, ``compose`` raises ``ValueError("pre-dedup
    check failed: ...")``, exactly as an id collision does. They exist because
    dedup on ``text_field`` runs before :func:`build_report`, so a report-time
    check can never observe a within-split ``text`` duplicate -- by then it has
    already been silently dropped. Failing here (rather than reporting) also
    means a post-teacher ``build_report`` rebuild needs no re-injection and
    :func:`dataforge.emit.write_dataset` can never see a build these checks
    rejected. Their keys are merged into ``report["leakage"]`` for the record,
    under the same collision rule as ``extra_leak_checks``.

    Finally, :func:`build_report` produces the governance report over the
    deduplicated result.
    """
    split_order = tuple(split_order)

    unknown_seed_splits = set(seed_splits) - set(split_order)
    if unknown_seed_splits:
        raise ValueError(
            f"seed_splits has splits not in split_order: {sorted(unknown_seed_splits)}"
        )
    unknown_curriculum_splits = {
        split for registered in registry.curricula for split in registered.splits
    } - set(split_order)
    if unknown_curriculum_splits:
        raise ValueError(
            "registered curricula target splits not in split_order: "
            f"{sorted(unknown_curriculum_splits)}"
        )

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

    pre_dedup_leakage: dict[str, Any] = {}
    for check in pre_dedup_checks:
        _merge_leakage(pre_dedup_leakage, check(splits), source="pre_dedup_checks")
    failing = _failing_leak_keys(pre_dedup_leakage)
    if failing:
        raise ValueError(
            f"pre-dedup check failed: {failing} -- "
            + ", ".join(f"{key}={pre_dedup_leakage[key]!r}" for key in failing)
        )

    priority = tuple(dedup_priority) if dedup_priority is not None else DEFAULT_DEDUP_PRIORITY
    if set(priority) != set(split_order):
        raise ValueError(
            "dedup_priority must be a permutation of split_order; got "
            f"dedup_priority={priority!r} for split_order={split_order!r}. Pass "
            "dedup_priority explicitly when using non-default split names."
        )
    deduplicated, duplicates_removed, within_split_removed = _dedup_across_splits(
        splits, priority, text_field
    )

    report = build_report(
        deduplicated,
        text_field=text_field,
        group_field=group_field,
        trajectory_field=trajectory_field,
        pair_field=pair_field,
        count_fields=count_fields,
        held_out_texts=held_out_texts,
        ngram_size=ngram_size,
        heldout_excluded_splits=heldout_excluded_splits,
        secondary_leak_fields=secondary_leak_fields,
        secondary_leak_min_tokens=secondary_leak_min_tokens,
        secondary_leak_row_predicate=secondary_leak_row_predicate,
        extra_leak_checks=extra_leak_checks,
        pii_fields=pii_fields,
        cross_split_duplicates_removed=duplicates_removed,
        within_split_duplicates_removed=within_split_removed,
    )
    _merge_leakage(report["leakage"], pre_dedup_leakage, source="pre_dedup_checks")
    return deduplicated, report
