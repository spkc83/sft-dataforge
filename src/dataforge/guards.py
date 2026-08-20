"""PII, held-out, and cross-split leakage guards."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from dataforge.rows import normalize_text

PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # email
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-shaped
    re.compile(r"\b(?:\d[ -]?){12,19}\b"),  # card-number-shaped digit run
    re.compile(r"\b(?:\+?1[ -.]?)?(?:\(?\d{3}\)?[ -.]?)\d{3}[ -.]\d{4}\b"),  # phone
)


def count_pii_matches(
    texts: Iterable[str], *, patterns: Sequence[re.Pattern[str]] = PII_PATTERNS
) -> int:
    return sum(1 for text in texts for pattern in patterns for _ in pattern.finditer(text))


def word_ngrams(text: str, *, size: int = 4) -> set[tuple[str, ...]]:
    tokens = normalize_text(text).split()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def is_heldout_text(text: str, held_out_texts: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(prompt) == normalized for prompt in held_out_texts)


def contains_heldout_ngram(text: str, held_out_texts: Iterable[str], *, size: int = 4) -> bool:
    held_out_texts = list(held_out_texts)
    if not held_out_texts:
        return False
    heldout_ngrams: set[tuple[str, ...]] = set().union(
        *(word_ngrams(prompt, size=size) for prompt in held_out_texts)
    )
    return bool(heldout_ngrams & word_ngrams(text, size=size))


def heldout_leaks(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    held_out_texts: Iterable[str],
    *,
    text_field: str = "text",
    group_field: str = "group_id",
    ngram_size: int = 4,
    excluded_splits: Sequence[str] = ("test",),
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Find held-out text that leaked (exactly or via a shared long n-gram) into
    splits other than ``excluded_splits`` (normally the held-out material's home
    split, e.g. ``"test"``)."""
    held_out_texts = list(held_out_texts)
    exact: list[dict[str, str]] = []
    long_ngram: list[dict[str, Any]] = []
    if not held_out_texts:
        return exact, long_ngram
    normalized_heldout = {normalize_text(prompt) for prompt in held_out_texts}
    heldout_ngrams: set[tuple[str, ...]] = set().union(
        *(word_ngrams(prompt, size=ngram_size) for prompt in held_out_texts)
    )
    for split, rows in splits.items():
        if split in excluded_splits:
            continue
        for row in rows:
            text = str(row.get(text_field, ""))
            normalized = normalize_text(text)
            matching = sorted(prompt for prompt in normalized_heldout if prompt in normalized)
            if matching:
                exact.append(
                    {
                        "split": split,
                        "group_id": str(row.get(group_field, "")),
                        "prompt": matching[0],
                    }
                )
            shared = sorted(heldout_ngrams & word_ngrams(text, size=ngram_size))
            if shared:
                long_ngram.append(
                    {
                        "split": split,
                        "group_id": str(row.get(group_field, "")),
                        "ngrams": [" ".join(ngram) for ngram in shared],
                    }
                )
    return exact, long_ngram


def secondary_field_leaks(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    field: str,
) -> dict[str, list[str]]:
    """Cross-split leaks of ``field``'s normalized value, independent of
    ``group_id``.

    This is what catches a state-conditioned or paraphrased utterance that
    reuses the same underlying text (e.g. the same ``current_text``) across
    splits under deliberately distinct group ids -- the identifier-based
    checks above never see it because the ids genuinely differ.
    """
    values: dict[str, set[str]] = {}
    for split, rows in splits.items():
        for row in rows:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            values.setdefault(normalize_text(value), set()).add(split)
    return {key: sorted(vals) for key, vals in values.items() if len(vals) > 1}


def leakage_report(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    group_field: str = "group_id",
    trajectory_field: str = "trajectory_id",
    pair_field: str | None = "pair_id",
    text_field: str = "text",
    held_out_texts: Iterable[str] = (),
    ngram_size: int = 4,
    heldout_excluded_splits: Sequence[str] = ("test",),
    secondary_leak_fields: Sequence[str] = (),
) -> dict[str, Any]:
    """Report identifiers and held-out text that appear in more than one split."""
    groups: dict[str, set[str]] = {}
    trajectories: dict[str, set[str]] = {}
    pairs: dict[str, set[str]] = {}
    for split, rows in splits.items():
        for row in rows:
            groups.setdefault(str(row[group_field]), set()).add(split)
            trajectory = str(row.get(trajectory_field, row[group_field]))
            trajectories.setdefault(trajectory, set()).add(split)
            if pair_field:
                pair_value = row.get(pair_field)
                if pair_value:
                    pairs.setdefault(str(pair_value), set()).add(split)

    group_leaks = {key: sorted(values) for key, values in groups.items() if len(values) > 1}
    trajectory_leaks = {
        key: sorted(values) for key, values in trajectories.items() if len(values) > 1
    }
    pair_leaks = {key: sorted(values) for key, values in pairs.items() if len(values) > 1}
    exact_leaks, ngram_leaks = heldout_leaks(
        splits,
        held_out_texts,
        text_field=text_field,
        group_field=group_field,
        ngram_size=ngram_size,
        excluded_splits=heldout_excluded_splits,
    )
    report: dict[str, Any] = {
        "group_split_leaks": group_leaks,
        "group_split_leak_count": len(group_leaks),
        "trajectory_split_leaks": trajectory_leaks,
        "trajectory_split_leak_count": len(trajectory_leaks),
        "pair_split_leaks": pair_leaks,
        "pair_split_leak_count": len(pair_leaks),
        "heldout_exact_leaks": exact_leaks,
        "heldout_ngram_leaks": ngram_leaks,
    }
    for field in secondary_leak_fields:
        leaks = secondary_field_leaks(splits, field)
        report[f"{field}_split_leaks"] = leaks
        report[f"{field}_split_leak_count"] = len(leaks)
    return report
