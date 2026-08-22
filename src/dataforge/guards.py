"""PII, held-out, and cross-split leakage guards."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any

from dataforge.rows import canonical_json_bytes, normalize_text, normalize_text_ascii

#: NOTE on the card-number pattern's false-positive class: `\b(?:\d[ -]?){12,19}\b`
#: matches any 12-19 CONSECUTIVE digits (optionally single-space/hyphen separated),
#: regardless of Luhn validity or real card formatting. It will false-positive on
#: any sufficiently long digit-hyphen identifier -- trajectory/timestamp ids with a
#: long numeric suffix, sequence numbers, zero-padded counters -- and, less often,
#: on hex hashes/digests that happen to contain a long uninterrupted digit run (rare
#: in practice: random sha256-hex strings hit this well under 1% of the time, since
#: the a-f hex letters usually break up any 12+ digit streak). If your row schema
#: has identifier fields that legitimately contain long digit runs, either exclude
#: them via `pii_fields` in `dataforge.curricula.build_report`/`compose`, or pass a
#: narrower `patterns` tuple to `count_pii_matches` for that scan.
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


def banned_wording_leaks(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    pattern: re.Pattern[str] | str,
    *,
    text_fields: Sequence[str] = ("text",),
    message_fields: Sequence[str] = (),
    message_roles: Collection[str] = ("user", "assistant"),
    trainable_splits: Sequence[str] = ("train", "validation"),
    group_field: str = "group_id",
) -> dict[str, Any]:
    """Flag banned wording anywhere in the *trainable* text of ``splits``.

    Only splits named in ``trainable_splits`` are scanned, so a frozen split
    is exempt **by construction** rather than by a gate-time exception: a
    held-out evaluation row may deliberately contain the banned phrasing that
    the model must never be trained to produce.

    ``text_fields`` names top-level string fields. ``message_fields`` names
    fields holding a list of message dicts; for each, the ``content`` of every
    message whose ``role`` is in ``message_roles`` **and** whose content is a
    ``str`` is scanned -- a tool-call assistant turn has ``content: None`` and
    is skipped, and ``system``/``tool`` turns are excluded by the default
    ``message_roles``. Passing a rendered full-transcript field (e.g.
    ``text_fields=(), message_fields=("messages",)``) therefore covers context
    turns, the user turn and the final assistant turn in one call.

    One entry per scanned string that matches, with ``term`` set to that
    string's **first** match (``re.search``). The returned keys end in
    ``_leaks``/``_leak_count``, so wiring this through ``extra_leak_checks``
    or ``pre_dedup_checks`` gates the build via
    :func:`dataforge.emit.default_gates` with no other change.
    """
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
    leaks: list[dict[str, str]] = []

    def _scan(split: str, row: Mapping[str, Any], field: str, value: str) -> None:
        match = compiled.search(value)
        if match is not None:
            leaks.append(
                {
                    "split": split,
                    "group_id": str(row.get(group_field, "")),
                    "field": field,
                    "term": match.group(0),
                }
            )

    for split, rows in splits.items():
        if split not in trainable_splits:
            continue
        for row in rows:
            for field in text_fields:
                value = row.get(field)
                if isinstance(value, str):
                    _scan(split, row, field, value)
            for field in message_fields:
                messages = row.get(field)
                if not isinstance(messages, list | tuple):
                    continue
                for message in messages:
                    if not isinstance(message, Mapping):
                        continue
                    if message.get("role") not in message_roles:
                        continue
                    content = message.get("content")
                    if isinstance(content, str):
                        _scan(split, row, field, content)
    return {"banned_wording_leaks": leaks, "banned_wording_leak_count": len(leaks)}


def duplicate_text_leaks(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    field: str,
    *,
    normalize: Callable[[str], str] = normalize_text_ascii,
    exempt: Callable[[Sequence[Mapping[str, Any]]], bool] | None = None,
    splits_in_scope: Sequence[str] | None = None,
    group_field: str = "group_id",
    split_field_name: str = "_split",
) -> dict[str, Any]:
    """Flag any normalized value of ``field`` shared by more than one row.

    Unlike :func:`secondary_field_leaks`, buckets are **global**: rows from
    every split in scope land in the same bucket, so a within-split duplicate
    is flagged just as a cross-split one is. ``splits_in_scope`` (``None``
    meaning every split) restricts which splits contribute rows; rows whose
    ``field`` is missing, non-``str`` or blank are ignored.

    ``exempt`` is offered every bucket of size > 1 and suppresses it by
    returning ``True``. **The rows it receives are shallow copies of the
    bucket's rows with ``split_field_name`` (default ``"_split"``) set to the
    name of the split each row came from**, so an exemption predicate can
    require a structurally proven, single-split construction without the rows
    themselves having to carry their split. The caller's rows are never
    mutated. See :func:`paired_counterfactual_exemption` for the exemption
    this was built for.

    The stamped split is **the key this function iterated**, i.e. where the row
    actually sits. :func:`dataforge.checks.unique_normalized` stamps the same
    field from the record's own ``source_split`` instead, having no split map to
    iterate; the two agree only while every row's ``source_split`` matches its
    real split. When they can diverge, pass ``split_of=`` there so one exemption
    predicate is not answering two different questions.

    Keys are ``{field}_duplicate_leaks`` (entries
    ``{"normalized", "members": [{"split", "group_id"}]}``) and
    ``{field}_duplicate_leak_count``.
    """
    in_scope = set(splits) if splits_in_scope is None else set(splits_in_scope)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for split, rows in splits.items():
        if split not in in_scope:
            continue
        for row in rows:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            stamped = dict(row)
            stamped[split_field_name] = split
            buckets.setdefault(normalize(value), []).append(stamped)

    leaks: list[dict[str, Any]] = []
    for normalized, bucket_rows in buckets.items():
        if len(bucket_rows) < 2:
            continue
        if exempt is not None and exempt(bucket_rows):
            continue
        leaks.append(
            {
                "normalized": normalized,
                "members": [
                    {
                        "split": str(row[split_field_name]),
                        "group_id": str(row.get(group_field, "")),
                    }
                    for row in bucket_rows
                ],
            }
        )
    return {f"{field}_duplicate_leaks": leaks, f"{field}_duplicate_leak_count": len(leaks)}


def paired_counterfactual_exemption(
    *,
    pair_field: str = "pair_id",
    target_field: str = "pair_target",
    context_fields: Sequence[str] = ("history",),
    split_field_name: str = "_split",
) -> Callable[[Sequence[Mapping[str, Any]]], bool]:
    """An ``exempt`` predicate for :func:`duplicate_text_leaks` that accepts a
    bucket **only** when it is a structurally proven set of counterfactual
    pairs -- the one case where repeating the same utterance is the point.

    A bucket is exempt when all of the following hold; anything else is a leak:

    1. its size is even and at least 2;
    2. it partitions by ``pair_field`` into pairs of **exactly two** rows (a
       row with a missing/empty ``pair_field`` disqualifies the bucket);
    3. both rows of every pair carry a **present and distinct** ``target_field``
       value, i.e. the same utterance genuinely resolves differently. A
       missing, ``None`` or empty target disqualifies the bucket rather than
       counting as a second distinct value against its partner's real one --
       it is evidence of a malformed pair, not of a counterfactual;
    4. every row sits in **one** split -- a counterfactual pair is a
       within-split construction, so the same text appearing in two splits is
       leakage, never an exemptible pair. The split is read from
       ``split_field_name``, which :func:`duplicate_text_leaks` stamps on the
       copies it hands to ``exempt``;
    5. the canonical-JSON signature of ``context_fields`` is **distinct across
       pairs** (v9's "distinct history per pair id"), so a bucket cannot be
       exempted by duplicating one pair. The signature covers the pair as a
       whole -- the order-independent set of its two rows' context projections
       -- because the two rows of a counterfactual pair are normally
       distinguished by exactly that context.
    """

    def exemption(bucket_rows: Sequence[Mapping[str, Any]]) -> bool:
        rows = list(bucket_rows)
        if len(rows) < 2 or len(rows) % 2 != 0:
            return False
        if len({row.get(split_field_name) for row in rows}) != 1:
            return False
        pairs: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            pair_value = row.get(pair_field)
            if pair_value is None or pair_value == "":
                return False
            pairs.setdefault(str(pair_value), []).append(row)
        signatures: set[bytes] = set()
        for members in pairs.values():
            if len(members) != 2:
                return False
            targets = [row.get(target_field) for row in members]
            if any(target is None or target == "" for target in targets):
                return False
            if len({canonical_json_bytes(target) for target in targets}) != 2:
                return False
            projections = sorted(
                canonical_json_bytes({name: row.get(name) for name in context_fields}).decode()
                for row in members
            )
            signature = canonical_json_bytes(projections)
            if signature in signatures:
                return False
            signatures.add(signature)
        return True

    return exemption


def secondary_field_leaks(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    field: str,
    *,
    min_tokens: int = 3,
    row_predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, list[str]]:
    """Cross-split leaks of ``field``'s normalized value, independent of
    ``group_id``.

    This is what catches a state-conditioned or paraphrased utterance that
    reuses the same underlying text (e.g. the same ``current_text``) across
    splits under deliberately distinct group ids -- the identifier-based
    checks above never see it because the ids genuinely differ.

    Two scoping knobs keep this from false-positiving on legitimately
    reused short text (e.g. a bare ``"yes"`` clarification answer that
    recurs across splits in genuinely distinct multiturn contexts, each
    disambiguated by its own history):

    ``min_tokens`` (default ``3``): a normalized value shorter than this
    many whitespace-split tokens is never flagged, regardless of how many
    splits it appears in. Three was chosen as the shortest length that
    still reliably indicates a *substantive* reused utterance rather than a
    generic short reply ("yes", "ok", "no thanks") that many unrelated
    multiturn examples can legitimately share verbatim.

    ``row_predicate``: an optional filter (default ``None``, meaning every
    row is considered) restricting the check to rows for which it returns
    ``True`` -- e.g. only rows whose ``example_kind`` is one of a curated
    set of state-conditioned generalization kinds, mirroring how hello-SLM
    scoped this check to specific example kinds rather than every row.

    Raises ``ValueError`` when the ``field`` **key** is absent from every row
    of every split, so a typo in a caller's ``secondary_leak_fields`` fails
    loudly instead of passing vacuously with an empty result. Key absence is
    the error, not "no usable value": a key present with a blank or non-string
    value still proves the field name is real and is simply skipped. Splits
    holding no rows at all carry no signal either way and are not an error.
    """
    rows_seen = False
    field_seen = False
    for rows in splits.values():
        for row in rows:
            rows_seen = True
            if field in row:
                field_seen = True
                break
        if field_seen:
            break
    if rows_seen and not field_seen:
        raise ValueError(
            f"secondary_field_leaks: field {field!r} is absent from every row in every split"
        )

    values: dict[str, set[str]] = {}
    for split, rows in splits.items():
        for row in rows:
            if row_predicate is not None and not row_predicate(row):
                continue
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            normalized = normalize_text(value)
            if len(normalized.split()) < min_tokens:
                continue
            values.setdefault(normalized, set()).add(split)
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
    secondary_leak_min_tokens: int = 3,
    secondary_leak_row_predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Report identifiers and held-out text that appear in more than one split.

    ``secondary_leak_min_tokens``/``secondary_leak_row_predicate`` are
    forwarded to :func:`secondary_field_leaks` for every field in
    ``secondary_leak_fields`` -- see that function for what they scope.
    """
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
        leaks = secondary_field_leaks(
            splits,
            field,
            min_tokens=secondary_leak_min_tokens,
            row_predicate=secondary_leak_row_predicate,
        )
        report[f"{field}_split_leaks"] = leaks
        report[f"{field}_split_leak_count"] = len(leaks)
    return report
