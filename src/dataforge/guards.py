"""PII, held-out, and cross-split leakage guards."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any

from dataforge.rows import canonical_json_bytes, normalize_text, normalize_text_ascii

#: One field invariant: given a field's raw value and the row it came from,
#: return a violation description, or ``None`` when the value is fine. The
#: factories below (:func:`no_digits`, :func:`banned_patterns`, ...) each build
#: one; :func:`field_invariant_leaks` runs a sequence of them over a corpus.
FieldInvariant = Callable[[str, Mapping[str, Any]], str | None]

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


def fuzzy_duplicate_leaks(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    field: str,
    threshold: float = 0.995,
    group_fn: Callable[[Mapping[str, Any]], Any] | None = None,
    splits_checked: Sequence[str] = ("train", "validation"),
) -> dict[str, Any]:
    """Flag pairs of rows whose normalized ``field`` values are *nearly* -- but
    not exactly -- the same.

    Pairs that normalize to the *same* value are skipped: those are
    :func:`duplicate_text_leaks`' job. What this one catches is the pair an
    exact-match check cannot see -- two rows a generator emitted from one
    template that differ by a comma, a name, or a trailing word. At training
    scale those are the same example twice: the model gets the repetition
    without the variety the second row was supposed to buy.

    The split of work between the two checks is close to a partition but not
    exactly one, and the difference is the normalizer. This function uses
    :func:`dataforge.rows.normalize_text` (Unicode-aware) while
    ``duplicate_text_leaks`` defaults to
    :func:`dataforge.rows.normalize_text_ascii` (every non-``[a-z0-9]`` run
    becomes a space). A pair that is equal under the ASCII normalizer but not
    under this one -- ``"café"``/``"cafe"`` -- is an exact duplicate there and a
    near-duplicate here, and is reported by both.

    Comparison is within one split and, when ``group_fn`` is given, within one
    group: ``group_fn(row)`` returns the group key, and only rows sharing it are
    compared. Both restrictions are cost bounds on an O(n^2) pass, not claims
    about where duplicates live -- group by the scenario family, or whatever
    partition makes two rows comparable at all, and the quadratic term is per
    family rather than per corpus. **A cross-split near-duplicate is therefore
    not covered by anything here**: :func:`secondary_field_leaks` buckets on
    exact normalized equality and will not see it either. Pass
    ``splits_checked=(one_split,)`` per split if that matters, or run the pass
    over a merged mapping.

    Rows whose ``field`` is missing, non-``str`` or blank are skipped. ``ratio``
    is ``difflib.SequenceMatcher.ratio()`` rounded to four places, and the cheap
    ``real_quick_ratio``/``quick_ratio`` upper bounds short-circuit the expensive
    comparison first.

    The key is ``fuzzy_duplicate_leaks`` (entries ``{"split", "group",
    "index_a", "index_b", "ratio"}``, the indexes being positions in that
    split's row sequence), so it gates via :func:`dataforge.emit.default_gates`
    like every other ``*_leaks`` key. There is deliberately no ``_leak_count``
    companion: the list alone gates, and a second key could only disagree with it.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"fuzzy_duplicate_leaks: threshold must be in (0, 1], got {threshold!r}")
    leaks: list[dict[str, Any]] = []
    for split, rows in splits.items():
        if split not in splits_checked:
            continue
        groups: dict[Any, list[tuple[int, str]]] = {}
        for index, row in enumerate(rows):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            normalized = normalize_text(value)
            if not normalized:
                continue
            key = "" if group_fn is None else group_fn(row)
            groups.setdefault(key, []).append((index, normalized))
        for group, members in groups.items():
            for position, (index_a, value_a) in enumerate(members):
                for index_b, value_b in members[position + 1 :]:
                    if value_a == value_b:
                        continue
                    matcher = SequenceMatcher(a=value_a, b=value_b, autojunk=False)
                    if matcher.real_quick_ratio() < threshold:
                        continue
                    if matcher.quick_ratio() < threshold:
                        continue
                    ratio = matcher.ratio()
                    if ratio < threshold:
                        continue
                    leaks.append(
                        {
                            "split": split,
                            "group": str(group),
                            "index_a": index_a,
                            "index_b": index_b,
                            "ratio": round(ratio, 4),
                        }
                    )
    return {"fuzzy_duplicate_leaks": leaks}


def _require_field_present(
    guard: str,
    checked: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    fields: Sequence[str],
) -> None:
    """Raise unless some checked row carries at least one name in ``fields``.

    The typo guard the invariant guards share, and the rule
    :func:`secondary_field_leaks` already applies: a field name that exists on
    no row is a disarmed gate that keeps reporting a reassuring zero. Key
    absence is the error, not "no usable value" -- a key present with a blank or
    non-``str`` value still proves the name is real. A corpus with no checked
    rows at all carries no signal either way and is not an error.
    """
    if not any(rows for _, rows in checked):
        return
    if any(field in row for _, rows in checked for row in rows for field in fields):
        return
    if len(fields) == 1:
        raise ValueError(f"{guard}: field {fields[0]!r} is absent from every checked row")
    names = ", ".join(repr(field) for field in fields)
    raise ValueError(f"{guard}: none of the fields {names} is present on any checked row")


def _normalized_unique(values: Iterable[str]) -> list[str]:
    """Normalized, non-blank, order-preserving and de-duplicated."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = normalize_text(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def probe_exclusion_leaks(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    probes: Iterable[str] = (),
    fragments: Iterable[str] = (),
    fields: Sequence[str] = ("text",),
    splits_checked: Sequence[str] = ("train", "validation"),
) -> dict[str, Any]:
    """Flag training text that reproduces an evaluation probe.

    An evaluation probe is only held out if something proves it was held out.
    Without a gate, "the model passed the probe" is a claim about memorization
    as easily as about generalization, and nothing in the build can tell the two
    apart after the fact. This is the gate: run it over the trainable splits
    with the probes the model will be scored on, and a probe that made its way
    into training fails the build instead of flattering the eval.

    ``probes`` are whole probe texts: a row leaks when a checked field's
    normalized value **equals** a normalized probe. ``fragments`` are the
    shorter distinctive phrases a probe is recognizable by (a paraphrased probe
    keeps them even when the whole text no longer matches): a row leaks when a
    normalized field value **contains** a normalized fragment. Both are matched
    on :func:`dataforge.rows.normalize_text`, so casing, punctuation and spacing
    do not hide a leak. Passing neither raises -- a probe gate with nothing to
    look for would pass vacuously, which is the failure it exists to prevent.

    ``fields`` decides the surface: the row's own utterance field catches an
    exact probe, while a rendered-transcript field also catches a probe phrase
    that arrived through a context turn. Raises when none of those names is
    present on any checked row -- like the empty-probe case, a field-name typo
    would otherwise leave the gate reporting a reassuring zero forever.
    ``splits_checked`` defaults to the trainable splits, so the probes' own
    frozen split is exempt by construction.

    A field that matched a whole probe is not then searched for fragments: the
    probe is the finding, and its fragments are by construction inside it, so
    reporting both would be one leak counted twice.

    The key is ``probe_exclusion_leaks`` (entries ``{"split", "index", "field",
    "kind", "value"}``, where ``kind`` is ``"probe"`` or ``"fragment"`` and
    ``value`` is the normalized probe or fragment that matched). There is
    deliberately no ``_leak_count`` companion: the list alone gates, and a
    second key could only disagree with it.
    """
    normalized_probes = _normalized_unique(probes)
    normalized_fragments = _normalized_unique(fragments)
    if not normalized_probes and not normalized_fragments:
        raise ValueError(
            "probe_exclusion_leaks: pass probes and/or fragments -- a probe gate with nothing "
            "to look for passes vacuously"
        )
    checked = [(split, rows) for split, rows in splits.items() if split in splits_checked]
    _require_field_present("probe_exclusion_leaks", checked, fields)
    probe_set = set(normalized_probes)
    leaks: list[dict[str, Any]] = []
    for split, rows in checked:
        for index, row in enumerate(rows):
            for field in fields:
                value = row.get(field)
                if not isinstance(value, str):
                    continue
                normalized = normalize_text(value)
                if not normalized:
                    continue
                if normalized in probe_set:
                    leaks.append(
                        {
                            "split": split,
                            "index": index,
                            "field": field,
                            "kind": "probe",
                            "value": normalized,
                        }
                    )
                    continue
                for fragment in normalized_fragments:
                    if fragment in normalized:
                        leaks.append(
                            {
                                "split": split,
                                "index": index,
                                "field": field,
                                "kind": "fragment",
                                "value": fragment,
                            }
                        )
    return {"probe_exclusion_leaks": leaks}


def _invariant_name(invariant: FieldInvariant) -> str:
    """The label an invariant reports under.

    The factories below set ``__name__`` to their own name, or to the caller's
    ``label`` where one is given, so a finding says which invariant fired rather
    than printing a closure's repr.
    """
    name = getattr(invariant, "__name__", "")
    return str(name) if name else repr(invariant)


def no_digits() -> FieldInvariant:
    """The value must contain no decimal digit.

    The invariant behind "never invent an account number": a hand-authored
    behaviour curriculum teaches a *mapping*, and a stray digit in one of its
    frames is a fact the model has no way to know is fictional.
    """

    def invariant(value: str, row: Mapping[str, Any]) -> str | None:
        match = re.search(r"\d", value)
        return None if match is None else f"contains the digit {match.group(0)!r}"

    invariant.__name__ = "no_digits"
    return invariant


def no_questions() -> FieldInvariant:
    """The value must contain no ``?``.

    For a curriculum whose whole point is that the model *answers* rather than
    deflects, a question mark in the taught response is the deflection.
    """

    def invariant(value: str, row: Mapping[str, Any]) -> str | None:
        return "contains a question mark" if "?" in value else None

    invariant.__name__ = "no_questions"
    return invariant


def banned_patterns(patterns: Sequence[re.Pattern[str] | str], *, label: str) -> FieldInvariant:
    """No pattern in ``patterns`` may match the raw value.

    Unlike :func:`banned_wording_leaks`, which sweeps a whole corpus for one
    pattern, this is a per-row invariant meant to be composed with others in a
    single :func:`field_invariant_leaks` pass. ``label`` names it in the report,
    so several instances of this factory stay distinguishable.
    """
    if not label.strip():
        raise ValueError("banned_patterns: label must be a non-empty string")
    if not patterns:
        raise ValueError(f"banned_patterns({label!r}): pass at least one pattern")
    compiled = [
        re.compile(pattern) if isinstance(pattern, str) else pattern for pattern in patterns
    ]

    def invariant(value: str, row: Mapping[str, Any]) -> str | None:
        for pattern in compiled:
            match = pattern.search(value)
            if match is not None:
                return f"matches {pattern.pattern!r} at {match.group(0)!r}"
        return None

    invariant.__name__ = label
    return invariant


def forbidden_terms(terms: Sequence[str], *, label: str) -> FieldInvariant:
    """No term in ``terms`` may appear in the raw value, case-insensitively.

    Substring matching, deliberately: the terms this exists for are internal
    tool and system names that must never surface in text a customer reads, and
    they are just as wrong glued to a punctuation mark as standing alone.
    """
    if not label.strip():
        raise ValueError("forbidden_terms: label must be a non-empty string")
    if not terms:
        raise ValueError(f"forbidden_terms({label!r}): pass at least one term")
    lowered = [term.lower() for term in terms]

    def invariant(value: str, row: Mapping[str, Any]) -> str | None:
        haystack = value.lower()
        for term, original in zip(lowered, terms, strict=True):
            if term in haystack:
                return f"contains forbidden term {original!r}"
        return None

    invariant.__name__ = label
    return invariant


def required_markers(
    markers_by_tag: Mapping[str, Sequence[str]],
    *,
    tag_fn: Callable[[Mapping[str, Any]], str],
) -> FieldInvariant:
    """Every marker registered for the row's tag must appear in the value.

    ``tag_fn(row)`` picks the tag (an example kind, a behaviour key, a lane);
    a tag with no entry in ``markers_by_tag`` passes, so one invariant can be
    declared for the handful of tags that have a required phrasing without
    constraining the rest. Both the value and each marker are compared under
    :func:`dataforge.rows.normalize_text`, so a marker is not defeated by
    punctuation or casing.

    This is the positive counterpart to :func:`banned_patterns`: it is how a
    curriculum states that a particular behaviour must actually *say* the thing
    it exists to teach.

    An empty ``markers_by_tag``, and a tag whose markers all normalize away to
    nothing (``("?!",)``), both raise: either one is an invariant that can never
    fire, which here means every row silently satisfies a requirement nobody
    ever stated.
    """
    if not markers_by_tag:
        raise ValueError("required_markers: pass at least one tag with markers")
    normalized_markers: dict[str, list[str]] = {}
    for tag, markers in markers_by_tag.items():
        usable = [normalize_text(marker) for marker in markers if normalize_text(marker)]
        if not usable:
            raise ValueError(
                f"required_markers: tag {tag!r} has no marker left after normalization, so "
                "every row under it would pass unconditionally"
            )
        normalized_markers[tag] = usable

    def invariant(value: str, row: Mapping[str, Any]) -> str | None:
        tag = tag_fn(row)
        markers = normalized_markers.get(tag)
        if markers is None:
            return None
        normalized = normalize_text(value)
        for marker in markers:
            if marker not in normalized:
                return f"tag {tag!r} requires the marker {marker!r}"
        return None

    invariant.__name__ = "required_markers"
    return invariant


def min_word_count(n: int) -> FieldInvariant:
    """The value must carry at least ``n`` whitespace-separated words."""
    if n < 1:
        raise ValueError(f"min_word_count: n must be at least 1, got {n!r}")

    def invariant(value: str, row: Mapping[str, Any]) -> str | None:
        words = len(value.split())
        return None if words >= n else f"has {words} words, needs {n}"

    invariant.__name__ = "min_word_count"
    return invariant


def field_invariant_leaks(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    field: str,
    invariants: Sequence[FieldInvariant],
    row_predicate: Callable[[Mapping[str, Any]], bool] | None = None,
    splits_checked: Sequence[str] = ("train", "validation"),
) -> dict[str, Any]:
    """Run per-row invariants over one field and report every violation.

    These are meant to be wired through ``compose(pre_dedup_checks=...)``, not
    ``extra_leak_checks``: a violating row must fail the build *before* dedup
    can drop it. Dedup runs on the text field before the report is built, so a
    row that breaks an invariant and happens to duplicate a clean sibling would
    otherwise disappear -- and the build would pass on the strength of the row
    that survived rather than the row that was wrong.

    ``invariants`` is a sequence of callables ``(value, row) -> str | None``;
    the factories in this module (:func:`no_digits`, :func:`no_questions`,
    :func:`banned_patterns`, :func:`forbidden_terms`, :func:`required_markers`,
    :func:`min_word_count`) build the common ones, and any callable with a
    ``__name__`` works. Each is run against every checked row, so one pass
    reports every broken invariant instead of only the first.

    ``row_predicate`` scopes the check to the rows an invariant is *about* --
    a hand-authored behaviour curriculum's rows normally hold to rules the rest
    of the corpus does not. Raises ``ValueError`` when ``field`` is absent from
    every checked row, so a typo fails loudly instead of passing vacuously, and
    when ``invariants`` is empty, since a check that cannot fire is a disarmed
    gate that still reports a reassuring zero.

    The key is ``field_invariant_leaks`` (entries ``{"split", "index",
    "invariant", "detail"}``). There is deliberately no ``_leak_count``
    companion: the list alone gates, and a second key could only disagree with it.
    """
    if not invariants:
        raise ValueError(f"field_invariant_leaks({field!r}): pass at least one invariant")
    checked = [(split, rows) for split, rows in splits.items() if split in splits_checked]
    _require_field_present("field_invariant_leaks", checked, (field,))

    leaks: list[dict[str, Any]] = []
    for split, rows in checked:
        for index, row in enumerate(rows):
            if row_predicate is not None and not row_predicate(row):
                continue
            value = row.get(field)
            if not isinstance(value, str):
                continue
            for invariant in invariants:
                detail = invariant(value, row)
                if detail is not None:
                    leaks.append(
                        {
                            "split": split,
                            "index": index,
                            "invariant": _invariant_name(invariant),
                            "detail": detail,
                        }
                    )
    return {"field_invariant_leaks": leaks}


def unsupported_claim_leaks(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    field: str,
    claim_patterns: Sequence[re.Pattern[str] | str],
    evidence_fn: Callable[[Mapping[str, Any]], bool],
    row_predicate: Callable[[Mapping[str, Any]], bool] | None = None,
    splits_checked: Sequence[str] = ("train", "validation"),
) -> dict[str, Any]:
    """Flag rows that claim a completed action without evidence of one.

    A final response asserting "I have frozen the card" on a row that carries no
    tool call teaches the model that the sentence is what an account action
    looks like -- so it produces the sentence when it has done nothing, which is
    the most expensive failure a servicing model has. The rule "no evidence, no
    claim" is easy to state and impossible to hold to by review alone; this is
    what makes it enforceable at build time.

    ``claim_patterns`` are the regexes that recognize a completed-action claim
    in ``field``. ``evidence_fn(row)`` answers whether the row actually carries
    the evidence -- normally "does this row have tool-call turns", but the
    caller owns the definition, because what counts as evidence is the row
    schema's business. A row is flagged once per matching pattern when
    ``evidence_fn`` is ``False``; a claim backed by evidence is exactly what the
    dataset is for and is never flagged.

    The key is ``unsupported_claim_leaks`` (entries ``{"split", "index",
    "pattern"}``). There is deliberately no ``_leak_count`` companion: the list
    alone gates, and a second key could only disagree with it.
    """
    if not claim_patterns:
        raise ValueError(f"unsupported_claim_leaks({field!r}): pass at least one claim pattern")
    compiled = [
        re.compile(pattern) if isinstance(pattern, str) else pattern for pattern in claim_patterns
    ]
    leaks: list[dict[str, Any]] = []
    for split, rows in splits.items():
        if split not in splits_checked:
            continue
        for index, row in enumerate(rows):
            if row_predicate is not None and not row_predicate(row):
                continue
            value = row.get(field)
            if not isinstance(value, str):
                continue
            matched = [pattern for pattern in compiled if pattern.search(value) is not None]
            if not matched or evidence_fn(row):
                continue
            leaks.extend(
                {"split": split, "index": index, "pattern": pattern.pattern}
                for pattern in matched
            )
    return {"unsupported_claim_leaks": leaks}


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
