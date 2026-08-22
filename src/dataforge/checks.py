"""Teacher-batch checker: accumulate findings over a request/response batch.

A teacher round trip is only as trustworthy as the batch it produced. This
module reads the wire files :mod:`dataforge.teacher` writes and reads
(``export_teacher_requests`` -> ``<teacher>`` -> ``import_teacher_responses``)
plus the records they were built from, and reports **every** problem it finds
instead of stopping at the first one.

Two properties are load bearing, both ported from the v9 pipeline this
generalizes:

* **Accumulate, don't raise.** Content defects become :class:`Finding` records;
  only unreadable *input* raises (:class:`CheckerInputError`). A batch of 300
  rewrites is inspected once, not 300 times.
* **Deterministic output.** :func:`check_teacher_batch` returns findings sorted
  by ``(record_id, rule, detail)`` with exact duplicates removed, so two runs
  over the same batch produce byte-identical reports.

The rules themselves are generic: this module ships the mechanisms (hash
pinning, field presence, uniqueness against untouched records, an opening
n-gram cap per family, ...) and no domain vocabulary. Voice specifications,
banned-term lists and policy grounding belong to the caller, which composes
them with :func:`row_rule` or writes a plain ``Rule``.

Typical use::

    batch, findings = build_batch(requests_path, responses_path, records)
    findings = check_teacher_batch(requests_path, responses_path, records, rules)
    if findings:
        print(format_findings(findings))
    else:
        print(json.dumps(summarize(batch, findings), indent=2, sort_keys=True))
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataforge.rows import normalize_text_ascii

__all__ = [
    "Batch",
    "CheckerInputError",
    "Finding",
    "Pair",
    "Rule",
    "banned_pattern",
    "build_batch",
    "check_teacher_batch",
    "fields_present",
    "format_findings",
    "hash_pinned",
    "max_sentences",
    "min_words",
    "no_extra_keys",
    "opening_ngram_cap",
    "preserved_literals",
    "row_rule",
    "summarize",
    "unique_normalized",
    "untouched_field",
]

#: Record id attributed to a response row that carries none of its own.
UNKNOWN_RECORD_ID = "<unknown>"

#: Sentence terminator: punctuation followed by whitespace or end of string.
#: v9's ``SENTENCE_END_RE``; "3.5" and "e.g." inside a sentence do not count.
SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")

#: Lowercase alphabetic token, for opening n-grams. Digits are not tokens, so
#: "1792 is frozen" opens on ``("is", "frozen")``.
ALPHA_TOKEN_RE = re.compile(r"[a-z]+")


class CheckerInputError(ValueError):
    """Raised when the batch cannot be *read* -- a missing JSONL file or a row
    that is not a JSON object.

    This is the checker's only exception: every defect in the batch's
    *content* is reported as a :class:`Finding` instead. A malformed JSON line
    surfaces as the underlying :class:`json.JSONDecodeError` (also a
    ``ValueError``), which carries the line's own diagnostics.
    """


@dataclass(frozen=True)
class Finding:
    """One violation: which record, which rule, and what exactly is wrong."""

    record_id: str
    rule: str
    detail: str


@dataclass(frozen=True)
class Pair:
    """A response matched to both its request row and its source record."""

    record_id: str
    request: Mapping[str, Any]
    response: Mapping[str, Any]
    record: Mapping[str, Any]


@dataclass(frozen=True)
class Batch:
    """Everything a rule may look at.

    ``pairs`` holds only fully matched responses (request *and* record found);
    ``records`` maps every known record id to its record, so a rule can reach
    rows outside the batch; ``untouched`` lists, in input order, the records no
    response row claimed. Cross-row rules such as :func:`unique_normalized`
    need ``untouched`` to catch a rewrite that merely reproduces a row nobody
    asked the teacher to touch.
    """

    pairs: tuple[Pair, ...]
    records: Mapping[str, Mapping[str, Any]]
    untouched: tuple[Mapping[str, Any], ...]


#: A rule sees the whole batch and yields findings. Per-row rules are usually
#: written as a ``Pair -> details`` function and adapted with :func:`row_rule`.
Rule = Callable[[Batch], Iterable[Finding]]


def row_rule(name: str, fn: Callable[[Pair], Iterable[str]]) -> Rule:
    """Adapt a per-pair function into a :data:`Rule`.

    ``fn`` yields *details*; each becomes a :class:`Finding` tagged ``name``
    and attributed to the pair's record id. Every built-in factory except the
    two cross-row ones is written this way.
    """

    def rule(batch: Batch) -> list[Finding]:
        return [
            Finding(pair.record_id, name, detail) for pair in batch.pairs for detail in fn(pair)
        ]

    return rule


# ---------------------------------------------------------------------------
# reading the batch
# ---------------------------------------------------------------------------


def _as_paths(value: Path | str | Sequence[Path | str]) -> list[Path]:
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(item) for item in value]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CheckerInputError(f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CheckerInputError(f"{path}:{number} row must be a JSON object")
            rows.append(value)
    return rows


def _read_all(paths: Path | str | Sequence[Path | str]) -> list[dict[str, Any]]:
    return [row for path in _as_paths(paths) for row in _read_rows(path)]


def _row_id(row: Mapping[str, Any], record_id_field: str) -> str | None:
    value = row.get(record_id_field)
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def build_batch(
    requests: Path | str | Sequence[Path | str],
    responses: Path | str | Sequence[Path | str],
    records: Iterable[Mapping[str, Any]],
    *,
    record_id_field: str = "record_id",
) -> tuple[Batch, list[Finding]]:
    """Read the wire files and match them to ``records``.

    Returns the :class:`Batch` the rules will see and the ``input`` findings
    produced while matching. :func:`check_teacher_batch` is this plus the rule
    pass; call ``build_batch`` directly when you also want the batch (for
    :func:`summarize`, or to write a one-off cross-row check).

    Both path arguments accept a single path or a sequence of paths, which are
    concatenated in order -- a batch split across several teacher shards is one
    batch, so the same ``record_id`` in two response files is still a duplicate.

    ``input`` findings, none of which produce a pair:

    * ``row has no record_id`` (attributed to ``<unknown>``)
    * ``duplicate response row`` -- a second response for an id already seen
    * ``no matching request row`` -- nothing was exported for this id
    * ``no matching record`` -- the id is not among ``records``

    Request rows without a usable id are ignored: they can match no response,
    so whatever they were meant for is reported as ``no matching request row``
    instead. When two request rows share an id the last one wins, as it does in
    ``import_teacher_responses``; the same holds for two records sharing an id,
    so ``untouched`` carries at most one record per id, in first-seen order.
    Records without a usable id are ignored likewise -- unmatchable, and
    unreportable.
    """
    request_rows = _read_all(requests)
    response_rows = _read_all(responses)

    requests_by_id: dict[str, Mapping[str, Any]] = {}
    for row in request_rows:
        record_id = _row_id(row, record_id_field)
        if record_id is not None:
            requests_by_id[record_id] = row

    records_by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        record_id = _row_id(record, record_id_field)
        if record_id is not None:
            records_by_id[record_id] = record

    findings: list[Finding] = []
    pairs: list[Pair] = []
    seen: set[str] = set()
    for row in response_rows:
        record_id = _row_id(row, record_id_field)
        if record_id is None:
            findings.append(Finding(UNKNOWN_RECORD_ID, "input", "row has no record_id"))
            continue
        if record_id in seen:
            findings.append(Finding(record_id, "input", "duplicate response row"))
            continue
        seen.add(record_id)
        request_row = requests_by_id.get(record_id)
        source = records_by_id.get(record_id)
        if request_row is None:
            findings.append(Finding(record_id, "input", "no matching request row"))
        if source is None:
            findings.append(Finding(record_id, "input", "no matching record"))
        if request_row is not None and source is not None:
            pairs.append(Pair(record_id, request_row, row, source))

    untouched = tuple(
        record for record_id, record in records_by_id.items() if record_id not in seen
    )
    return Batch(pairs=tuple(pairs), records=records_by_id, untouched=untouched), findings


def check_teacher_batch(
    requests: Path | str | Sequence[Path | str],
    responses: Path | str | Sequence[Path | str],
    records: Iterable[Mapping[str, Any]],
    rules: Iterable[Rule],
    *,
    record_id_field: str = "record_id",
) -> list[Finding]:
    """Check a teacher batch and return every finding, sorted and deduplicated.

    See :func:`build_batch` for how rows are matched and which ``input``
    findings that produces. Rules then run over the resulting :class:`Batch`;
    their findings are pooled with the input ones, deduplicated exactly, and
    sorted by ``(record_id, rule, detail)``. An empty list means the batch is
    clean.

    Raises only :class:`CheckerInputError` (and :class:`json.JSONDecodeError`)
    for unreadable input. Exceptions raised *by a rule* are programming errors
    in the rule and propagate unchanged.
    """
    batch, findings = build_batch(requests, responses, records, record_id_field=record_id_field)
    collected = list(findings)
    for rule in rules:
        collected.extend(rule(batch))
    return sorted(
        set(collected), key=lambda finding: (finding.record_id, finding.rule, finding.detail)
    )


def format_findings(findings: Iterable[Finding]) -> str:
    """Render findings as ``record_id: rule: detail``, one per line."""
    return "\n".join(
        f"{finding.record_id}: {finding.rule}: {finding.detail}" for finding in findings
    )


def summarize(batch: Batch, findings: Sequence[Finding]) -> dict[str, Any]:
    """A small JSON-serializable audit object for a checked batch.

    ``checked`` counts pairs (responses that matched both a request and a
    record), ``untouched`` the records no response claimed, ``records`` every
    known record, ``violations`` the findings, and ``rules`` the sorted names
    of the rules that produced them -- so a clean run reports an empty list and
    a failing one names what failed without reprinting the detail lines.
    """
    return {
        "checked": len(batch.pairs),
        "untouched": len(batch.untouched),
        "records": len(batch.records),
        "violations": len(findings),
        "rules": sorted({finding.rule for finding in findings}),
    }


# ---------------------------------------------------------------------------
# helpers shared by the built-in rules
# ---------------------------------------------------------------------------


def _fields(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    fields = row.get("fields")
    return fields if isinstance(fields, Mapping) else None


def _text(row: Mapping[str, Any], field: str) -> str | None:
    """The submitted value of ``field``, or None when it is absent or blank.

    Content rules skip absent values rather than reporting them twice: use
    :func:`fields_present` to require a field, and the content rules to
    constrain it once it is there.
    """
    fields = _fields(row)
    if fields is None:
        return None
    value = fields.get(field)
    if not isinstance(value, str) or not value.strip():
        return None
    return value


# ---------------------------------------------------------------------------
# built-in rules: per row
# ---------------------------------------------------------------------------


def hash_pinned() -> Rule:
    """The response must echo the request's ``immutable_hash`` unchanged.

    This is the checker-side mirror of ``import_teacher_responses``' pre-hash
    check: catching a moved hash here reports every affected row at once,
    where the importer would raise on the first.
    """

    def check(pair: Pair) -> Iterable[str]:
        if pair.response.get("immutable_hash") != pair.request.get("immutable_hash"):
            yield "immutable_hash mismatch"

    return row_rule("hash_pinned", check)


def fields_present(fields: Sequence[str]) -> Rule:
    """Every named field must be present in ``response["fields"]`` as non-empty text."""

    def check(pair: Pair) -> Iterable[str]:
        submitted = _fields(pair.response)
        if submitted is None:
            yield "response has no fields mapping"
            return
        for field in fields:
            if field not in submitted:
                yield f"missing field {field!r}"
                continue
            value = submitted[field]
            if not isinstance(value, str) or not value.strip():
                yield f"field {field!r} must be non-empty text"

    return row_rule("fields_present", check)


def no_extra_keys(allowed: Collection[str]) -> Rule:
    """The response's top-level keys must be a subset of ``allowed``.

    A teacher that invents keys is a teacher that may have invented anything
    else; the importer ignores extra keys for control flow but hashes them into
    provenance, so they are worth reporting.
    """
    permitted = set(allowed)

    def check(pair: Pair) -> Iterable[str]:
        extra = set(pair.response) - permitted
        if extra:
            yield f"unexpected top-level keys: {sorted(extra)}"

    return row_rule("no_extra_keys", check)


def untouched_field(field: str) -> Rule:
    """``field`` must come back exactly as it was sent.

    The generic form of v9's ``--finals-only`` mode: when a run is only meant
    to rewrite the final response, the user turn is exported for context but
    must return byte-identical. A response that does not submit ``field`` at
    all edited nothing and is not flagged.
    """

    def check(pair: Pair) -> Iterable[str]:
        submitted = _fields(pair.response)
        if submitted is None or field not in submitted:
            return
        original = _fields(pair.request)
        before = None if original is None else original.get(field)
        if submitted[field] != before:
            yield f"field {field!r} must not be edited"

    return row_rule("untouched_field", check)


def min_words(field: str, n: int, normalize: Callable[[str], str] = normalize_text_ascii) -> Rule:
    """``field`` must carry at least ``n`` normalized words.

    Guards against a teacher that answers a whole scenario with "Done." --
    a rewrite that is syntactically valid and semantically empty.
    """

    def check(pair: Pair) -> Iterable[str]:
        value = _text(pair.response, field)
        if value is None:
            return
        words = len(normalize(value).split())
        if words < n:
            yield f"{field} has {words} normalized words, needs {n}"

    return row_rule("min_words", check)


def max_sentences(field: str, n: int) -> Rule:
    """``field`` may end at most ``n`` sentences (``[.!?]`` before space or end)."""

    def check(pair: Pair) -> Iterable[str]:
        value = _text(pair.response, field)
        if value is None:
            return
        sentences = len(SENTENCE_END_RE.findall(value))
        if sentences > n:
            yield f"{field} has {sentences} sentences, at most {n}"

    return row_rule("max_sentences", check)


def banned_pattern(field: str, pattern: re.Pattern[str] | str) -> Rule:
    """``field`` must not match ``pattern``; the detail quotes the first match.

    One finding per row, not per occurrence: the term that has to go is enough
    to act on, and a row full of the same word would otherwise drown the report.
    """
    compiled = re.compile(pattern) if isinstance(pattern, str) else pattern

    def check(pair: Pair) -> Iterable[str]:
        value = _text(pair.response, field)
        if value is None:
            return
        match = compiled.search(value)
        if match is not None:
            yield f"{field} contains banned wording {match.group(0)!r}"

    return row_rule("banned_pattern", check)


def preserved_literals(field: str, extract: Callable[[Pair], Iterable[str]]) -> Rule:
    """Every literal ``extract`` yields must survive into the rewritten ``field``.

    ``extract`` sees the whole :class:`Pair` -- request, response and record --
    so account digits, ISO dates, tool-envelope values, policy citations and
    grounding facts are all reachable, and what counts as load bearing stays
    the caller's decision. Comparison is case-insensitive; literals that differ
    only in case are checked once.
    """

    def check(pair: Pair) -> Iterable[str]:
        value = _text(pair.response, field)
        if value is None:
            return
        haystack = value.lower()
        required: dict[str, str] = {}
        for literal in extract(pair):
            if isinstance(literal, str) and literal.strip():
                required.setdefault(literal.lower(), literal)
        for lowered, literal in required.items():
            if lowered not in haystack:
                yield f"{field} dropped literal {literal!r}"

    return row_rule("preserved_literals", check)


# ---------------------------------------------------------------------------
# built-in rules: cross row
# ---------------------------------------------------------------------------


def unique_normalized(
    field: str,
    *,
    normalize: Callable[[str], str] = normalize_text_ascii,
    record_value: Callable[[Mapping[str, Any]], str | None],
    exempt: Callable[[Sequence[Mapping[str, Any]]], bool] | None = None,
    set_value: Callable[[Mapping[str, Any], str], Mapping[str, Any]] | None = None,
    split_of: Callable[[Mapping[str, Any]], Any] | None = None,
    split_field_name: str = "_split",
) -> Rule:
    """No rewrite may normalize to another rewrite, or to an untouched record.

    A teacher asked for 300 rewrites will happily emit the same sentence twice;
    worse, it may reproduce a released row it was never asked to touch, which
    is a duplicate the dataset build cannot see because that row's own text
    never changed. Both are reported here (v9 rules j and n):

    * ``duplicates the rewrite of {other}`` -- attributed to **both** rewrites,
      since neither is more at fault than the other;
    * ``duplicates untouched record {id}`` -- attributed to the rewriter only.

    Two untouched records that already share text are *not* flagged: they are
    the build's business, not the teacher's, and the dataset guards
    (:func:`dataforge.guards.duplicate_text_leaks`) own that check.

    Buckets are keyed on ``normalize(value)``; the rewritten value is read from
    ``response["fields"][field]`` and an untouched record's from
    ``record_value(record)``. Missing, non-string and blank values are skipped.

    ``exempt`` is offered every bucket of more than one member and suppresses it
    by returning True. **Its members are records, not response rows**: for a
    rewrite, ``set_value(pair.record, rewritten)`` -- by default a shallow copy
    of the record with ``field`` replaced -- and for an untouched row, a copy of
    the record itself. That is what lets
    :func:`dataforge.guards.paired_counterfactual_exemption` be reused verbatim
    here: it inspects ``pair_id``/``pair_target``/``history`` on records. Each
    member is additionally stamped with ``split_field_name`` (default
    ``"_split"``, which that predicate requires) from ``split_of(record)``, or
    from ``record["source_split"]`` when ``split_of`` is not given. The caller's
    records are never mutated. Members are ordered rewrites first (batch order)
    then untouched records (record order).

    Pass ``set_value`` whenever ``record_value`` is not a plain lookup of
    ``field`` -- it is ``record_value``'s inverse, and the exemption predicate
    only sees what it writes.
    """
    write_value = set_value if set_value is not None else _replace_field(field)

    def split_for(record: Mapping[str, Any]) -> Any:
        return split_of(record) if split_of is not None else record.get("source_split")

    def stamped(record: Mapping[str, Any]) -> dict[str, Any]:
        copy = dict(record)
        copy[split_field_name] = split_for(record)
        return copy

    def rule(batch: Batch) -> list[Finding]:
        rewrites: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for pair in batch.pairs:
            value = _text(pair.response, field)
            if value is None:
                continue
            member = stamped(write_value(pair.record, value))
            rewrites.setdefault(normalize(value), []).append((pair.record_id, member))

        # `Batch.records` is keyed by whatever `record_id_field` the batch was
        # built with, so recover an untouched record's id from there by
        # identity rather than assuming the key's name.
        ids_by_identity = {id(record): key for key, record in batch.records.items()}
        untouched: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for record in batch.untouched:
            value = record_value(record)
            if not isinstance(value, str) or not value.strip():
                continue
            record_id = ids_by_identity.get(id(record), "<unknown>")
            untouched.setdefault(normalize(value), []).append((record_id, stamped(record)))

        findings: list[Finding] = []
        for key, owners in rewrites.items():
            others = untouched.get(key, [])
            if len(owners) + len(others) < 2:
                continue
            if exempt is not None and exempt([member for _, member in (*owners, *others)]):
                continue
            for record_id, _ in owners:
                details = [
                    f"duplicates the rewrite of {other_id}"
                    for other_id, _ in owners
                    if other_id != record_id
                ]
                details += [f"duplicates untouched record {other_id}" for other_id, _ in others]
                findings.extend(
                    Finding(record_id, "unique_normalized", detail) for detail in details
                )
        return findings

    return rule


def _replace_field(field: str) -> Callable[[Mapping[str, Any], str], Mapping[str, Any]]:
    def write(record: Mapping[str, Any], value: str) -> Mapping[str, Any]:
        updated = dict(record)
        updated[field] = value
        return updated

    return write


def opening_ngram_cap(
    field: str,
    family_of: Callable[[Pair], str],
    *,
    n: int = 3,
    max_uses: int = 4,
) -> Rule:
    """Cap how often one opening n-gram may be reused within a family.

    A teacher converges: left alone it will open two thirds of a family with
    "I have frozen ...". The key is ``(family_of(pair), first n alphabetic
    tokens of the rewritten field, lowercased)``, and once a key is used more
    than ``max_uses`` times **every** pair using it is flagged -- there is no
    principled way to pick which of five identical openings is the offender,
    and the fix is to rewrite several of them.

    ``family_of`` is required because the cap is only meaningful within a set of
    comparable scenarios: four "I have frozen" openings across four different
    families is variety, within one family it is a tic.

    Counts cover **rewritten rows only**, as v9 does: an untouched record that
    already opens the same way is released text and cannot be rewritten by this
    batch, so charging it against the cap would flag rows nobody can fix.
    """

    def rule(batch: Batch) -> list[Finding]:
        keyed: dict[tuple[str, tuple[str, ...]], list[str]] = {}
        for pair in batch.pairs:
            value = _text(pair.response, field)
            if value is None:
                continue
            gram = tuple(ALPHA_TOKEN_RE.findall(value.lower())[:n])
            if not gram:
                continue
            keyed.setdefault((family_of(pair), gram), []).append(pair.record_id)

        findings: list[Finding] = []
        for (family, gram), record_ids in keyed.items():
            uses = len(record_ids)
            if uses <= max_uses:
                continue
            detail = f"opening {n}-gram {' '.join(gram)!r} used {uses} times in {family}"
            findings.extend(
                Finding(record_id, "opening_ngram_cap", detail) for record_id in record_ids
            )
        return findings

    return rule
