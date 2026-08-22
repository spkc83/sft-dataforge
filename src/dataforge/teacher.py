"""Teacher realization: let an LLM teacher edit wording, never semantics.

The deterministic skeleton (taxonomy + curricula) owns every label and
tool-call decision. A teacher model may only rewrite a declared set of
free-text fields (e.g. the user turn, the final response). This module
enforces that boundary with a hash over everything *except* those fields:
if the hash changes after a teacher response is applied, the response is
rejected.

That boundary has a gap if an editable field is, or is a *dependency* of, a
derived field the hash would otherwise treat as immutable (e.g.
``dataforge.rows.make_row``'s rendered ``text`` is derived from
``current_text``): editing the dependency -- or the derived field itself --
without recomputing the derivative correctly leaves it stale, and the hash,
which only ever compares top-level dict keys, cannot see that the two have
drifted apart. Closing that gap needs two things working together, and this
module requires both whenever a derived field is affected:

* a ``rederive`` callback (see :data:`dataforge.rows.rederive_text`) that
  recomputes the derived field from the (possibly just-edited) fields it
  depends on; and
* a ``validate`` structural check (see
  :data:`dataforge.rows.validate_row_consistency`) -- analogous to
  hello-SLM's ``validate_records`` -- run after ``rederive``, so a
  ``rederive`` that is present but wrong (a no-op stub, a bug, or just the
  wrong function) is caught rather than silently trusted. Only the fields
  ``rederive`` is actually responsible for (the *affected* derived fields)
  are excluded from the immutable-hash comparison -- passing a `rederive`
  defensively, for editable fields that don't affect anything, never widens
  what the hash ignores.

``validate`` defaults to :func:`dataforge.rows.validate_row_consistency`
whenever it's needed and the default :data:`dataforge.rows.DERIVED_FIELDS`
map is in use; callers using a custom ``derived_fields`` map must supply
their own ``validate`` explicitly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Collection, Iterable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from dataforge.emit import file_sha256
from dataforge.rows import DERIVED_FIELDS, canonical_json_bytes, validate_row_consistency

#: Sentinel default for `validate`: resolves to `validate_row_consistency`
#: when it's required and the default DERIVED_FIELDS map is in use, else to
#: None. Passing `validate=None` explicitly is a deliberate opt-out attempt
#: and is refused (not silently honored) whenever a derived field is
#: affected -- see `_resolve_wiring`.
_AUTO_VALIDATE = object()


class TeacherRealizationError(ValueError):
    """Raised when a teacher response is missing, malformed, tampers with a
    field outside its ``allowed_edits``, or leaves a derived field stale."""


def immutable_hash(
    record: Mapping[str, Any],
    editable_fields: Sequence[str],
    *,
    provenance_field: str = "provenance",
) -> str:
    """Hash a record's non-editable projection.

    Two records with the same immutable hash are guaranteed to differ, if
    at all, only within ``editable_fields`` and ``provenance_field``.

    ``provenance_field`` is excluded because provenance is *written onto* a
    record by the very pipeline the hash protects: :func:`import_teacher_responses`
    stamps ``teacher_model``/``teacher_prompt_hash``/``teacher_realization_hash``,
    and :func:`scrub_fields` stamps ``pre_scrub_immutable_hashes``. If those
    stamps were hashed, a record's own hash would move the moment it was
    realized, so exporting a second round of teacher requests from realized
    records would invalidate every request already in flight. Keeping
    provenance outside the projection is what makes export -> import -> export
    idempotent -- and it is also what makes the pre-scrub stamps *claims*
    rather than proofs (see :func:`import_teacher_responses`).
    """
    excluded = set(editable_fields)
    excluded.add(provenance_field)
    projection = {key: value for key, value in record.items() if key not in excluded}
    return f"sha256:{hashlib.sha256(canonical_json_bytes(projection)).hexdigest()}"


def _affected_derived_fields(
    editable_fields: Sequence[str],
    derived_fields: Mapping[str, Sequence[str]],
) -> set[str]:
    """Derived fields an edit to ``editable_fields`` could invalidate.

    A derived field is affected if it is itself editable (a caller could
    hand-edit it directly, out of sync with its inputs) or if any of its
    declared dependencies is editable.
    """
    editable = set(editable_fields)
    return {
        derived
        for derived, deps in derived_fields.items()
        if derived in editable or editable & set(deps)
    }


def _resolve_wiring(
    editable_fields: Sequence[str],
    derived_fields: Mapping[str, Sequence[str]],
    rederive: Callable[[dict[str, Any]], None] | None,
    validate: Callable[[Mapping[str, Any]], None] | None,
) -> tuple[tuple[str, ...], Callable[[Mapping[str, Any]], None] | None]:
    """Validate rederive/validate wiring; return (excluded_fields, validate).

    ``excluded_fields`` -- the fields the immutable hash ignores -- is
    always ``editable_fields`` plus only the *affected* derived fields
    (never the full ``derived_fields`` map), so an unrelated, defensively
    passed ``rederive`` never widens the hash's blind spot.
    """
    affected = _affected_derived_fields(editable_fields, derived_fields)
    if validate is _AUTO_VALIDATE:
        validate = (
            validate_row_consistency if affected and derived_fields is DERIVED_FIELDS else None
        )
    if affected:
        if rederive is None:
            raise TeacherRealizationError(
                f"editable_fields {sorted(set(editable_fields))} affect derived field(s) "
                f"{sorted(affected)}; pass a `rederive` callback that recomputes them (e.g. "
                "dataforge.rows.rederive_text), or drop those fields from editable_fields, or "
                "pass derived_fields={} to opt out of this check"
            )
        if validate is None:
            raise TeacherRealizationError(
                f"editable_fields {sorted(set(editable_fields))} affect derived field(s) "
                f"{sorted(affected)}; a `rederive` callback alone is not verified to have "
                "worked -- also pass a `validate` callback (e.g. "
                "dataforge.rows.validate_row_consistency) to structurally confirm the derived "
                "field(s) are consistent after the edit, or pass derived_fields={} to opt out "
                "of this check"
            )
    excluded_fields = tuple(editable_fields) + tuple(sorted(affected))
    return excluded_fields, validate


def export_teacher_requests(
    records: Iterable[Mapping[str, Any]],
    path: Path,
    *,
    editable_fields: Sequence[str],
    record_id_field: str = "record_id",
    provenance_field: str = "provenance",
    instructions: str = "Rewrite only the listed fields for fluency; do not change their meaning.",
    derived_fields: Mapping[str, Sequence[str]] = DERIVED_FIELDS,
    rederive: Callable[[dict[str, Any]], None] | None = None,
    validate: Callable[[Mapping[str, Any]], None] | None = _AUTO_VALIDATE,  # type: ignore[assignment]
) -> None:
    """Write one teacher request per record: its immutable hash and current
    values of the editable fields, to be rewritten and returned unchanged
    in shape by :func:`import_teacher_responses`.

    ``derived_fields``/``rederive``/``validate`` must match whatever will be
    passed to the later :func:`import_teacher_responses` call, so the two
    agree on what is (and isn't) covered by ``immutable_hash``; see
    :func:`_resolve_wiring`. ``provenance_field`` must match too: it names the
    key :func:`immutable_hash` leaves out of the projection.
    """
    excluded_fields, _validate = _resolve_wiring(
        editable_fields, derived_fields, rederive, validate
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            record_id = record[record_id_field]
            row = {
                "record_id": record_id,
                "immutable_hash": immutable_hash(
                    record, excluded_fields, provenance_field=provenance_field
                ),
                "fields": {field: record.get(field) for field in editable_fields},
                "allowed_edits": list(editable_fields),
                "instructions": instructions,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _apply_substitutions(text: str, substitutions: Sequence[tuple[str, str]]) -> str:
    for old, new in substitutions:
        text = text.replace(old, new)
    return text


def scrub_fields(
    record: MutableMapping[str, Any],
    substitutions: Sequence[tuple[str, str]],
    *,
    fields: Sequence[str],
    editable_fields: Sequence[str],
    message_roles: Collection[str] = ("user", "assistant"),
    provenance_field: str = "provenance",
    derived_fields: Mapping[str, Sequence[str]] = DERIVED_FIELDS,
    rederive: Callable[[dict[str, Any]], None] | None = None,
    validate: Callable[[Mapping[str, Any]], None] | None = _AUTO_VALIDATE,  # type: ignore[assignment]
) -> bool:
    """Rewrite literal substrings in non-editable ``fields``, in place.

    A scrub is the pipeline's own edit of text the *teacher* may not touch:
    house-style fixes, product-surface names, wording that betrays a synthetic
    corpus. Each ``(old, new)`` pair is applied with plain :meth:`str.replace`
    in order. ``fields`` may name plain string fields or message-list fields
    (a list of ``{"role", "content"}`` dicts, of which only the ``content`` of
    messages whose ``role`` is in ``message_roles`` is scrubbed -- system and
    tool turns are left alone). Returns whether any string actually changed.

    Refuses (:class:`TeacherRealizationError`) any field that is also in
    ``editable_fields``: editable text belongs to the teacher, and scrubbing it
    behind the teacher's back both races the realization and defeats the point
    of the hash. Constrain the teacher's own output instead (e.g. a
    banned-pattern check over the returned fields).

    Scrubbing a field that *feeds* a derived field requires the same
    ``rederive`` + ``validate`` wiring the teacher path requires, for the same
    reason: otherwise the derived field silently goes stale. When the scrub
    changed something and a derived field is affected, ``rederive`` and then
    ``validate`` run against the scrubbed record.

    **Stamping.** A scrub can touch fields the immutable hash covers, which
    would invalidate teacher request files already exported against the old
    hash. When the hash moves, an entry
    ``{"before_hash", "after_hash", "excluded_fields"}`` is appended to
    ``record[provenance_field]["pre_scrub_immutable_hashes"]`` (created if
    absent; never rewritten or truncated, so a chain of scrubs keeps every
    hash the record ever had). Both hashes and the ``excluded_fields`` tag are
    computed from the **teacher's** projection -- ``editable_fields`` plus the
    derived fields *they* affect -- i.e. the exact exclusion set
    :func:`import_teacher_responses` recomputes, not the projection implied by
    the scrubbed ``fields``. ``after_hash`` is informational; only
    ``before_hash`` and the tag are matched on import, and only when the
    importer is asked to trust the stamps (``accept_pre_scrub_hashes=True``).

    Because the tag is the *resolved* exclusion set -- ``editable_fields`` plus
    the derived fields they affect -- ``derived_fields`` must be pinned
    identically here and at import, not just ``editable_fields``. A scrub
    stamped under the default :data:`dataforge.rows.DERIVED_FIELDS` and an
    import passing ``derived_fields={}`` resolve to different tags, so the
    stamp will not match and the pre-scrub request will be rejected.
    """
    overlap = set(fields) & set(editable_fields)
    if overlap:
        raise TeacherRealizationError(
            f"cannot scrub editable field(s) {sorted(overlap)}: editable text belongs to the "
            "teacher, so scrubbing it here would race the realization and be overwritten by it; "
            "constrain the teacher's own output instead (e.g. a banned-pattern check on the "
            "returned fields), or drop those fields from editable_fields"
        )
    # Two wiring calls, two purposes. This one only enforces that scrubbing a
    # field which feeds a derived field brings rederive + validate along.
    _scrub_excluded, scrub_validate = _resolve_wiring(fields, derived_fields, rederive, validate)
    scrub_affects_derived = bool(_affected_derived_fields(fields, derived_fields))
    # This one reproduces the teacher's projection, so the stamped hashes and
    # the tag match what import_teacher_responses will compute.
    teacher_excluded, _teacher_validate = _resolve_wiring(
        editable_fields, derived_fields, rederive, validate
    )

    before_hash = immutable_hash(record, teacher_excluded, provenance_field=provenance_field)
    changed = False
    for field in fields:
        value = record.get(field)
        if isinstance(value, str):
            scrubbed = _apply_substitutions(value, substitutions)
            if scrubbed != value:
                record[field] = scrubbed
                changed = True
        elif isinstance(value, list):
            for message in value:
                if not isinstance(message, MutableMapping):
                    continue
                if message.get("role") not in message_roles:
                    continue
                content = message.get("content")
                if not isinstance(content, str):
                    continue
                scrubbed = _apply_substitutions(content, substitutions)
                if scrubbed != content:
                    message["content"] = scrubbed
                    changed = True

    if changed and scrub_affects_derived:
        if rederive is not None:
            rederive(record)  # type: ignore[arg-type]
        if scrub_validate is not None:
            scrub_validate(record)

    after_hash = immutable_hash(record, teacher_excluded, provenance_field=provenance_field)
    if after_hash != before_hash:
        provenance = record.get(provenance_field)
        if provenance is None:
            provenance = {}
            record[provenance_field] = provenance
        elif not isinstance(provenance, MutableMapping):
            raise TeacherRealizationError(
                f"cannot stamp a pre-scrub hash: {provenance_field!r} is "
                f"{type(provenance).__name__}, not a mutable mapping"
            )
        stamps = provenance.get("pre_scrub_immutable_hashes")
        if stamps is None:
            stamps = []
            provenance["pre_scrub_immutable_hashes"] = stamps
        elif not isinstance(stamps, list):
            raise TeacherRealizationError(
                "cannot stamp a pre-scrub hash: "
                f"{provenance_field}['pre_scrub_immutable_hashes'] is "
                f"{type(stamps).__name__}, not a list"
            )
        stamps.append(
            {
                "before_hash": before_hash,
                "after_hash": after_hash,
                "excluded_fields": sorted(teacher_excluded),
            }
        )
    return changed


def _stamped_pre_scrub_hashes(
    record: Mapping[str, Any],
    excluded_fields: Sequence[str],
    provenance_field: str,
) -> set[str]:
    """Pre-scrub hashes stamped on ``record`` for *this* hash projection.

    Entries tagged with a different ``excluded_fields`` set describe a
    different hash and are ignored -- accepting them would let a stamp written
    for a narrow projection unlock a request pinned to a wide one.
    """
    provenance = record.get(provenance_field)
    if not isinstance(provenance, Mapping):
        return set()
    stamps = provenance.get("pre_scrub_immutable_hashes")
    if not isinstance(stamps, list):
        return set()
    tag = sorted(excluded_fields)
    return {
        entry["before_hash"]
        for entry in stamps
        if isinstance(entry, Mapping)
        and entry.get("excluded_fields") == tag
        and isinstance(entry.get("before_hash"), str)
    }


def import_teacher_responses(
    records: Iterable[Mapping[str, Any]],
    path: Path,
    *,
    editable_fields: Sequence[str],
    teacher_model: str,
    teacher_prompt_hash: str,
    record_id_field: str = "record_id",
    provenance_field: str = "provenance",
    accept_pre_scrub_hashes: bool = False,
    derived_fields: Mapping[str, Sequence[str]] = DERIVED_FIELDS,
    rederive: Callable[[dict[str, Any]], None] | None = None,
    validate: Callable[[Mapping[str, Any]], None] | None = _AUTO_VALIDATE,  # type: ignore[assignment]
) -> list[dict[str, Any]]:
    """Apply teacher responses, verifying the immutable hash before and after.

    Records without a matching response row pass through unchanged. Every
    applied response must reproduce the exact ``immutable_hash`` recorded at
    export time and must not change the record's immutable hash after the
    edit (and, if given, ``rederive``/``validate``) run -- either failure
    raises :class:`TeacherRealizationError`.

    If any ``editable_fields`` entry is, or is a dependency of, a field in
    ``derived_fields`` (default :data:`dataforge.rows.DERIVED_FIELDS`), both
    a ``rederive`` callback AND a ``validate`` callback are required (see
    the module docstring): ``rederive`` runs first, mutating the record in
    place immediately after the editable fields are set; ``validate`` then
    runs against the fully-edited record and must raise on any
    inconsistency. Only the affected derived fields are excluded from the
    immutable-hash comparison. Pass ``derived_fields={}`` to opt out of this
    guard entirely (the caller then owns the consistency of any derived
    field on their own).

    ``accept_pre_scrub_hashes`` widens **only** the request-side check: a
    response row may then pin either the record's live hash or the
    ``before_hash`` of any :func:`scrub_fields` stamp on the record whose
    ``excluded_fields`` tag matches this call's projection. This is how a
    teacher file exported before a scrub stays usable after it. The widening
    is strictly one-directional: the post-edit check still compares against
    the record's *live* hash, so the record's own semantics still cannot move.
    There is deliberately no chain or after-hash condition -- a later benign
    mutation (stamping an id, a second scrub) must not lock the old teacher
    file out, which is the whole point of the mechanism. The tag compared is
    the *resolved* exclusion set (``editable_fields`` plus the derived fields
    they affect), so ``derived_fields`` -- not only ``editable_fields`` -- must
    be pinned identically to whatever :func:`scrub_fields` was given: a scrub
    stamped under the default :data:`dataforge.rows.DERIVED_FIELDS` and an
    import passing ``derived_fields={}`` resolve to different tags and the
    stamp will not match.

    **Trust boundary.** ``provenance`` is outside the hash (that is what makes
    it writable at all) and :func:`scrub_fields` is public, so a stamp
    certifies nothing by itself: anything that can write the record's
    provenance can mint a stamp for any hash it likes, and this importer will
    then accept a teacher file pinned to that hash. Passing
    ``accept_pre_scrub_hashes=True`` therefore does not mean "the library
    verified a scrub happened"; it means "I trust every step that could write
    provenance between export and import". Keep the default ``False``
    (identical to the pre-flag behaviour: exactly one acceptable request hash,
    recomputed live) unless the pipeline in between is yours.
    """
    excluded_fields, validate = _resolve_wiring(editable_fields, derived_fields, rederive, validate)
    responses = {row["record_id"]: row for row in _read_jsonl(path)}
    realized = [dict(record) for record in records]
    for record in realized:
        record_id = record[record_id_field]
        row = responses.get(record_id)
        if row is None:
            continue
        before_hash = immutable_hash(record, excluded_fields, provenance_field=provenance_field)
        accepted_hashes = {before_hash}
        if accept_pre_scrub_hashes:
            accepted_hashes |= _stamped_pre_scrub_hashes(record, excluded_fields, provenance_field)
        if row.get("immutable_hash") not in accepted_hashes:
            raise TeacherRealizationError(f"{record_id} teacher request hash mismatch")
        fields = row.get("fields")
        if not isinstance(fields, Mapping):
            raise TeacherRealizationError(f"{record_id} teacher response is missing fields")
        for field in editable_fields:
            if field not in fields:
                raise TeacherRealizationError(
                    f"{record_id} teacher response missing field {field!r}"
                )
            value = fields[field]
            if not isinstance(value, str) or not value.strip():
                raise TeacherRealizationError(f"{record_id} field {field!r} must be non-empty text")
            record[field] = value.strip()
        if rederive is not None:
            rederive(record)
        if validate is not None:
            try:
                validate(record)
            except Exception as error:
                raise TeacherRealizationError(
                    f"{record_id} failed post-application validation: {error}"
                ) from error
        # Post-edit: the *live* hash, never a stamped one (see the docstring's
        # one-directional widening).
        after_hash = immutable_hash(record, excluded_fields, provenance_field=provenance_field)
        if after_hash != before_hash:
            raise TeacherRealizationError(
                f"{record_id} teacher response changed immutable semantics"
            )
        provenance = dict(record.get(provenance_field, {}))
        provenance["teacher_model"] = teacher_model
        provenance["teacher_prompt_hash"] = teacher_prompt_hash
        # The realization hash digests the response row *whole* -- no field is
        # excluded, not even `provenance_field`. This is a record of what the
        # teacher actually returned, not a projection of a record: anything the
        # response carried (extra keys the importer ignores for control flow
        # included) has to be inside the digest for it to be worth keeping.
        provenance["teacher_realization_hash"] = (
            f"sha256:{hashlib.sha256(canonical_json_bytes(row)).hexdigest()}"
        )
        record[provenance_field] = provenance
    return realized


def compute_teacher_prompt_hash(prompt_path: Path, request_path: Path) -> str:
    """Digest the teacher prompt spec plus the request file sent with it.

    ``teacher_prompt_hash`` is an opaque provenance string as far as
    :func:`import_teacher_responses` is concerned -- the library never
    recomputes or verifies it. This is an explicit, reproducible definition of
    it for callers who want one: ``"sha256:"`` + sha256 over the canonical JSON
    of ``{"prompt_sha256": ..., "requests_sha256": ...}``, so it moves when
    either the prompt spec or the exact set of requests moves, and neither file
    has to be kept around to re-verify the pairing later.
    """
    payload = {
        "prompt_sha256": file_sha256(prompt_path),
        "requests_sha256": file_sha256(request_path),
    }
    return f"sha256:{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
