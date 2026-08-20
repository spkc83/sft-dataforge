"""Teacher realization: let an LLM teacher edit wording, never semantics.

The deterministic skeleton (taxonomy + curricula) owns every label and
tool-call decision. A teacher model may only rewrite a declared set of
free-text fields (e.g. the user turn, the final response). This module
enforces that boundary with a hash over everything *except* those fields:
if the hash changes after a teacher response is applied, the response is
rejected.

That boundary has a gap if one of the editable fields is itself a
*dependency* of a derived field the hash also treats as immutable (e.g.
``dataforge.rows.make_row``'s rendered ``text`` is derived from
``current_text``): editing the dependency without recomputing the derivative
leaves the derivative stale, and the hash -- which only ever compares
top-level dict keys -- cannot see that the two have drifted apart. This
module closes that gap two ways: by default it refuses an ``editable_fields``
set that affects a known derived field unless the caller also supplies a
``rederive`` callback (see :data:`dataforge.rows.DERIVED_FIELDS`), and it
accepts an optional ``validate`` structural hook -- analogous to hello-SLM's
``validate_records`` -- run after every edit is applied.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from dataforge.rows import DERIVED_FIELDS, canonical_json_bytes


class TeacherRealizationError(ValueError):
    """Raised when a teacher response is missing, malformed, tampers with a
    field outside its ``allowed_edits``, or leaves a derived field stale."""


def immutable_hash(record: Mapping[str, Any], editable_fields: Sequence[str]) -> str:
    """Hash a record's non-editable projection.

    Two records with the same immutable hash are guaranteed to differ, if
    at all, only within ``editable_fields``.
    """
    projection = {key: value for key, value in record.items() if key not in editable_fields}
    return f"sha256:{hashlib.sha256(canonical_json_bytes(projection)).hexdigest()}"


def _affected_derived_fields(
    editable_fields: Sequence[str],
    derived_fields: Mapping[str, Sequence[str]],
) -> set[str]:
    editable = set(editable_fields)
    return {derived for derived, deps in derived_fields.items() if editable & set(deps)}


def _check_derivation_wiring(
    editable_fields: Sequence[str],
    derived_fields: Mapping[str, Sequence[str]],
    rederive: Callable[[dict[str, Any]], None] | None,
) -> None:
    affected = _affected_derived_fields(editable_fields, derived_fields)
    if affected and rederive is None:
        raise TeacherRealizationError(
            f"editable_fields {sorted(set(editable_fields))} affect derived field(s) "
            f"{sorted(affected)}; pass a `rederive` callback that recomputes them (e.g. "
            "dataforge.rows.rederive_text), or drop those fields from editable_fields, or "
            "pass derived_fields={} to opt out of this check"
        )


def export_teacher_requests(
    records: Iterable[Mapping[str, Any]],
    path: Path,
    *,
    editable_fields: Sequence[str],
    record_id_field: str = "record_id",
    instructions: str = "Rewrite only the listed fields for fluency; do not change their meaning.",
    derived_fields: Mapping[str, Sequence[str]] = DERIVED_FIELDS,
    rederive: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Write one teacher request per record: its immutable hash and current
    values of the editable fields, to be rewritten and returned unchanged
    in shape by :func:`import_teacher_responses`.

    ``derived_fields``/``rederive`` must match whatever will be passed to
    the later :func:`import_teacher_responses` call, so the two agree on
    what is (and isn't) covered by ``immutable_hash``; see
    :func:`_check_derivation_wiring`.
    """
    _check_derivation_wiring(editable_fields, derived_fields, rederive)
    excluded_fields = tuple(editable_fields) + (
        tuple(derived_fields) if rederive is not None else ()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            record_id = record[record_id_field]
            row = {
                "record_id": record_id,
                "immutable_hash": immutable_hash(record, excluded_fields),
                "fields": {field: record.get(field) for field in editable_fields},
                "allowed_edits": list(editable_fields),
                "instructions": instructions,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def import_teacher_responses(
    records: Iterable[Mapping[str, Any]],
    path: Path,
    *,
    editable_fields: Sequence[str],
    teacher_model: str,
    teacher_prompt_hash: str,
    record_id_field: str = "record_id",
    provenance_field: str = "provenance",
    derived_fields: Mapping[str, Sequence[str]] = DERIVED_FIELDS,
    rederive: Callable[[dict[str, Any]], None] | None = None,
    validate: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Apply teacher responses, verifying the immutable hash before and after.

    Records without a matching response row pass through unchanged. Every
    applied response must reproduce the exact ``immutable_hash`` recorded at
    export time and must not change the record's immutable hash after the
    edit (and, if given, ``rederive``/``validate``) run -- either failure
    raises :class:`TeacherRealizationError`.

    If any ``editable_fields`` entry is a dependency of a field in
    ``derived_fields`` (default :data:`dataforge.rows.DERIVED_FIELDS`), a
    ``rederive`` callback is required: it runs, mutating the record in
    place, immediately after the editable fields are set, and the derived
    fields it owns are excluded from the immutable-hash comparison (they are
    *expected* to change as a function of the edit). ``validate`` is an
    optional additional structural check -- run after ``rederive`` -- for
    invariants no single hash can express (e.g. "the rendered text still
    matches current_text"); it receives the fully-edited record and should
    raise on any inconsistency, analogous to hello-SLM's ``validate_records``.
    """
    _check_derivation_wiring(editable_fields, derived_fields, rederive)
    excluded_fields = tuple(editable_fields) + (
        tuple(derived_fields) if rederive is not None else ()
    )
    responses = {row["record_id"]: row for row in _read_jsonl(path)}
    realized = [dict(record) for record in records]
    for record in realized:
        record_id = record[record_id_field]
        row = responses.get(record_id)
        if row is None:
            continue
        before_hash = immutable_hash(record, excluded_fields)
        if row.get("immutable_hash") != before_hash:
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
        if immutable_hash(record, excluded_fields) != before_hash:
            raise TeacherRealizationError(
                f"{record_id} teacher response changed immutable semantics"
            )
        provenance = dict(record.get(provenance_field, {}))
        provenance["teacher_model"] = teacher_model
        provenance["teacher_prompt_hash"] = teacher_prompt_hash
        provenance["teacher_realization_hash"] = immutable_hash(row, [])
        record[provenance_field] = provenance
    return realized


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
