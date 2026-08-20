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
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

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
    :func:`_resolve_wiring`.
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
    """
    excluded_fields, validate = _resolve_wiring(editable_fields, derived_fields, rederive, validate)
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
