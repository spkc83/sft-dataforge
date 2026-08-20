"""Teacher realization: let an LLM teacher edit wording, never semantics.

The deterministic skeleton (taxonomy + curricula) owns every label and
tool-call decision. A teacher model may only rewrite a declared set of
free-text fields (e.g. the user turn, the final response). This module
enforces that boundary with a hash over everything *except* those fields:
if the hash changes after a teacher response is applied, the response is
rejected.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from dataforge.emit import canonical_json_bytes


class TeacherRealizationError(ValueError):
    """Raised when a teacher response is missing, malformed, or tampers with
    a field outside its ``allowed_edits``."""


def immutable_hash(record: Mapping[str, Any], editable_fields: Sequence[str]) -> str:
    """Hash a record's non-editable projection.

    Two records with the same immutable hash are guaranteed to differ, if
    at all, only within ``editable_fields``.
    """
    import hashlib

    projection = {key: value for key, value in record.items() if key not in editable_fields}
    return f"sha256:{hashlib.sha256(canonical_json_bytes(projection)).hexdigest()}"


def export_teacher_requests(
    records: Iterable[Mapping[str, Any]],
    path: Path,
    *,
    editable_fields: Sequence[str],
    record_id_field: str = "record_id",
    instructions: str = "Rewrite only the listed fields for fluency; do not change their meaning.",
) -> None:
    """Write one teacher request per record: its immutable hash and current
    values of the editable fields, to be rewritten and returned unchanged
    in shape by :func:`import_teacher_responses`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            record_id = record[record_id_field]
            row = {
                "record_id": record_id,
                "immutable_hash": immutable_hash(record, editable_fields),
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
) -> list[dict[str, Any]]:
    """Apply teacher responses, verifying the immutable hash before and after.

    Records without a matching response row pass through unchanged. Every
    applied response must reproduce the exact ``immutable_hash`` recorded at
    export time and must not change the record's immutable hash after the
    edit is applied -- either failure raises :class:`TeacherRealizationError`.
    """
    responses = {row["record_id"]: row for row in _read_jsonl(path)}
    realized = [dict(record) for record in records]
    for record in realized:
        record_id = record[record_id_field]
        row = responses.get(record_id)
        if row is None:
            continue
        before_hash = immutable_hash(record, editable_fields)
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
        if immutable_hash(record, editable_fields) != before_hash:
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
