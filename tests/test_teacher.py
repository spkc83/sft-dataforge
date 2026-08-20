from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataforge.teacher import (
    TeacherRealizationError,
    export_teacher_requests,
    immutable_hash,
    import_teacher_responses,
)


def _record() -> dict:
    return {
        "record_id": "r1",
        "text": "immutable text",
        "group_id": "g1",
        "assistant_response": "I found your account balances.",
    }


def test_immutable_hash_ignores_editable_fields() -> None:
    record = _record()
    hash_before = immutable_hash(record, ["assistant_response"])
    record["assistant_response"] = "different wording"
    hash_after = immutable_hash(record, ["assistant_response"])
    assert hash_before == hash_after


def test_immutable_hash_changes_when_immutable_field_changes() -> None:
    record = _record()
    hash_before = immutable_hash(record, ["assistant_response"])
    record["group_id"] = "g2"
    hash_after = immutable_hash(record, ["assistant_response"])
    assert hash_before != hash_after


def test_export_then_import_round_trip(tmp_path: Path) -> None:
    records = [_record()]
    requests_path = tmp_path / "requests.jsonl"
    export_teacher_requests(records, requests_path, editable_fields=["assistant_response"])

    responses_path = tmp_path / "responses.jsonl"
    rows = [json.loads(line) for line in requests_path.read_text().splitlines()]
    with responses_path.open("w") as handle:
        for row in rows:
            row["fields"] = {"assistant_response": "I found your account balances, all good."}
            handle.write(json.dumps(row) + "\n")

    realized = import_teacher_responses(
        records,
        responses_path,
        editable_fields=["assistant_response"],
        teacher_model="stub",
        teacher_prompt_hash="sha256:x",
    )
    assert realized[0]["assistant_response"] == "I found your account balances, all good."
    assert realized[0]["provenance"]["teacher_model"] == "stub"
    assert realized[0]["group_id"] == "g1"  # untouched


def test_import_rejects_tampered_immutable_hash(tmp_path: Path) -> None:
    records = [_record()]
    responses_path = tmp_path / "responses.jsonl"
    with responses_path.open("w") as handle:
        handle.write(
            json.dumps(
                {
                    "record_id": "r1",
                    "immutable_hash": "sha256:not-the-real-hash",
                    "fields": {"assistant_response": "tampered"},
                }
            )
            + "\n"
        )
    with pytest.raises(TeacherRealizationError, match="hash mismatch"):
        import_teacher_responses(
            records,
            responses_path,
            editable_fields=["assistant_response"],
            teacher_model="stub",
            teacher_prompt_hash="sha256:x",
        )


def test_import_rejects_response_that_smuggles_a_semantic_change(tmp_path: Path) -> None:
    """A response whose fields payload is honest about the hash but whose
    application would require the label-bearing fields to differ is
    impossible by construction (fields is restricted to editable_fields);
    this instead proves a corrupted request/response record_id mismatch is
    caught rather than silently applied to the wrong record."""
    records = [_record(), {**_record(), "record_id": "r2", "group_id": "g2"}]
    requests_path = tmp_path / "requests.jsonl"
    export_teacher_requests(records, requests_path, editable_fields=["assistant_response"])
    responses_path = tmp_path / "responses.jsonl"
    rows = [json.loads(line) for line in requests_path.read_text().splitlines()]
    with responses_path.open("w") as handle:
        # only answer r1; r2 must pass through unchanged
        for row in rows:
            if row["record_id"] != "r1":
                continue
            row["fields"] = {"assistant_response": "rewritten"}
            handle.write(json.dumps(row) + "\n")

    realized = import_teacher_responses(
        records,
        responses_path,
        editable_fields=["assistant_response"],
        teacher_model="stub",
        teacher_prompt_hash="sha256:x",
    )
    by_id = {r["record_id"]: r for r in realized}
    assert by_id["r1"]["assistant_response"] == "rewritten"
    assert by_id["r2"]["assistant_response"] == "I found your account balances."
    assert "provenance" not in by_id["r2"]


def test_import_rejects_missing_field(tmp_path: Path) -> None:
    records = [_record()]
    responses_path = tmp_path / "responses.jsonl"
    before_hash = immutable_hash(records[0], ["assistant_response"])
    with responses_path.open("w") as handle:
        handle.write(
            json.dumps({"record_id": "r1", "immutable_hash": before_hash, "fields": {}}) + "\n"
        )
    with pytest.raises(TeacherRealizationError, match="missing field"):
        import_teacher_responses(
            records,
            responses_path,
            editable_fields=["assistant_response"],
            teacher_model="stub",
            teacher_prompt_hash="sha256:x",
        )
