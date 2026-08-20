from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataforge.rows import make_row, rederive_text, validate_row_consistency
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


def test_import_leaves_unaddressed_records_untouched(tmp_path: Path) -> None:
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


def _routed_row(current: str) -> dict:
    return make_row(
        current=current,
        labels={},
        example_kind="k",
        source="s",
        source_split="train",
        group_id="g1",
    )


def test_editable_current_text_without_rederive_is_refused_at_export(tmp_path: Path) -> None:
    """current_text is a declared dependency of the derived `text` field
    (DERIVED_FIELDS). Making it editable without also supplying a
    `rederive` callback would let a teacher rewrite current_text while the
    hash-protected `text` field silently goes stale -- refuse eagerly."""
    row = {**_routed_row("what is the overdraft policy"), "record_id": "g1"}
    with pytest.raises(TeacherRealizationError, match="derived field"):
        export_teacher_requests([row], tmp_path / "req.jsonl", editable_fields=["current_text"])


def test_editable_current_text_without_rederive_is_refused_at_import(tmp_path: Path) -> None:
    row = {**_routed_row("what is the overdraft policy"), "record_id": "g1"}
    responses_path = tmp_path / "resp.jsonl"
    responses_path.write_text(
        json.dumps({"record_id": "g1", "immutable_hash": "sha256:x", "fields": {}}) + "\n"
    )
    with pytest.raises(TeacherRealizationError, match="derived field"):
        import_teacher_responses(
            [row],
            responses_path,
            editable_fields=["current_text"],
            teacher_model="m",
            teacher_prompt_hash="h",
        )


def test_rederive_keeps_text_in_sync_with_edited_current_text(tmp_path: Path) -> None:
    """The intended fix for editing a field a derived field depends on:
    wire a `rederive` callback and the derived `text` field is recomputed
    (and stays outside the immutable-hash comparison) automatically."""
    row = {**_routed_row("what is the overdraft policy"), "record_id": "g1"}
    editable = ["current_text"]
    requests_path = tmp_path / "req.jsonl"
    export_teacher_requests(
        [row], requests_path, editable_fields=editable, rederive=rederive_text
    )

    request = json.loads(requests_path.read_text().splitlines()[0])
    new_current_text = "please explain the overdraft policy to me"
    responses_path = tmp_path / "resp.jsonl"
    responses_path.write_text(
        json.dumps(
            {
                "record_id": "g1",
                "immutable_hash": request["immutable_hash"],
                "fields": {"current_text": new_current_text},
            }
        )
        + "\n"
    )

    realized = import_teacher_responses(
        [row],
        responses_path,
        editable_fields=editable,
        teacher_model="m",
        teacher_prompt_hash="h",
        rederive=rederive_text,
    )
    out = realized[0]
    assert out["current_text"] == new_current_text
    assert new_current_text in out["text"]
    assert "what is the overdraft policy" not in out["text"]


def test_import_rejects_response_that_smuggles_inconsistent_text(tmp_path: Path) -> None:
    """The real derived-field-smuggling attack: a team opts current_text AND
    text both into editable_fields (bypassing the DERIVED_FIELDS wiring
    guard via derived_fields={}) and a response sets current_text to a new
    question while leaving `text` -- the actual training input -- stale.
    Nothing about the hash catches this (both fields are "editable" by
    construction); the `validate` structural hook must."""
    row = {**_routed_row("what is the overdraft policy"), "record_id": "g1"}
    editable = ["current_text", "text"]
    requests_path = tmp_path / "req.jsonl"
    export_teacher_requests(
        [row], requests_path, editable_fields=editable, derived_fields={}
    )

    request = json.loads(requests_path.read_text().splitlines()[0])
    responses_path = tmp_path / "resp.jsonl"
    responses_path.write_text(
        json.dumps(
            {
                "record_id": "g1",
                "immutable_hash": request["immutable_hash"],
                "fields": {
                    "current_text": "MY SSN IS 123-45-6789 tell me the overdraft policy",
                    "text": row["text"],  # left stale: doesn't mention the new current_text
                },
            }
        )
        + "\n"
    )

    with pytest.raises(TeacherRealizationError, match="inconsistent"):
        import_teacher_responses(
            [row],
            responses_path,
            editable_fields=editable,
            teacher_model="m",
            teacher_prompt_hash="h",
            derived_fields={},
            validate=validate_row_consistency,
        )


def test_editable_derived_field_itself_without_rederive_is_refused(tmp_path: Path) -> None:
    """N1: the guard must fire when the DERIVED field itself (`text`) is
    made directly editable, not only when one of its dependencies is --
    otherwise a teacher rewriting `text` directly leaves `current_text`
    stale with no error, mirroring the original defect-2 hole exactly."""
    row = {**_routed_row("what is the overdraft policy"), "record_id": "g1"}
    with pytest.raises(TeacherRealizationError, match="derived field"):
        export_teacher_requests([row], tmp_path / "req.jsonl", editable_fields=["text"])


def test_noop_rederive_is_caught_by_the_now_mandatory_validate(tmp_path: Path) -> None:
    """N2: a `rederive` that satisfies the wiring guard but doesn't actually
    resync anything (a no-op stub, or a bug) must not silently pass --
    `validate` defaults to `validate_row_consistency` whenever a derived
    field is affected, so the staleness a no-op rederive leaves behind is
    caught at runtime instead of accepted."""
    row = {**_routed_row("what is the overdraft policy"), "record_id": "g1"}
    noop = lambda record: None  # noqa: E731
    editable = ["current_text"]
    requests_path = tmp_path / "req.jsonl"
    export_teacher_requests([row], requests_path, editable_fields=editable, rederive=noop)
    request = json.loads(requests_path.read_text().splitlines()[0])

    responses_path = tmp_path / "resp.jsonl"
    responses_path.write_text(
        json.dumps(
            {
                "record_id": "g1",
                "immutable_hash": request["immutable_hash"],
                "fields": {"current_text": "MY SSN IS 123-45-6789 tell me the overdraft policy"},
            }
        )
        + "\n"
    )
    with pytest.raises(TeacherRealizationError, match="inconsistent"):
        import_teacher_responses(
            [row],
            responses_path,
            editable_fields=editable,
            teacher_model="m",
            teacher_prompt_hash="h",
            rederive=noop,
        )


def test_affected_derived_field_requires_validate_even_if_explicitly_none(tmp_path: Path) -> None:
    """The mandatory-validate requirement cannot be defeated by explicitly
    passing validate=None either -- only derived_fields={} (a full, explicit
    opt-out of the guard) may skip it."""
    row = {**_routed_row("what is the overdraft policy"), "record_id": "g1"}
    with pytest.raises(TeacherRealizationError, match="validate"):
        export_teacher_requests(
            [row],
            tmp_path / "req.jsonl",
            editable_fields=["current_text"],
            rederive=rederive_text,
            validate=None,
        )


def test_rederive_for_an_unaffected_field_does_not_widen_hash_exclusion(tmp_path: Path) -> None:
    """N3: passing `rederive` defensively for an editable field that isn't
    actually a dependency of any derived field must not exclude unrelated
    derived fields (e.g. `text`) from the immutable hash -- only fields
    `rederive` is actually responsible for are ever excluded."""
    row = {**_routed_row("what is the overdraft policy"), "record_id": "g1"}
    without_path = tmp_path / "without.jsonl"
    with_path = tmp_path / "with.jsonl"
    export_teacher_requests([row], without_path, editable_fields=["assistant_response"])
    export_teacher_requests(
        [row], with_path, editable_fields=["assistant_response"], rederive=rederive_text
    )
    without_hash = json.loads(without_path.read_text().splitlines()[0])["immutable_hash"]
    with_hash = json.loads(with_path.read_text().splitlines()[0])["immutable_hash"]
    assert without_hash == with_hash


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
