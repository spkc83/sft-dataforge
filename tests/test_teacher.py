from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from dataforge.rows import make_row, rederive_text, validate_row_consistency
from dataforge.teacher import (
    TeacherRealizationError,
    compute_teacher_prompt_hash,
    export_teacher_requests,
    immutable_hash,
    import_teacher_responses,
    scrub_fields,
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


# --- provenance outside the hash -------------------------------------------


def test_immutable_hash_ignores_provenance() -> None:
    """Provenance is written *by* the pipeline the hash protects (teacher
    stamps, pre-scrub stamps), so hashing it would make a record's own hash
    move the moment it was realized."""
    record = _record()
    hash_before = immutable_hash(record, ["assistant_response"])
    record["provenance"] = {"teacher_model": "stub", "teacher_prompt_hash": "sha256:x"}
    assert immutable_hash(record, ["assistant_response"]) == hash_before


def test_immutable_hash_honours_a_custom_provenance_field_name() -> None:
    record = _record()
    hash_before = immutable_hash(record, ["assistant_response"], provenance_field="lineage")
    record["lineage"] = {"teacher_model": "stub"}
    assert immutable_hash(record, ["assistant_response"], provenance_field="lineage") == hash_before
    # ...and the default name is then hashed like any other field
    record["provenance"] = {"teacher_model": "stub"}
    assert immutable_hash(record, ["assistant_response"], provenance_field="lineage") != hash_before


def test_export_import_export_round_trip_is_idempotent(tmp_path: Path) -> None:
    """The point of keeping provenance out of the hash: a second export from
    already-realized records reproduces the same immutable hash, so requests
    exported before the import are still valid afterwards."""
    records = [_record()]
    first_path = tmp_path / "req1.jsonl"
    export_teacher_requests(records, first_path, editable_fields=["assistant_response"])
    first_request = json.loads(first_path.read_text().splitlines()[0])

    responses_path = tmp_path / "resp.jsonl"
    responses_path.write_text(
        json.dumps(
            {
                "record_id": "r1",
                "immutable_hash": first_request["immutable_hash"],
                "fields": {"assistant_response": "Here are your balances."},
            }
        )
        + "\n"
    )
    realized = import_teacher_responses(
        records,
        responses_path,
        editable_fields=["assistant_response"],
        teacher_model="stub",
        teacher_prompt_hash="sha256:x",
    )
    assert "teacher_realization_hash" in realized[0]["provenance"]

    second_path = tmp_path / "req2.jsonl"
    export_teacher_requests(realized, second_path, editable_fields=["assistant_response"])
    second_request = json.loads(second_path.read_text().splitlines()[0])
    assert second_request["immutable_hash"] == first_request["immutable_hash"]

    # and the original response file still imports against the realized records
    import_teacher_responses(
        realized,
        responses_path,
        editable_fields=["assistant_response"],
        teacher_model="stub",
        teacher_prompt_hash="sha256:x",
    )


# --- scrub_fields -----------------------------------------------------------

APP = [("the mobile app", "your account")]


def _scrubbable_record() -> dict:
    return {
        "record_id": "r1",
        "group_id": "g1",
        "context_note": "while I check the mobile app",
        "assistant_response": "I found your account balances.",
    }


def _context_record() -> dict:
    return {
        "record_id": "r1",
        "group_id": "g1",
        "context_messages": [
            {"role": "system", "content": "Never mention the mobile app."},
            {"role": "user", "content": "is the mobile app down"},
            {"role": "assistant", "content": "the mobile app is fine"},
            {"role": "tool", "content": "the mobile app status: ok"},
        ],
        "assistant_response": "I found your account balances.",
    }


def _history_row() -> dict:
    row = make_row(
        current="what is the overdraft policy",
        history=[
            {"role": "user", "content": "is the mobile app down"},
            {"role": "assistant", "content": "the mobile app is fine"},
        ],
        labels={},
        example_kind="k",
        source="s",
        source_split="train",
        group_id="g1",
    )
    row["record_id"] = "g1"
    row["assistant_response"] = "I found your account balances."
    return row


def test_scrub_refuses_a_field_the_teacher_is_allowed_to_edit() -> None:
    """F10: editable text belongs to the teacher. Scrubbing it here would race
    the realization (and be overwritten by it); constrain the teacher's own
    output instead."""
    record = _scrubbable_record()
    with pytest.raises(TeacherRealizationError, match="editable"):
        scrub_fields(
            record,
            APP,
            fields=["assistant_response"],
            editable_fields=["assistant_response"],
        )


def test_scrub_of_a_derived_dependency_without_rederive_is_refused() -> None:
    """`history` feeds the derived `text`; scrubbing it without rederive would
    leave the field a model actually trains on stale -- the same invariant the
    teacher path enforces."""
    row = _history_row()
    with pytest.raises(TeacherRealizationError, match="derived field"):
        scrub_fields(row, APP, fields=["history"], editable_fields=["assistant_response"])


def test_scrub_of_a_derived_dependency_rederives_when_wired() -> None:
    row = _history_row()
    changed = scrub_fields(
        row,
        APP,
        fields=["history"],
        editable_fields=["assistant_response"],
        rederive=rederive_text,
    )
    assert changed is True
    assert row["history"][0]["content"] == "is your account down"
    assert "the mobile app" not in row["text"]
    assert "is your account down" in row["text"]
    validate_row_consistency(row)


def test_scrub_of_a_message_list_only_touches_the_named_roles() -> None:
    record = _context_record()
    changed = scrub_fields(
        record,
        APP,
        fields=["context_messages"],
        editable_fields=["assistant_response"],
    )
    assert changed is True
    contents = [message["content"] for message in record["context_messages"]]
    assert contents == [
        "Never mention the mobile app.",  # system: untouched
        "is your account down",
        "your account is fine",
        "the mobile app status: ok",  # tool: untouched
    ]


def test_scrub_that_changes_nothing_writes_no_stamp() -> None:
    record = _scrubbable_record()
    changed = scrub_fields(
        record,
        [("no such phrase", "x")],
        fields=["context_note"],
        editable_fields=["assistant_response"],
    )
    assert changed is False
    assert "provenance" not in record


def test_scrub_stamps_are_append_only_and_chain(tmp_path: Path) -> None:
    """F6: a second scrub must not overwrite the first stamp -- a teacher file
    exported before *either* scrub has to stay usable."""
    record = _scrubbable_record()
    record["provenance"] = {"source": "self-authored-synthetic"}
    original_hash = immutable_hash(record, ["assistant_response"])

    scrub_fields(record, APP, fields=["context_note"], editable_fields=["assistant_response"])
    scrub_fields(
        record,
        [("while I check", "one moment while I check")],
        fields=["context_note"],
        editable_fields=["assistant_response"],
    )

    stamps = record["provenance"]["pre_scrub_immutable_hashes"]
    assert len(stamps) == 2
    assert stamps[0]["before_hash"] == original_hash
    assert stamps[1]["before_hash"] == stamps[0]["after_hash"]
    assert stamps[1]["after_hash"] == immutable_hash(record, ["assistant_response"])
    assert record["provenance"]["source"] == "self-authored-synthetic"  # merged, not replaced


def test_scrub_stamp_is_tagged_with_the_teacher_projection_not_the_scrubbed_fields() -> None:
    """N6: the tag must be the exclusion set import_teacher_responses will
    recompute (editable_fields + the derived fields *they* affect), not the one
    implied by the scrubbed fields -- otherwise no stamp could ever match."""
    row = _history_row()
    scrub_fields(
        row,
        APP,
        fields=["history"],
        editable_fields=["assistant_response"],
        rederive=rederive_text,
    )
    stamp = row["provenance"]["pre_scrub_immutable_hashes"][0]
    assert stamp["excluded_fields"] == ["assistant_response"]


def test_scrub_stamp_tag_includes_derived_fields_the_teacher_affects() -> None:
    row = _history_row()
    scrub_fields(
        row,
        APP,
        fields=["history"],
        editable_fields=["current_text"],
        rederive=rederive_text,
    )
    stamp = row["provenance"]["pre_scrub_immutable_hashes"][0]
    assert stamp["excluded_fields"] == ["current_text", "text"]


# --- accept_pre_scrub_hashes ------------------------------------------------


def _export_then_scrub(tmp_path: Path) -> tuple[dict, Path]:
    """Export a request, then scrub a hashed context field behind it."""
    record = _scrubbable_record()
    requests_path = tmp_path / "req.jsonl"
    export_teacher_requests([record], requests_path, editable_fields=["assistant_response"])
    request = json.loads(requests_path.read_text().splitlines()[0])
    scrub_fields(record, APP, fields=["context_note"], editable_fields=["assistant_response"])
    assert immutable_hash(record, ["assistant_response"]) != request["immutable_hash"]

    responses_path = tmp_path / "resp.jsonl"
    responses_path.write_text(
        json.dumps(
            {
                "record_id": "r1",
                "immutable_hash": request["immutable_hash"],
                "fields": {"assistant_response": "Here are your balances."},
            }
        )
        + "\n"
    )
    return record, responses_path


def test_import_rejects_a_pre_scrub_request_by_default(tmp_path: Path) -> None:
    record, responses_path = _export_then_scrub(tmp_path)
    with pytest.raises(TeacherRealizationError, match="hash mismatch"):
        import_teacher_responses(
            [record],
            responses_path,
            editable_fields=["assistant_response"],
            teacher_model="m",
            teacher_prompt_hash="h",
        )


def test_import_accepts_a_pre_scrub_request_when_opted_in(tmp_path: Path) -> None:
    record, responses_path = _export_then_scrub(tmp_path)
    realized = import_teacher_responses(
        [record],
        responses_path,
        editable_fields=["assistant_response"],
        teacher_model="m",
        teacher_prompt_hash="h",
        accept_pre_scrub_hashes=True,
    )
    assert realized[0]["assistant_response"] == "Here are your balances."
    assert realized[0]["context_note"] == "while I check your account"


def test_import_rejects_a_stamp_tagged_with_a_different_projection(tmp_path: Path) -> None:
    """The stamped before_hash is right but describes a different hash
    projection; accepting it would let a stamp written for a narrow projection
    unlock a request pinned to a wide one."""
    record, responses_path = _export_then_scrub(tmp_path)
    record["provenance"]["pre_scrub_immutable_hashes"][0]["excluded_fields"] = [
        "assistant_response",
        "context_note",
    ]
    with pytest.raises(TeacherRealizationError, match="hash mismatch"):
        import_teacher_responses(
            [record],
            responses_path,
            editable_fields=["assistant_response"],
            teacher_model="m",
            teacher_prompt_hash="h",
            accept_pre_scrub_hashes=True,
        )


def test_a_benign_mutation_after_the_scrub_does_not_lock_the_old_request_out(
    tmp_path: Path,
) -> None:
    """N3: acceptance is by stamped before_hash alone -- no chain or
    after-hash condition. A later benign mutation (here: stamping a new field)
    moves the live hash again, and the pre-scrub teacher file must still
    import, which is the whole purpose of the mechanism."""
    record, responses_path = _export_then_scrub(tmp_path)
    record["batch_id"] = "b7"  # hashed, but semantically benign
    realized = import_teacher_responses(
        [record],
        responses_path,
        editable_fields=["assistant_response"],
        teacher_model="m",
        teacher_prompt_hash="h",
        accept_pre_scrub_hashes=True,
    )
    assert realized[0]["assistant_response"] == "Here are your balances."


def test_post_edit_semantic_change_is_still_rejected_with_pre_scrub_hashes_accepted(
    tmp_path: Path,
) -> None:
    """The widening is one-directional: it only widens what the *request* may
    pin. The post-edit check still compares against the record's live hash, so
    an edit that moves immutable semantics is rejected either way."""
    record, responses_path = _export_then_scrub(tmp_path)

    def sabotage(row: dict) -> None:
        row["group_id"] = "g2"

    with pytest.raises(TeacherRealizationError, match="changed immutable semantics"):
        import_teacher_responses(
            [record],
            responses_path,
            editable_fields=["assistant_response"],
            teacher_model="m",
            teacher_prompt_hash="h",
            accept_pre_scrub_hashes=True,
            derived_fields={},
            rederive=sabotage,
        )


# --- compute_teacher_prompt_hash -------------------------------------------


def test_compute_teacher_prompt_hash_is_deterministic_and_moves_with_either_file(
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "voice_spec.md"
    requests = tmp_path / "requests.jsonl"
    prompt.write_text("rewrite for fluency\n")
    requests.write_text('{"record_id": "r1"}\n')

    digest = compute_teacher_prompt_hash(prompt, requests)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    assert compute_teacher_prompt_hash(prompt, requests) == digest

    prompt.write_text("rewrite for fluency, briefly\n")
    moved_prompt = compute_teacher_prompt_hash(prompt, requests)
    assert moved_prompt != digest

    requests.write_text('{"record_id": "r2"}\n')
    assert compute_teacher_prompt_hash(prompt, requests) not in {digest, moved_prompt}


def test_compute_teacher_prompt_hash_does_not_collide_when_the_files_swap(
    tmp_path: Path,
) -> None:
    """The two digests are keyed, not concatenated, so a prompt and a request
    file with swapped contents are not the same provenance string."""
    left = tmp_path / "a.md"
    right = tmp_path / "b.jsonl"
    left.write_text("alpha\n")
    right.write_text("beta\n")
    assert compute_teacher_prompt_hash(left, right) != compute_teacher_prompt_hash(right, left)
