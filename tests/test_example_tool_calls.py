"""End-to-end test for the tool-calling worked example.

Covers the mechanisms the example exists to demonstrate: determinism, the
release gates, the pre-scrub stamp and its projection tag, the frozen-split
exemption from the banned-wording gate, teacher provenance -- and the two ways
the build is supposed to fail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dataforge.rows import make_conversation_row
from dataforge.teacher import compute_teacher_prompt_hash
from examples.banking import build_tool_calls
from examples.banking.build_tool_calls import (
    TEACHER_CLOSER,
    TEACHER_MODEL,
    VOICE_SPEC_PATH,
    _stub_teacher_rewrite,
    build,
)
from examples.banking.taxonomy import TAXONOMY
from examples.banking.tool_curricula import BANNED_WORDING, SCRUB_RECORD_ID

BUILT_FILES = (
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "manifest.json",
    "README.md",
    "source.lock.json",
    "teacher_requests.jsonl",
    "teacher_responses.jsonl",
)


def _rows(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        split: [json.loads(line) for line in (run_dir / f"{split}.jsonl").read_text().splitlines()]
        for split in ("train", "validation", "test")
    }


def _row_by_id(run_dir: Path, record_id: str) -> dict[str, Any]:
    for rows in _rows(run_dir).values():
        for row in rows:
            if row["record_id"] == record_id:
                return row
    raise AssertionError(f"{record_id} was not emitted")


def _duplicate_user_text_row() -> dict[str, Any]:
    """A train row reusing another train row's ``user_text`` with no pair id.

    Structurally indistinguishable from the governed counterfactual pair
    except for the one thing that makes that pair legitimate.
    """
    return make_conversation_row(
        record_id="train-ungoverned-duplicate-0",
        context_messages=[
            {"role": "user", "content": "my wallet is at home somewhere", "loss": False},
            {"role": "assistant", "content": "I can still help with your cards.", "loss": False},
        ],
        user_text="freeze the travel debit one for now",
        action_turns=[],
        final_response="Just to confirm, should I freeze the travel debit ending in 4821?",
        labels=TAXONOMY.labels_for_example(
            intent="freeze_card", action="clarify", entity_resolution="missing"
        ),
        example_kind="freeze_card_ungoverned",
        source="synthetic-ungoverned",
        source_split="train",
        group_id="tool-ungoverned|train|0",
    )


def test_tool_calls_example_is_deterministic(tmp_path: Path) -> None:
    build(tmp_path / "run1")
    build(tmp_path / "run2")
    for name in BUILT_FILES:
        assert (tmp_path / "run1" / name).read_bytes() == (tmp_path / "run2" / name).read_bytes()


def test_tool_calls_example_passes_its_release_gates(tmp_path: Path) -> None:
    manifest = build(tmp_path / "run1")
    report = manifest["report"]
    assert report["pii_matches"] == 0
    leakage = report["leakage"]
    # The gate default_gates applies, asserted key by key so a check that
    # silently stopped running would show up as a missing key, not a pass.
    for key in (
        "banned_wording_leak_count",
        "final_response_duplicate_leak_count",
        "user_text_split_leak_count",
        "group_split_leak_count",
    ):
        assert leakage[key] == 0
    assert leakage["banned_wording_leaks"] == []
    assert report["split_counts"] == {"train": 5, "validation": 3, "test": 2}


def test_scrubbed_row_carries_a_projection_tagged_pre_scrub_stamp(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    build(run_dir)
    row = _row_by_id(run_dir, SCRUB_RECORD_ID)
    stamps = row["provenance"]["pre_scrub_immutable_hashes"]
    assert len(stamps) == 1
    # The tag is the *teacher's* projection, not the scrubbed field's: that is
    # the exclusion set import_teacher_responses recomputes when it decides
    # whether the pre-scrub request file still applies.
    assert stamps[0]["excluded_fields"] == ["final_response", "messages"]
    assert stamps[0]["before_hash"] != stamps[0]["after_hash"]
    # The scrub actually happened, and the derived transcript followed it.
    context_text = " ".join(
        message["content"]
        for message in row["context_messages"]
        if isinstance(message.get("content"), str)
    )
    assert "mobile app" not in context_text
    assert "while I am going through my accounts" in context_text
    assert "while I am going through my accounts" in row["text"]


def test_banned_wording_survives_only_in_the_frozen_split(tmp_path: Path) -> None:
    build(tmp_path / "run1")
    rows = _rows(tmp_path / "run1")
    frozen = [row for row in rows["test"] if "shown in the app" in row["final_response"]]
    assert len(frozen) == 1
    assert "shown in the app" in frozen[0]["messages"][-1]["content"]
    for split in ("train", "validation"):
        for row in rows[split]:
            for message in row["messages"]:
                content = message.get("content")
                if isinstance(content, str):
                    assert BANNED_WORDING.search(content) is None, (
                        f"{row['record_id']} ({split}) still contains banned wording"
                    )


def test_realized_rows_carry_the_teacher_model_and_prompt_hash(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    build(run_dir)
    expected_hash = compute_teacher_prompt_hash(VOICE_SPEC_PATH, run_dir / "teacher_requests.jsonl")
    rows = _rows(run_dir)
    for split in ("train", "validation"):
        for row in rows[split]:
            provenance = row["provenance"]
            assert provenance["teacher_model"] == TEACHER_MODEL
            assert provenance["teacher_prompt_hash"] == expected_hash
            assert row["final_response"].endswith(TEACHER_CLOSER)
            assert row["messages"][-1]["content"] == row["final_response"]
    # Frozen rows were never exported, so they carry no teacher provenance.
    for row in rows["test"]:
        assert "provenance" not in row
        assert not row["final_response"].endswith(TEACHER_CLOSER)


def test_a_teacher_that_smuggles_banned_wording_fails_the_batch_check(tmp_path: Path) -> None:
    def tampering_teacher(record_id: str, final_response: str) -> str:
        rewritten = _stub_teacher_rewrite(record_id, final_response)
        if record_id == "train-list-cards-0":
            return f"{rewritten} You can also see this in the demo."
        return rewritten

    with pytest.raises(ValueError, match="teacher batch check failed") as error:
        build(tmp_path / "run1", teacher_rewrite=tampering_teacher)
    message = str(error.value)
    # Rule name and record id only: the checker owns its detail strings and is
    # free to reword them, but which rule fired on which row is the contract.
    assert "train-list-cards-0" in message
    assert "banned_pattern" in message
    assert not (tmp_path / "run1" / "train.jsonl").exists()


def test_without_the_scrub_the_banned_wording_gate_fails_the_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scrub is load bearing: skip it and write_dataset refuses to emit.

    Proves the banned-wording gate is not passing vacuously -- the phrase the
    scrub removes is exactly what the gate is looking for, and it reaches the
    gate through the derived ``messages`` of a context turn no teacher can edit.
    """
    monkeypatch.setattr(build_tool_calls, "_scrub_context_wording", lambda splits: None)
    with pytest.raises(ValueError, match="nonempty banned_wording_leaks"):
        build(tmp_path / "run1")
    assert not (tmp_path / "run1" / "train.jsonl").exists()


def test_an_ungoverned_duplicate_user_text_fails_the_pre_dedup_check(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="^pre-dedup check failed:") as error:
        build(tmp_path / "run1", seed_splits={"train": [_duplicate_user_text_row()]})
    message = str(error.value)
    assert "user_text_duplicate_leak_count" in message
    assert "freeze the travel debit one for now" in message
    assert not (tmp_path / "run1" / "train.jsonl").exists()
