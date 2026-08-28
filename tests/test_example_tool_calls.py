"""End-to-end test for the tool-calling worked example.

Covers the mechanisms the example exists to demonstrate: determinism, the
release gates, the pre-scrub stamp and its projection tag, the frozen-split
exemption from the banned-wording gate, teacher provenance, the hand-authored
behaviour curriculum and its ``uses`` tag -- and every way the build is supposed
to fail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dataforge.curricula import foreign_use_rows
from dataforge.rows import make_conversation_row
from dataforge.teacher import compute_teacher_prompt_hash
from examples.banking import build_tool_calls
from examples.banking.build_tool_calls import (
    ROUTER_USE,
    TEACHER_CLOSER,
    TEACHER_MODEL,
    VOICE_SPEC_PATH,
    _stub_teacher_rewrite,
    build,
)
from examples.banking.taxonomy import TAXONOMY
from examples.banking.tool_curricula import (
    BANNED_WORDING,
    BEHAVIOUR_CURRICULUM,
    CURRICULUM_FIELD,
    EVAL_PROBES,
    REFUSAL_HONESTY_SEED,
    REGISTRY,
    SCRUB_RECORD_ID,
)

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
    # The pre-dedup invariant gates, likewise by key: each carries only a
    # `_leaks` list, and an empty one is what "it ran and found nothing" looks
    # like -- a missing key would be "it never ran".
    for key in ("field_invariant_leaks", "probe_exclusion_leaks", "unsupported_claim_leaks"):
        assert leakage[key] == []
    assert report["split_counts"] == {"train": 11, "validation": 5, "test": 2}
    # compose's pre-dedup findings are carried into the emitted report, so the
    # manifest on disk distinguishes "the gate ran and passed" from "it was
    # never wired". build_report cannot recompute them: they are about the rows
    # as they stood before deduplication.
    assert leakage["user_text_duplicate_leak_count"] == 0
    assert leakage["user_text_duplicate_leaks"] == []


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


def test_identical_finals_for_the_governed_pair_are_a_checker_finding(tmp_path: Path) -> None:
    """The pair exemption covers the shared utterance, never a shared answer.

    Without this, the collision surfaces only as `default_gates` refusing the
    release with no record id and no rule name -- while the checker, whose job
    is to hand back actionable per-row findings, reports clean.
    """

    def colliding_teacher(record_id: str, final_response: str) -> str:
        if record_id in {"train-pair-execute-0", "train-pair-clarify-0"}:
            return "I have taken care of that card for you right away."
        return _stub_teacher_rewrite(record_id, final_response)

    with pytest.raises(ValueError, match="teacher batch check failed") as error:
        build(tmp_path / "run1", teacher_rewrite=colliding_teacher)
    message = str(error.value)
    # Rule name and the field it is about; the detail wording is the checker's.
    assert "unique_normalized" in message
    assert "final_response" in message
    assert "train-pair-clarify-0" in message
    assert not (tmp_path / "run1" / "train.jsonl").exists()


def test_an_ungoverned_duplicate_user_text_fails_the_pre_dedup_check(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="^pre-dedup check failed:") as error:
        build(tmp_path / "run1", seed_splits={"train": [_duplicate_user_text_row()]})
    message = str(error.value)
    assert "user_text_duplicate_leak_count" in message
    assert "freeze the travel debit one for now" in message
    assert not (tmp_path / "run1" / "train.jsonl").exists()


def _behaviour_rows(run_dir: Path, split: str) -> list[dict[str, Any]]:
    return [row for row in _rows(run_dir)[split] if row["curriculum"] == BEHAVIOUR_CURRICULUM]


def test_the_behaviour_curriculum_repeats_one_mapping_across_frames(tmp_path: Path) -> None:
    run_dir = tmp_path / "run1"
    build(run_dir)
    train = _behaviour_rows(run_dir, "train")
    assert len(train) == 6  # 3 frames x 2 train subjects: the repetition is the point
    assert {row["behaviour"] for row in train} == {"refusal_honesty"}
    assert {row["action_name"] for row in train} == {"refuse_out_of_scope"}


def test_the_behaviour_validation_subject_is_held_back_from_train(tmp_path: Path) -> None:
    """The generalization signal: validation is scored on a subject train never
    saw, so passing it cannot be recall of a memorized frame."""
    run_dir = tmp_path / "run1"
    build(run_dir)
    (held_back,) = REFUSAL_HONESTY_SEED.subjects["validation"]
    train_text = " ".join(row["user_text"] for row in _behaviour_rows(run_dir, "train"))
    validation = _behaviour_rows(run_dir, "validation")
    assert held_back not in train_text
    assert len(validation) == 2  # first two frames only
    assert all(held_back in row["user_text"] for row in validation)


def test_the_router_export_excludes_the_sft_only_behaviour_family() -> None:
    """Both halves of the ``uses`` policy: the registry never builds the family
    into a router export, and the audit agrees about a build that contains it."""
    everything = REGISTRY.build("train")
    router = REGISTRY.build("train", use=ROUTER_USE)
    assert any(row[CURRICULUM_FIELD] == BEHAVIOUR_CURRICULUM for row in everything)
    assert all(row[CURRICULUM_FIELD] != BEHAVIOUR_CURRICULUM for row in router)
    foreign = foreign_use_rows(
        REGISTRY, everything, use=ROUTER_USE, name_field=CURRICULUM_FIELD
    )
    assert {row["record_id"] for row in foreign} == {
        f"train-refusal-honesty-{index}" for index in range(6)
    }
    assert foreign_use_rows(REGISTRY, router, use=ROUTER_USE, name_field=CURRICULUM_FIELD) == []


def _seed_row(record_id: str, user_text: str, final_response: str, **extra: Any) -> dict[str, Any]:
    row = make_conversation_row(
        record_id=record_id,
        context_messages=[],
        user_text=user_text,
        action_turns=[],
        final_response=final_response,
        labels=TAXONOMY.labels_for_example(intent=None),
        example_kind="seeded_defect",
        source="synthetic-seeded",
        source_split="train",
        group_id=f"seeded|{record_id}",
    )
    row.update(extra)
    return row


def test_a_probe_that_reaches_training_fails_the_build(tmp_path: Path) -> None:
    """The gate behind "the model passed the probe": without it, a probe that
    leaked into training makes the evaluation a memorization test."""
    leaked = _seed_row(
        "train-leaked-probe-0",
        EVAL_PROBES[0],
        "That charge was authorized before the freeze, so it has not settled yet.",
    )
    with pytest.raises(ValueError, match="^pre-dedup check failed:") as error:
        build(tmp_path / "run1", seed_splits={"train": [leaked]})
    assert "probe_exclusion_leaks" in str(error.value)
    assert not (tmp_path / "run1" / "train.jsonl").exists()


def test_a_behaviour_row_that_breaks_an_invariant_fails_the_build(tmp_path: Path) -> None:
    broken = _seed_row(
        "train-broken-behaviour-0",
        "can you refinance my mortgage from this chat",
        "Which part of that did you want me to look at first?",
        **{CURRICULUM_FIELD: BEHAVIOUR_CURRICULUM},
    )
    with pytest.raises(ValueError, match="^pre-dedup check failed:") as error:
        build(tmp_path / "run1", seed_splits={"train": [broken]})
    message = str(error.value)
    assert "field_invariant_leaks" in message
    assert "no_questions" in message
    assert not (tmp_path / "run1" / "train.jsonl").exists()


def test_a_final_claiming_an_action_it_never_took_fails_the_build(tmp_path: Path) -> None:
    """No tool turns, so the claim is a fabrication -- and training on it teaches
    the sentence rather than the action."""
    fabricated = _seed_row(
        "train-fabricated-claim-0",
        "please deal with the card i mentioned a moment ago",
        "I have frozen that card for you and nothing else on the account changed.",
    )
    with pytest.raises(ValueError, match="^pre-dedup check failed:") as error:
        build(tmp_path / "run1", seed_splits={"train": [fabricated]})
    assert "unsupported_claim_leaks" in str(error.value)
    assert not (tmp_path / "run1" / "train.jsonl").exists()


def test_every_curriculum_stamps_its_own_name_on_its_rows() -> None:
    """``foreign_use_rows`` can only judge a row whose curriculum it can name,
    so a stamp that drifted from its registration would disarm the audit while
    still reporting clean."""
    for registered in REGISTRY.curricula:
        for split in registered.splits:
            for row in registered.func(split):
                assert row[CURRICULUM_FIELD] == registered.name
