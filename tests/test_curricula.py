from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from dataforge.curricula import (
    DEFAULT_DEDUP_PRIORITY,
    BehaviourSeed,
    Registry,
    behaviour_rows,
    build_report,
    compose,
    foreign_use_rows,
)


def _row(
    text: str,
    group_id: str,
    trajectory_id: str | None = None,
    pair_id: str | None = None,
    current_text: str | None = None,
) -> dict:
    return {
        "text": text,
        "current_text": current_text if current_text is not None else text,
        "group_id": group_id,
        "trajectory_id": trajectory_id or group_id,
        "pair_id": pair_id,
        "example_kind": "unit",
    }


def test_dedup_prefers_eval_splits_by_default() -> None:
    registry = Registry()

    @registry.register("dupes", splits=("train", "validation", "test"))
    def dupes(split: str) -> list[dict]:
        # identical text in every split, distinct groups
        return [_row("duplicate text here", f"g-{split}")]

    splits, report = compose({"train": [], "validation": [], "test": []}, registry)
    # default priority is reversed(split_order) => test, validation, train: test wins
    assert len(splits["test"]) == 1
    assert len(splits["validation"]) == 0
    assert len(splits["train"]) == 0
    assert report["cross_split_duplicates_removed"] == 2


def test_dedup_priority_is_configurable() -> None:
    registry = Registry()

    @registry.register("dupes", splits=("train", "validation", "test"))
    def dupes(split: str) -> list[dict]:
        return [_row("duplicate text here", f"g-{split}")]

    splits, _ = compose(
        {"train": [], "validation": [], "test": []},
        registry,
        dedup_priority=("train", "validation", "test"),
    )
    assert len(splits["train"]) == 1
    assert len(splits["test"]) == 0


def test_group_leak_across_splits_raises() -> None:
    registry = Registry()

    @registry.register("bad", splits=("train", "test"))
    def bad(split: str) -> list[dict]:
        return [_row(f"text for {split}", "shared-group")]

    with pytest.raises(ValueError, match="shared-group"):
        compose({"train": [], "validation": [], "test": []}, registry)


def test_pair_leak_across_splits_raises() -> None:
    registry = Registry()

    @registry.register("bad-pair", splits=("train", "test"))
    def bad(split: str) -> list[dict]:
        return [_row(f"text for {split}", f"g-{split}", pair_id="shared-pair")]

    with pytest.raises(ValueError, match="shared-pair"):
        compose({"train": [], "validation": [], "test": []}, registry)


def test_report_counts_by_field() -> None:
    registry = Registry()

    @registry.register("counted", splits=("train",))
    def counted(split: str) -> list[dict]:
        return [_row("a", "g1"), _row("b", "g2")]

    _, report = compose(
        {"train": [], "validation": [], "test": []}, registry, count_fields=("example_kind",)
    )
    assert report["counts"]["example_kind"]["train"] == {"unit": 2}
    assert report["split_counts"]["train"] == 2


def test_held_out_texts_flagged_when_leaked_into_train() -> None:
    registry = Registry()

    @registry.register("leaky", splits=("train",))
    def leaky(split: str) -> list[dict]:
        return [_row("this is a held out phrase", "g1")]

    _, report = compose(
        {"train": [], "validation": [], "test": []},
        registry,
        held_out_texts=["this is a held out phrase"],
    )
    assert report["leakage"]["heldout_exact_leaks"]


def test_dedup_priority_default_is_a_fixed_constant_not_derived_from_split_order() -> None:
    """Reordering split_order (a cosmetic/emission-order knob) must never
    silently flip which split wins deduplication -- the default dedup
    priority is a fixed constant independent of split_order's permutation."""
    registry = Registry()

    @registry.register("dupes", splits=("train", "validation", "test"))
    def dupes(split: str) -> list[dict]:
        return [_row("duplicate text here", f"g-{split}")]

    for order in [
        ("train", "validation", "test"),
        ("test", "train", "validation"),
        ("validation", "test", "train"),
    ]:
        splits, _ = compose({k: [] for k in order}, registry, split_order=order)
        kept = {k for k, v in splits.items() if v}
        assert kept == {"test"}, f"split_order={order} flipped dedup winner to {kept}"


def test_dedup_priority_must_be_permutation_of_split_order() -> None:
    registry = Registry()

    @registry.register("dupes", splits=("train", "validation", "test", "holdout"))
    def dupes(split: str) -> list[dict]:
        return [_row("duplicate text here", f"g-{split}")]

    with pytest.raises(ValueError, match="dedup_priority"):
        compose(
            {"train": [], "validation": [], "test": [], "holdout": []},
            registry,
            split_order=("train", "validation", "test", "holdout"),
            dedup_priority=DEFAULT_DEDUP_PRIORITY,  # only ("test","validation","train")
        )


def test_dedup_priority_required_explicitly_for_nonstandard_split_names() -> None:
    registry = Registry()

    @registry.register("dupes", splits=("train", "holdout"))
    def dupes(split: str) -> list[dict]:
        return [_row("duplicate text here", f"g-{split}")]

    with pytest.raises(ValueError, match="dedup_priority"):
        compose({"train": [], "holdout": []}, registry, split_order=("train", "holdout"))

    # explicit dedup_priority works fine
    splits, _ = compose(
        {"train": [], "holdout": []},
        registry,
        split_order=("train", "holdout"),
        dedup_priority=("holdout", "train"),
    )
    assert {k for k, v in splits.items() if v} == {"holdout"}


def test_current_text_secondary_leak_catches_state_conditioned_reuse() -> None:
    """A state-conditioned followup can render very differently in `text`
    (a state prefix differs per split) while reusing the identical
    `current_text` across splits under deliberately distinct group ids --
    the group/trajectory checks can't see this; the secondary leak check on
    current_text (on by default) must."""
    registry = Registry()

    @registry.register("state_followup", splits=("train", "test"))
    def state_followup(split: str) -> list[dict]:
        return [
            _row(
                text=f"[PRIOR_STATE]\n{{'pending': '{split}'}}\n[CURRENT_USER]\nfollowup",
                group_id=f"state-family|{split}",
                current_text="can you explain that policy again",
            )
        ]

    _, report = compose({"train": [], "validation": [], "test": []}, registry)
    assert report["leakage"]["current_text_split_leak_count"] == 1
    assert report["leakage"]["group_split_leak_count"] == 0  # groups genuinely differ


def test_secondary_leak_fields_is_configurable() -> None:
    registry = Registry()

    @registry.register("r", splits=("train", "test"))
    def r(split: str) -> list[dict]:
        return [_row(f"unique text {split}", f"g-{split}", current_text="shared current text")]

    _, report_off = compose(
        {"train": [], "validation": [], "test": []}, registry, secondary_leak_fields=()
    )
    assert "current_text_split_leak_count" not in report_off["leakage"]


def test_extra_leak_checks_hook_merges_into_leakage() -> None:
    registry = Registry()

    @registry.register("r", splits=("train",))
    def r(split: str) -> list[dict]:
        return [_row("a", "g1")]

    def custom_check(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> Mapping[str, Any]:
        return {"custom_split_leaks": {}, "custom_split_leak_count": 0}

    _, report = compose(
        {"train": [], "validation": [], "test": []},
        registry,
        extra_leak_checks=[custom_check],
    )
    assert report["leakage"]["custom_split_leak_count"] == 0


def test_extra_leak_checks_reject_colliding_keys() -> None:
    registry = Registry()

    @registry.register("r", splits=("train",))
    def r(split: str) -> list[dict]:
        return [_row("a", "g1")]

    def colliding_check(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> Mapping[str, Any]:
        return {"group_split_leak_count": 999}

    with pytest.raises(ValueError, match="colliding"):
        compose(
            {"train": [], "validation": [], "test": []},
            registry,
            extra_leak_checks=[colliding_check],
        )


def test_extra_leakage_merges_precomputed_findings_into_the_report() -> None:
    """The rebuild path for findings build_report cannot recompute.

    ``compose``'s pre-dedup checks see the rows before deduplication; a report
    rebuilt on the deduplicated (or post-teacher) splits can only be handed
    their result.
    """
    report = build_report(
        {"train": [_row("a", "g1")], "validation": [], "test": []},
        extra_leakage={"user_text_duplicate_leaks": [], "user_text_duplicate_leak_count": 0},
    )
    assert report["leakage"]["user_text_duplicate_leak_count"] == 0
    assert report["leakage"]["user_text_duplicate_leaks"] == []


def test_extra_leakage_rejects_keys_colliding_with_the_built_in_leakage_keys() -> None:
    with pytest.raises(ValueError, match="colliding"):
        build_report(
            {"train": [_row("a", "g1")], "validation": [], "test": []},
            extra_leakage={"group_split_leak_count": 999},
        )


def test_extra_leakage_rejects_keys_colliding_with_an_extra_leak_check() -> None:
    def custom_check(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> Mapping[str, Any]:
        return {"custom_leak_count": 0}

    with pytest.raises(ValueError, match="colliding"):
        build_report(
            {"train": [_row("a", "g1")], "validation": [], "test": []},
            extra_leak_checks=[custom_check],
            extra_leakage={"custom_leak_count": 0},
        )


def test_extra_leakage_still_gates_a_nonzero_value_it_carries_forward() -> None:
    """A carried-forward finding is not a footnote: it gates like any other."""
    from dataforge.emit import default_gates

    report = build_report(
        {"train": [_row("a", "g1")], "validation": [], "test": []},
        extra_leakage={"user_text_duplicate_leak_count": 1},
    )
    with pytest.raises(ValueError, match="user_text_duplicate_leak_count"):
        default_gates(report)


def test_pii_fields_none_scans_every_string_field() -> None:
    registry = Registry()

    @registry.register("r", splits=("train",))
    def r(split: str) -> list[dict]:
        row = _row("clean text", "g1")
        row["assistant_response"] = "call me at 555-123-4567"
        return [row]

    _, report = compose({"train": [], "validation": [], "test": []}, registry)
    assert report["pii_matches"] > 0


def test_pii_fields_explicit_list_restricts_scan() -> None:
    registry = Registry()

    @registry.register("r", splits=("train",))
    def r(split: str) -> list[dict]:
        row = _row("clean text", "g1")
        row["assistant_response"] = "call me at 555-123-4567"
        return [row]

    _, report = compose(
        {"train": [], "validation": [], "test": []}, registry, pii_fields=("text",)
    )
    assert report["pii_matches"] == 0


def test_pii_scan_recurses_into_nested_dict_fields() -> None:
    """N4: PII living only inside a nested dict field (e.g. prior_state or
    provenance), never in a top-level string field, must still be caught."""
    row = _row("clean text with no pii", "g1")
    row["prior_state"] = {"caller_phone": "555-867-5309", "email": "a.customer@examplebank.com"}
    row["provenance"] = {"analyst_note": "SSN 123-45-6789 on file"}
    report = build_report({"train": [row], "validation": [], "test": []})
    assert report["pii_matches"] > 0


def test_pii_scan_recurses_into_nested_list_fields() -> None:
    row = _row("clean text with no pii", "g1")
    row["history"] = [{"role": "user", "content": "my ssn is 123-45-6789"}]
    report = build_report({"train": [row], "validation": [], "test": []})
    assert report["pii_matches"] > 0


def test_pii_fields_explicit_list_still_recurses_within_each_field() -> None:
    row = _row("clean text with no pii", "g1")
    row["provenance"] = {"analyst_note": "contact fraud@examplebank.com"}
    report = build_report(
        {"train": [row], "validation": [], "test": []}, pii_fields=("provenance",)
    )
    assert report["pii_matches"] > 0


def test_seed_split_outside_split_order_raises() -> None:
    registry = Registry()
    seed = {
        "train": [],
        "validation": [],
        "test": [],
        "calibration": [_row("calibration probe utterance", "cal|0")],
    }
    with pytest.raises(ValueError, match="calibration"):
        compose(seed, registry)


def test_curriculum_split_outside_split_order_raises() -> None:
    registry = Registry()

    @registry.register("r", splits=("train", "calibration"))
    def r(split: str) -> list[dict]:
        return [_row(f"text {split}", f"g-{split}")]

    with pytest.raises(ValueError, match="calibration"):
        compose({"train": [], "validation": [], "test": []}, registry)


def test_build_report_contract_and_fingerprint() -> None:
    splits = {"train": [_row("a", "g1")], "validation": [], "test": []}
    report = build_report(splits)
    assert report["contract"] == "dataforge-curriculum-report"
    assert report["splits_fingerprint"].startswith("sha256:")
    # a fresh build over identical content reproduces the same fingerprint
    assert build_report(splits)["splits_fingerprint"] == report["splits_fingerprint"]
    # mutating a row's content changes the fingerprint
    splits["train"][0]["text"] = "a changed"
    assert build_report(splits)["splits_fingerprint"] != report["splits_fingerprint"]


def test_within_split_duplicates_are_reported_separately_from_the_lumped_total() -> None:
    """N5: `cross_split_duplicates_removed` keeps its historical meaning (both
    kinds of removal); the new `within_split_duplicates_removed` is a sub-count
    of it, not a replacement."""
    registry = Registry()

    @registry.register("dupes", splits=("train", "test"))
    def dupes(split: str) -> list[dict]:
        if split == "train":
            return [
                _row("train only duplicate", "g-a"),
                _row("train only duplicate", "g-b"),
                _row("shared across splits", "g-c"),
            ]
        return [_row("shared across splits", "g-d")]

    splits, report = compose({"train": [], "validation": [], "test": []}, registry)
    assert len(splits["train"]) == 1
    assert len(splits["test"]) == 1
    assert report["cross_split_duplicates_removed"] == 2  # unchanged: both kinds
    assert report["within_split_duplicates_removed"] == 1


def test_build_report_defaults_within_split_duplicates_removed_to_zero() -> None:
    report = build_report({"train": [_row("a", "g1")]})
    assert report["within_split_duplicates_removed"] == 0


def test_pre_dedup_checks_run_before_dedup_can_hide_a_within_split_duplicate() -> None:
    """The vacuity fix: `compose` dedups on `text` before `build_report`, so a
    report-time check can never see a within-split `text` duplicate. A
    pre-dedup check can."""
    registry = Registry()

    @registry.register("dupes", splits=("train",))
    def dupes(split: str) -> list[dict]:
        return [_row("the very same text", "g1"), _row("the very same text", "g2")]

    seen: list[int] = []

    def counting_check(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> Mapping[str, Any]:
        seen.append(sum(len(rows) for rows in splits.values()))
        return {"prededup_rowcount_leaks": [], "prededup_rowcount_leak_count": 0}

    splits, report = compose(
        {"train": [], "validation": [], "test": []},
        registry,
        pre_dedup_checks=[counting_check],
    )
    assert seen == [2]  # both rows were visible to the check
    assert len(splits["train"]) == 1  # only one survived dedup
    assert report["leakage"]["prededup_rowcount_leak_count"] == 0


def test_pre_dedup_checks_fail_the_build_fast_with_their_findings() -> None:
    registry = Registry()

    @registry.register("r", splits=("train",))
    def r(split: str) -> list[dict]:
        return [_row("a", "g1")]

    def failing_check(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> Mapping[str, Any]:
        return {
            "user_text_duplicate_leaks": [{"normalized": "a", "members": []}],
            "user_text_duplicate_leak_count": 1,
        }

    with pytest.raises(ValueError, match="^pre-dedup check failed:"):
        compose(
            {"train": [], "validation": [], "test": []},
            registry,
            pre_dedup_checks=[failing_check],
        )


def test_pre_dedup_checks_fail_on_a_non_empty_leaks_list_with_no_count_key() -> None:
    registry = Registry()

    @registry.register("r", splits=("train",))
    def r(split: str) -> list[dict]:
        return [_row("a", "g1")]

    def failing_check(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> Mapping[str, Any]:
        return {"wording_leaks": [{"split": "train"}]}

    with pytest.raises(ValueError, match="wording_leaks"):
        compose(
            {"train": [], "validation": [], "test": []},
            registry,
            pre_dedup_checks=[failing_check],
        )


def test_pre_dedup_check_keys_are_carried_into_the_final_report_leakage() -> None:
    registry = Registry()

    @registry.register("r", splits=("train",))
    def r(split: str) -> list[dict]:
        return [_row("a", "g1")]

    def clean_check(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> Mapping[str, Any]:
        return {"wording_leaks": [], "wording_leak_count": 0}

    _, report = compose(
        {"train": [], "validation": [], "test": []},
        registry,
        pre_dedup_checks=[clean_check],
    )
    assert report["leakage"]["wording_leak_count"] == 0
    assert report["leakage"]["wording_leaks"] == []


def test_pre_dedup_checks_reject_keys_colliding_with_the_built_in_leakage_keys() -> None:
    registry = Registry()

    @registry.register("r", splits=("train",))
    def r(split: str) -> list[dict]:
        return [_row("a", "g1")]

    def colliding_check(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> Mapping[str, Any]:
        return {"group_split_leak_count": 0}

    with pytest.raises(ValueError, match="colliding"):
        compose(
            {"train": [], "validation": [], "test": []},
            registry,
            pre_dedup_checks=[colliding_check],
        )


def test_two_pre_dedup_checks_reject_colliding_keys_with_each_other() -> None:
    registry = Registry()

    @registry.register("r", splits=("train",))
    def r(split: str) -> list[dict]:
        return [_row("a", "g1")]

    def check(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> Mapping[str, Any]:
        return {"wording_leak_count": 0}

    with pytest.raises(ValueError, match="colliding"):
        compose(
            {"train": [], "validation": [], "test": []},
            registry,
            pre_dedup_checks=[check, check],
        )


def _seed(**overrides: Any) -> BehaviourSeed:
    defaults: dict[str, Any] = {
        "key": "unsupported_capability",
        "family": "refusal_honesty",
        "frames": ("can you {s} for me", "i need you to {s} today", "could you {s} from here"),
        "finals": ("I cannot {s}.", "That is outside what I can do, so I did not {s}.", "No: {s}."),
        "subjects": {
            "train": ("book a flight", "file a tax return"),
            "validation": ("buy shares",),
        },
    }
    defaults.update(overrides)
    return BehaviourSeed(**defaults)


def _behaviour_row(**kwargs: Any) -> dict[str, Any]:
    return {
        "text": kwargs["text"],
        "final_response": kwargs["final"],
        "group_id": f"{kwargs['seed'].key}|{kwargs['split']}|{kwargs['variant']}",
        "example_kind": kwargs["seed"].family,
    }


def test_behaviour_rows_gives_train_every_frame_and_other_splits_the_first_two() -> None:
    """Repetition across frames is what reaches the weights; an eval split only
    needs enough frames to tell whether the behaviour generalized."""
    seed = _seed()
    train = behaviour_rows([seed], "train", row_fn=_behaviour_row)
    validation = behaviour_rows([seed], "validation", row_fn=_behaviour_row)
    assert len(train) == 6  # 3 frames x 2 subjects
    assert len(validation) == 2  # 2 frames x 1 held-back subject


def test_behaviour_rows_frames_per_validation_subject_is_configurable() -> None:
    seed = _seed()
    assert len(behaviour_rows([seed], "validation", row_fn=_behaviour_row,
                              frames_per_validation_subject=3)) == 3
    # asking for more frames than the seed has yields every frame, not an error
    assert len(behaviour_rows([seed], "validation", row_fn=_behaviour_row,
                              frames_per_validation_subject=9)) == 3


def test_behaviour_rows_substitutes_the_subject_into_the_frame_and_the_final() -> None:
    rows = behaviour_rows([_seed()], "validation", row_fn=_behaviour_row)
    assert rows[0]["text"] == "can you buy shares for me"
    assert rows[0]["final_response"] == "I cannot buy shares."


def test_behaviour_rows_passes_row_fn_the_full_construction_context() -> None:
    seen: list[dict[str, Any]] = []

    def recording_row_fn(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs)
        return {"text": kwargs["text"]}

    seed = _seed()
    behaviour_rows([seed], "train", row_fn=recording_row_fn)
    assert [call["variant"] for call in seen] == [0, 1, 2, 3, 4, 5]
    assert [call["frame_index"] for call in seen] == [0, 1, 2, 0, 1, 2]
    assert [call["subject"] for call in seen] == ["book a flight"] * 3 + ["file a tax return"] * 3
    assert {call["split"] for call in seen} == {"train"}
    assert {id(call["seed"]) for call in seen} == {id(seed)}
    assert seen[0]["text"] == "can you book a flight for me"
    assert seen[0]["final"] == "I cannot book a flight."


def test_behaviour_rows_counts_variants_per_seed_not_per_call() -> None:
    seeds = [_seed(), _seed(key="other")]
    seen: list[int] = []

    def recording_row_fn(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs["variant"])
        return {}

    behaviour_rows(seeds, "validation", row_fn=recording_row_fn)
    assert seen == [0, 1, 0, 1]


def test_behaviour_rows_rejects_a_subject_used_in_both_train_and_a_held_back_split() -> None:
    seed = _seed(subjects={"train": ("book a flight",), "validation": ("book a flight",)})
    with pytest.raises(ValueError, match="generalization signal"):
        behaviour_rows([seed], "train", row_fn=_behaviour_row)


def test_behaviour_rows_rejects_frames_and_finals_of_different_lengths() -> None:
    seed = _seed(finals=("I cannot {s}.",))
    with pytest.raises(ValueError, match="3 frames but 1 finals"):
        behaviour_rows([seed], "train", row_fn=_behaviour_row)


def test_behaviour_rows_rejects_a_frame_without_a_subject_placeholder() -> None:
    seed = _seed(frames=("can you {s} for me", "i need help today", "could you {s} from here"))
    with pytest.raises(ValueError, match="frame 1 has no"):
        behaviour_rows([seed], "train", row_fn=_behaviour_row)


def test_behaviour_rows_rejects_two_seeds_sharing_a_family_and_key() -> None:
    with pytest.raises(ValueError, match="share \\(family, key\\)"):
        behaviour_rows([_seed(), _seed()], "train", row_fn=_behaviour_row)


def test_behaviour_rows_rejects_a_seed_with_no_subjects_for_the_split() -> None:
    with pytest.raises(ValueError, match="no subjects for split 'test'"):
        behaviour_rows([_seed()], "test", row_fn=_behaviour_row)


def test_behaviour_rows_rejects_an_empty_frame_list() -> None:
    with pytest.raises(ValueError, match="frames must not be empty"):
        behaviour_rows([_seed(frames=(), finals=())], "train", row_fn=_behaviour_row)


def test_behaviour_rows_tags_travel_with_the_seed_for_a_row_fn_to_copy() -> None:
    seed = _seed(tags=(("behaviour", "refusal_honesty"),))
    seen: list[dict[str, Any]] = []

    def recording_row_fn(**kwargs: Any) -> dict[str, Any]:
        seen.append(dict(kwargs["seed"].tags))
        return {}

    behaviour_rows([seed], "validation", row_fn=recording_row_fn)
    assert seen[0] == {"behaviour": "refusal_honesty"}


def _named_row(text: str, group_id: str, curriculum: str) -> dict:
    """A row carrying the name of the curriculum that produced it, which is
    what ``foreign_use_rows`` reads."""
    row = _row(text, group_id)
    row["curriculum"] = curriculum
    return row


def _use_registry() -> Registry:
    registry = Registry()

    @registry.register("shared", splits=("train", "validation"))
    def shared(split: str) -> list[dict]:
        return [_named_row(f"shared text for {split}", f"shared|{split}", "shared")]

    @registry.register("sft_only", splits=("train", "validation"), uses=("sft",))
    def sft_only(split: str) -> list[dict]:
        return [_named_row(f"sft only text for {split}", f"sft|{split}", "sft_only")]

    return registry


def test_build_without_a_use_is_unchanged_and_includes_every_curriculum() -> None:
    registry = _use_registry()
    assert len(registry.build("train")) == 2
    assert len(registry.build("train", use=None)) == 2


def test_build_with_a_use_drops_the_curricula_that_do_not_permit_it() -> None:
    registry = _use_registry()
    router = registry.build("train", use="router")
    assert [row["curriculum"] for row in router] == ["shared"]  # "*" still permits router
    sft = registry.build("train", use="sft")
    assert [row["curriculum"] for row in sft] == ["shared", "sft_only"]


def test_register_rejects_an_empty_or_blank_uses_tuple() -> None:
    registry = Registry()
    with pytest.raises(ValueError, match="at least one consumer"):
        registry.register("empty", splits=("train",), uses=())
    with pytest.raises(ValueError, match="non-blank strings"):
        registry.register("blank", splits=("train",), uses=("sft", "  "))


def test_foreign_use_rows_finds_the_rows_a_use_should_never_have_seen() -> None:
    registry = _use_registry()
    rows = registry.build("train") + registry.build("validation")
    foreign = foreign_use_rows(registry, rows, use="router", name_field="curriculum")
    assert [row["group_id"] for row in foreign] == ["sft|train", "sft|validation"]
    assert foreign_use_rows(registry, rows, use="sft", name_field="curriculum") == []


def test_foreign_use_rows_skips_rows_whose_curriculum_it_does_not_know() -> None:
    """A corpus assembled from several sources is still checkable against the
    one registry that knows about the restricted families."""
    registry = _use_registry()
    rows = [_named_row("from elsewhere", "g1", "not_registered"), {"group_id": "g2"}]
    assert foreign_use_rows(registry, rows, use="router", name_field="curriculum") == []


def test_compose_use_filters_the_build_end_to_end() -> None:
    registry = _use_registry()
    splits, report = compose({"train": [], "validation": [], "test": []}, registry, use="router")
    assert report["split_counts"] == {"train": 1, "validation": 1, "test": 0}
    assert foreign_use_rows(
        registry,
        [row for rows in splits.values() for row in rows],
        use="router",
        name_field="curriculum",
    ) == []
    unfiltered, _ = compose({"train": [], "validation": [], "test": []}, registry)
    assert len(unfiltered["train"]) == 2


def test_behaviour_rows_rejects_a_template_str_format_cannot_use() -> None:
    """A literal brace or a second placeholder otherwise surfaces as a bare
    KeyError from inside the expansion, naming neither seed nor string."""
    bad_frame = _seed(frames=("can you {s}", "what about {s} and {other}", "could you {s}"))
    with pytest.raises(ValueError, match="frame 1 is not a usable template"):
        behaviour_rows([bad_frame], "train", row_fn=_behaviour_row)
    bad_final = _seed(finals=("a stray { brace", "I cannot {s}.", "No: {s}."))
    with pytest.raises(ValueError, match="final 0 is not a usable template"):
        behaviour_rows([bad_final], "train", row_fn=_behaviour_row)


def test_behaviour_rows_checks_templates_the_split_would_never_reach() -> None:
    """Validation only expands the first two frames, but a defect in the third
    is still a defect in the seed."""
    seed = _seed(frames=("can you {s}", "i need you to {s}", "could you {s} and {other}"))
    with pytest.raises(ValueError, match="frame 2 is not a usable template"):
        behaviour_rows([seed], "validation", row_fn=_behaviour_row)


def test_behaviour_rows_rejects_an_empty_subject_list_for_the_split() -> None:
    seed = _seed(subjects={"train": ("book a flight",), "validation": ()})
    with pytest.raises(ValueError, match="empty subject list"):
        behaviour_rows([seed], "validation", row_fn=_behaviour_row)


def test_foreign_use_rows_raises_when_no_row_carries_the_name_field() -> None:
    """Unregistered names are skipped, so a typo'd `name_field` would make every
    row unjudgeable and return a clean, entirely meaningless empty list."""
    registry = _use_registry()
    rows = registry.build("train")
    with pytest.raises(ValueError, match="curiculum"):
        foreign_use_rows(registry, rows, use="router", name_field="curiculum")


def test_foreign_use_rows_accepts_an_empty_row_sequence() -> None:
    """An empty export carries no signal either way."""
    assert foreign_use_rows(_use_registry(), [], use="router", name_field="curriculum") == []
