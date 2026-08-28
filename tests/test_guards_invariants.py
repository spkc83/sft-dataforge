"""The build-time invariant guards: fuzzy duplicates, probe exclusion, field
invariants and unsupported action claims.

Kept apart from ``test_guards.py`` (the leakage and duplicate guards) because
these four answer a different question: not "did material cross a split
boundary" but "is this row allowed to exist at all".
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from dataforge.curricula import Registry, compose
from dataforge.guards import (
    banned_patterns,
    field_invariant_leaks,
    forbidden_terms,
    fuzzy_duplicate_leaks,
    min_word_count,
    no_digits,
    no_questions,
    probe_exclusion_leaks,
    required_markers,
    unsupported_claim_leaks,
)

CARD_4821 = "please freeze my travel debit card ending in four eight two one"
CARD_4822 = "please freeze my travel debit card ending in four eight two two"

#: A pair that is very close (ratio ~0.9885) but not close enough for the
#: default threshold -- the fixture behind "the default is deliberately strict".
LONG_BASE = "the account holder asked about the pending charge and the freeze on the card " * 5
LONG_TODAY = LONG_BASE + "today"
LONG_TOMORROW = LONG_BASE + "tomorrow"


def _row(text: str, **extra: Any) -> dict[str, Any]:
    return {"user_text": text, **extra}


def test_fuzzy_duplicate_leaks_flags_a_near_duplicate_pair() -> None:
    splits = {"train": [_row(CARD_4821), _row(CARD_4822)]}
    result = fuzzy_duplicate_leaks(splits, field="user_text", threshold=0.9)
    (entry,) = result["fuzzy_duplicate_leaks"]
    assert entry == {
        "split": "train",
        "group": "",
        "index_a": 0,
        "index_b": 1,
        "ratio": 0.9683,
    }


def test_fuzzy_duplicate_leaks_ignores_an_exact_duplicate() -> None:
    """Exact duplicates are ``duplicate_text_leaks``' job; reporting them here
    too would mean two checks arguing about one bucket."""
    splits = {"train": [_row("Freeze my card!"), _row("freeze  my  card")]}
    assert fuzzy_duplicate_leaks(splits, field="user_text", threshold=0.5) == {
        "fuzzy_duplicate_leaks": []
    }


def test_fuzzy_duplicate_leaks_leaves_a_below_threshold_pair_alone() -> None:
    splits = {"train": [_row(CARD_4821), _row("what is the balance on my savings account")]}
    assert fuzzy_duplicate_leaks(splits, field="user_text", threshold=0.9) == {
        "fuzzy_duplicate_leaks": []
    }


def test_fuzzy_duplicate_leaks_default_threshold_is_deliberately_strict() -> None:
    splits = {"train": [_row(LONG_TODAY), _row(LONG_TOMORROW)]}
    assert fuzzy_duplicate_leaks(splits, field="user_text")["fuzzy_duplicate_leaks"] == []
    loosened = fuzzy_duplicate_leaks(splits, field="user_text", threshold=0.98)
    assert len(loosened["fuzzy_duplicate_leaks"]) == 1


def test_fuzzy_duplicate_leaks_group_fn_scopes_which_pairs_are_compared() -> None:
    splits = {
        "train": [
            _row(CARD_4821, family="travel"),
            _row(CARD_4822, family="everyday"),
        ]
    }
    ungrouped = fuzzy_duplicate_leaks(splits, field="user_text", threshold=0.9)
    assert len(ungrouped["fuzzy_duplicate_leaks"]) == 1
    grouped = fuzzy_duplicate_leaks(
        splits, field="user_text", threshold=0.9, group_fn=lambda row: row["family"]
    )
    assert grouped["fuzzy_duplicate_leaks"] == []


def test_fuzzy_duplicate_leaks_reports_the_group_key_it_compared_within() -> None:
    splits = {"train": [_row(CARD_4821, family="travel"), _row(CARD_4822, family="travel")]}
    result = fuzzy_duplicate_leaks(
        splits, field="user_text", threshold=0.9, group_fn=lambda row: row["family"]
    )
    assert result["fuzzy_duplicate_leaks"][0]["group"] == "travel"


def test_fuzzy_duplicate_leaks_never_pairs_rows_from_two_splits() -> None:
    splits = {"train": [_row(CARD_4821)], "validation": [_row(CARD_4822)]}
    assert fuzzy_duplicate_leaks(splits, field="user_text", threshold=0.9) == {
        "fuzzy_duplicate_leaks": []
    }


def test_fuzzy_duplicate_leaks_skips_splits_outside_splits_checked() -> None:
    splits = {"test": [_row(CARD_4821), _row(CARD_4822)]}
    assert fuzzy_duplicate_leaks(splits, field="user_text", threshold=0.9) == {
        "fuzzy_duplicate_leaks": []
    }
    checked = fuzzy_duplicate_leaks(
        splits, field="user_text", threshold=0.9, splits_checked=("test",)
    )
    assert len(checked["fuzzy_duplicate_leaks"]) == 1


def test_fuzzy_duplicate_leaks_skips_missing_and_blank_values_and_still_pairs_the_rest() -> None:
    """The skip has to be load bearing, so the split also holds a genuine
    near-duplicate pair: an unskipped ``None`` would fail on comparison rather
    than pass, and indexes that were positions in the filtered subset rather
    than in the split would read (0, 1)."""
    splits = {"train": [{"group_id": "g1"}, _row("   "), _row(CARD_4821), _row(CARD_4822)]}
    result = fuzzy_duplicate_leaks(splits, field="user_text", threshold=0.9)
    (entry,) = result["fuzzy_duplicate_leaks"]
    assert (entry["index_a"], entry["index_b"]) == (2, 3)


def test_fuzzy_duplicate_leaks_rejects_an_out_of_range_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        fuzzy_duplicate_leaks({"train": []}, field="user_text", threshold=1.5)


PROBE = "why does my frozen card still have a pending charge on it"
FRAGMENT = "still have a pending charge"


def test_probe_exclusion_leaks_flags_an_exact_probe_in_training() -> None:
    splits = {"train": [_row("Why does my frozen card still have a pending charge on it?")]}
    result = probe_exclusion_leaks(splits, probes=[PROBE], fields=("user_text",))
    (entry,) = result["probe_exclusion_leaks"]
    assert entry == {
        "split": "train",
        "index": 0,
        "field": "user_text",
        "kind": "probe",
        "value": PROBE,
    }


def test_probe_exclusion_leaks_flags_a_fragment_inside_a_longer_utterance() -> None:
    splits = {
        "validation": [_row("i still have a pending charge showing and i want to know why")]
    }
    result = probe_exclusion_leaks(splits, fragments=[FRAGMENT], fields=("user_text",))
    (entry,) = result["probe_exclusion_leaks"]
    assert entry["kind"] == "fragment"
    assert entry["value"] == FRAGMENT


def test_probe_exclusion_leaks_is_insensitive_to_casing_and_spacing() -> None:
    splits = {"train": [_row("  WHY   does my Frozen card   STILL have a pending charge on it ")]}
    result = probe_exclusion_leaks(splits, probes=[PROBE], fields=("user_text",))
    assert len(result["probe_exclusion_leaks"]) == 1


def test_probe_exclusion_leaks_is_empty_on_a_clean_corpus() -> None:
    splits = {"train": [_row("please freeze my travel debit card")]}
    result = probe_exclusion_leaks(
        splits, probes=[PROBE], fragments=[FRAGMENT], fields=("user_text",)
    )
    assert result == {"probe_exclusion_leaks": []}


def test_probe_exclusion_leaks_leaves_the_probes_own_split_alone() -> None:
    """The probes live in the frozen split; scanning it would flag every probe
    against itself. Exemption by construction, as ``banned_wording_leaks`` does."""
    splits = {"test": [_row(PROBE)]}
    assert probe_exclusion_leaks(splits, probes=[PROBE], fields=("user_text",)) == {
        "probe_exclusion_leaks": []
    }


def test_probe_exclusion_leaks_scans_every_named_field() -> None:
    splits = {
        "train": [
            {
                "user_text": "freeze the travel debit one",
                "text": "[PREVIOUS_USER] i still have a pending charge on the account",
            }
        ]
    }
    result = probe_exclusion_leaks(
        splits, fragments=[FRAGMENT], fields=("user_text", "text")
    )
    assert [entry["field"] for entry in result["probe_exclusion_leaks"]] == ["text"]


def test_probe_exclusion_leaks_refuses_to_run_with_nothing_to_look_for() -> None:
    with pytest.raises(ValueError, match="vacuously"):
        probe_exclusion_leaks({"train": [_row("anything")]})


def test_probe_exclusion_leaks_reports_a_whole_probe_once_not_also_its_fragment() -> None:
    """The fragments of a matched probe are inside it by construction, so
    reporting both would be one leak counted twice."""
    splits = {"train": [_row(PROBE)]}
    result = probe_exclusion_leaks(
        splits, probes=[PROBE], fragments=[FRAGMENT], fields=("user_text",)
    )
    (entry,) = result["probe_exclusion_leaks"]
    assert entry["kind"] == "probe"


def test_probe_exclusion_leaks_raises_when_no_named_field_exists_on_any_row() -> None:
    """A field-name typo is the other way this gate passes vacuously."""
    splits = {"train": [_row(PROBE)]}
    with pytest.raises(ValueError, match="usr_text"):
        probe_exclusion_leaks(splits, probes=[PROBE], fields=("usr_text",))
    with pytest.raises(ValueError, match="none of the fields"):
        probe_exclusion_leaks(splits, probes=[PROBE], fields=("usr_text", "txet"))


def test_probe_exclusion_leaks_field_check_is_silent_when_no_row_is_checked() -> None:
    """An empty corpus carries no signal either way; only a corpus that has
    rows and none of the fields is evidence of a typo."""
    splits = {"train": [], "test": [_row(PROBE)]}
    assert probe_exclusion_leaks(splits, probes=[PROBE], fields=("no_such_field",)) == {
        "probe_exclusion_leaks": []
    }


def _final(text: str, **extra: Any) -> dict[str, Any]:
    return {"final_response": text, **extra}


def test_no_digits_fires_on_a_digit_and_passes_otherwise() -> None:
    splits = {"train": [_final("your card ending in 4821 is frozen"), _final("that is frozen")]}
    result = field_invariant_leaks(splits, field="final_response", invariants=[no_digits()])
    (entry,) = result["field_invariant_leaks"]
    assert entry["split"] == "train"
    assert entry["index"] == 0
    assert entry["invariant"] == "no_digits"
    assert "4" in entry["detail"]


def test_no_questions_fires_on_a_question_mark() -> None:
    splits = {"train": [_final("which card should I freeze?"), _final("that card is frozen")]}
    result = field_invariant_leaks(splits, field="final_response", invariants=[no_questions()])
    assert [entry["index"] for entry in result["field_invariant_leaks"]] == [0]


def test_banned_patterns_reports_under_its_label() -> None:
    invariant = banned_patterns(
        [r"\bI have (?:frozen|closed)\b", re.compile(r"\ball set\b")],
        label="no_completed_action_claim",
    )
    splits = {"train": [_final("I have frozen it"), _final("you are all set"), _final("fine")]}
    result = field_invariant_leaks(splits, field="final_response", invariants=[invariant])
    entries = result["field_invariant_leaks"]
    assert [entry["index"] for entry in entries] == [0, 1]
    assert {entry["invariant"] for entry in entries} == {"no_completed_action_claim"}


def test_forbidden_terms_matches_case_insensitively_as_a_substring() -> None:
    invariant = forbidden_terms(["freeze_card", "list_cards"], label="no_tool_names")
    splits = {"train": [_final("I called Freeze_Card for you."), _final("the card is frozen")]}
    result = field_invariant_leaks(splits, field="final_response", invariants=[invariant])
    (entry,) = result["field_invariant_leaks"]
    assert entry["invariant"] == "no_tool_names"
    assert "freeze_card" in entry["detail"]


def test_required_markers_are_keyed_by_the_rows_tag() -> None:
    invariant = required_markers(
        {"refusal": ("i cannot",)}, tag_fn=lambda row: str(row.get("example_kind", ""))
    )
    splits = {
        "train": [
            _final("I am afraid that is not possible.", example_kind="refusal"),
            _final("I cannot do that from here.", example_kind="refusal"),
            _final("Anything at all.", example_kind="execute"),  # untagged kinds pass
        ]
    }
    result = field_invariant_leaks(splits, field="final_response", invariants=[invariant])
    (entry,) = result["field_invariant_leaks"]
    assert entry["index"] == 0
    assert entry["invariant"] == "required_markers"


def test_required_markers_rejects_an_invariant_that_could_never_fire() -> None:
    """An empty map, or a tag whose markers all normalize away, means every row
    silently satisfies a requirement nobody ever stated."""
    with pytest.raises(ValueError, match="at least one tag"):
        required_markers({}, tag_fn=lambda row: "refusal")
    with pytest.raises(ValueError, match="no marker left after normalization"):
        required_markers({"refusal": ("?!", "  ")}, tag_fn=lambda row: "refusal")


def test_min_word_count_fires_on_a_too_short_value() -> None:
    splits = {"train": [_final("Done."), _final("That card is frozen and nothing else changed.")]}
    result = field_invariant_leaks(splits, field="final_response", invariants=[min_word_count(5)])
    (entry,) = result["field_invariant_leaks"]
    assert entry["index"] == 0
    assert entry["detail"] == "has 1 words, needs 5"


def test_field_invariant_leaks_runs_every_invariant_over_every_row() -> None:
    splits = {"train": [_final("is card 4821 frozen?")]}
    result = field_invariant_leaks(
        splits, field="final_response", invariants=[no_digits(), no_questions()]
    )
    assert [entry["invariant"] for entry in result["field_invariant_leaks"]] == [
        "no_digits",
        "no_questions",
    ]


def test_field_invariant_leaks_row_predicate_scopes_which_rows_are_checked() -> None:
    splits = {
        "train": [
            _final("your card ending in 4821 is frozen", curriculum="tools"),
            _final("nothing has changed on the account", curriculum="behaviour"),
        ]
    }
    scoped = field_invariant_leaks(
        splits,
        field="final_response",
        invariants=[no_digits()],
        row_predicate=lambda row: row["curriculum"] == "behaviour",
    )
    assert scoped["field_invariant_leaks"] == []
    assert field_invariant_leaks(splits, field="final_response", invariants=[no_digits()])[
        "field_invariant_leaks"
    ]


def test_field_invariant_leaks_skips_splits_outside_splits_checked() -> None:
    splits = {"test": [_final("card 4821 is frozen")]}
    assert field_invariant_leaks(splits, field="final_response", invariants=[no_digits()]) == {
        "field_invariant_leaks": []
    }


def test_field_invariant_leaks_requires_at_least_one_invariant() -> None:
    with pytest.raises(ValueError, match="at least one invariant"):
        field_invariant_leaks({"train": [_final("a")]}, field="final_response", invariants=[])


def test_field_invariant_leaks_raises_when_the_field_is_absent_from_every_row() -> None:
    """A typo'd field name would otherwise disarm the gate and still report zero."""
    with pytest.raises(ValueError, match="fnial_response"):
        field_invariant_leaks(
            {"train": [_final("a")]}, field="fnial_response", invariants=[no_digits()]
        )


def test_invariant_factories_reject_a_blank_label_or_an_empty_rule_set() -> None:
    with pytest.raises(ValueError, match="label"):
        banned_patterns([r"x"], label="  ")
    with pytest.raises(ValueError, match="at least one pattern"):
        banned_patterns([], label="empty")
    with pytest.raises(ValueError, match="at least one term"):
        forbidden_terms([], label="empty")
    with pytest.raises(ValueError, match="at least 1"):
        min_word_count(0)


def _curriculum_row(text: str, group_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "text": text,
        "current_text": text,
        "final_response": text,
        "group_id": group_id,
        "trajectory_id": group_id,
        "pair_id": None,
        "example_kind": "unit",
        **extra,
    }


def test_a_broken_invariant_fails_the_build_through_pre_dedup_checks() -> None:
    """The wiring the invariants exist for: a violating row must fail the build
    before dedup gets the chance to drop it."""
    registry = Registry()

    @registry.register("behaviour", splits=("train",))
    def behaviour(split: str) -> list[dict[str, Any]]:
        return [_curriculum_row("your card ending in 4821 is frozen", "g1")]

    def invariant_check(splits: Any) -> Any:
        return field_invariant_leaks(splits, field="final_response", invariants=[no_digits()])

    with pytest.raises(ValueError, match="^pre-dedup check failed:") as error:
        compose(
            {"train": [], "validation": [], "test": []},
            registry,
            pre_dedup_checks=[invariant_check],
        )
    assert "field_invariant_leaks" in str(error.value)
    assert "no_digits" in str(error.value)


CLAIM_PATTERNS = (r"\bI have (?:frozen|blocked)\b", r"\bI checked\b")


def _claim_row(final: str, *, action_turns: list[Any]) -> dict[str, Any]:
    return {"final_response": final, "action_turns": action_turns}


def _has_tool_evidence(row: Any) -> bool:
    return bool(row.get("action_turns"))


def test_unsupported_claim_leaks_flags_a_claim_with_no_tool_evidence() -> None:
    splits = {"train": [_claim_row("I have frozen the card for you.", action_turns=[])]}
    result = unsupported_claim_leaks(
        splits,
        field="final_response",
        claim_patterns=CLAIM_PATTERNS,
        evidence_fn=_has_tool_evidence,
    )
    (entry,) = result["unsupported_claim_leaks"]
    assert entry == {"split": "train", "index": 0, "pattern": CLAIM_PATTERNS[0]}


def test_unsupported_claim_leaks_leaves_a_claim_backed_by_evidence_alone() -> None:
    splits = {
        "train": [_claim_row("I have frozen the card for you.", action_turns=[{"name": "freeze"}])]
    }
    result = unsupported_claim_leaks(
        splits,
        field="final_response",
        claim_patterns=CLAIM_PATTERNS,
        evidence_fn=_has_tool_evidence,
    )
    assert result == {"unsupported_claim_leaks": []}


def test_unsupported_claim_leaks_ignores_a_row_that_claims_nothing() -> None:
    splits = {"train": [_claim_row("Which card should I freeze first?", action_turns=[])]}
    result = unsupported_claim_leaks(
        splits,
        field="final_response",
        claim_patterns=CLAIM_PATTERNS,
        evidence_fn=_has_tool_evidence,
    )
    assert result == {"unsupported_claim_leaks": []}


def test_unsupported_claim_leaks_reports_every_matching_pattern() -> None:
    splits = {"train": [_claim_row("I checked, and I have frozen it.", action_turns=[])]}
    result = unsupported_claim_leaks(
        splits,
        field="final_response",
        claim_patterns=CLAIM_PATTERNS,
        evidence_fn=_has_tool_evidence,
    )
    assert [entry["pattern"] for entry in result["unsupported_claim_leaks"]] == list(
        CLAIM_PATTERNS
    )


def test_unsupported_claim_leaks_row_predicate_and_splits_checked_scope_the_pass() -> None:
    splits = {
        "train": [_claim_row("I have frozen it.", action_turns=[])],
        "test": [_claim_row("I have frozen it.", action_turns=[])],
    }
    scoped = unsupported_claim_leaks(
        splits,
        field="final_response",
        claim_patterns=CLAIM_PATTERNS,
        evidence_fn=_has_tool_evidence,
        row_predicate=lambda row: False,
    )
    assert scoped == {"unsupported_claim_leaks": []}
    default_scope = unsupported_claim_leaks(
        splits,
        field="final_response",
        claim_patterns=CLAIM_PATTERNS,
        evidence_fn=_has_tool_evidence,
    )
    assert [entry["split"] for entry in default_scope["unsupported_claim_leaks"]] == ["train"]


def test_unsupported_claim_leaks_requires_at_least_one_claim_pattern() -> None:
    with pytest.raises(ValueError, match="at least one claim pattern"):
        unsupported_claim_leaks(
            {"train": []},
            field="final_response",
            claim_patterns=(),
            evidence_fn=_has_tool_evidence,
        )
