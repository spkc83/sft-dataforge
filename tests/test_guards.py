from __future__ import annotations

import re
from typing import Any

import pytest

from dataforge.guards import (
    banned_wording_leaks,
    contains_heldout_ngram,
    count_pii_matches,
    duplicate_text_leaks,
    heldout_leaks,
    is_heldout_text,
    leakage_report,
    paired_counterfactual_exemption,
    secondary_field_leaks,
    word_ngrams,
)


def test_count_pii_matches_detects_email_and_card() -> None:
    texts = ["contact me at a@b.com", "card 4111 1111 1111 1111", "no pii here"]
    assert count_pii_matches(texts) >= 2


def test_word_ngrams_size() -> None:
    grams = word_ngrams("the quick brown fox jumps", size=4)
    assert ("the", "quick", "brown", "fox") in grams
    assert len(grams) == 2


def test_is_heldout_text_normalizes() -> None:
    assert is_heldout_text("Freeze My Card!", ["freeze my card"])
    assert not is_heldout_text("unrelated text", ["freeze my card"])


def test_contains_heldout_ngram() -> None:
    held_out = ["why is my card declined at the airport lounge"]
    leaked = "so why is my card declined at the airport lounge today"
    assert contains_heldout_ngram(leaked, held_out)
    assert not contains_heldout_ngram("completely unrelated sentence here", held_out)


def test_heldout_leaks_excludes_home_split() -> None:
    splits = {
        "train": [{"text": "why is my card declined at the airport lounge", "group_id": "g1"}],
        "test": [{"text": "why is my card declined at the airport lounge", "group_id": "g2"}],
    }
    exact, ngram = heldout_leaks(
        splits,
        ["why is my card declined at the airport lounge"],
        excluded_splits=("test",),
    )
    assert len(exact) == 1
    assert exact[0]["split"] == "train"


def test_leakage_report_flags_group_and_pair_across_splits() -> None:
    splits = {
        "train": [{"group_id": "shared", "trajectory_id": "shared", "pair_id": "p1", "text": "a"}],
        "test": [{"group_id": "shared", "trajectory_id": "shared", "pair_id": "p1", "text": "b"}],
    }
    report = leakage_report(splits)
    assert report["group_split_leak_count"] == 1
    assert report["trajectory_split_leak_count"] == 1
    assert report["pair_split_leak_count"] == 1


def test_leakage_report_clean_when_disjoint() -> None:
    splits = {
        "train": [{"group_id": "g-train", "text": "a"}],
        "test": [{"group_id": "g-test", "text": "b"}],
    }
    report = leakage_report(splits)
    assert report["group_split_leak_count"] == 0
    assert report["heldout_exact_leaks"] == []
    assert report["heldout_ngram_leaks"] == []


def test_secondary_field_leaks_catches_reuse_under_distinct_groups() -> None:
    splits = {
        "train": [{"group_id": "state-family|train", "current_text": "explain that policy again"}],
        "test": [{"group_id": "state-family|test", "current_text": "explain that policy again"}],
    }
    leaks = secondary_field_leaks(splits, "current_text")
    assert len(leaks) == 1
    (values,) = leaks.values()
    assert values == ["test", "train"]


def test_secondary_field_leaks_ignores_missing_or_blank_values() -> None:
    splits = {
        "train": [{"group_id": "g1"}],  # no current_text field at all
        "test": [{"group_id": "g2", "current_text": "   "}],
    }
    assert secondary_field_leaks(splits, "current_text") == {}


def test_leakage_report_secondary_leak_fields_produces_gate_ready_keys() -> None:
    splits = {
        "train": [{"group_id": "g1", "current_text": "this is the same text"}],
        "test": [{"group_id": "g2", "current_text": "this is the same text"}],
    }
    report = leakage_report(splits, secondary_leak_fields=("current_text",))
    assert report["current_text_split_leak_count"] == 1
    assert report["group_split_leak_count"] == 0  # groups are genuinely distinct


def test_secondary_field_leaks_min_tokens_floor_ignores_short_reuse() -> None:
    """A bare one-word confirmation ("yes") legitimately recurs across
    splits under distinct multiturn contexts; the default 3-token floor
    must not flag it."""
    splits = {
        "train": [{"group_id": "a", "current_text": "yes"}],
        "test": [{"group_id": "b", "current_text": "yes"}],
    }
    assert secondary_field_leaks(splits, "current_text") == {}
    assert secondary_field_leaks(splits, "current_text", min_tokens=1) != {}


def test_secondary_field_leaks_min_tokens_is_configurable() -> None:
    splits = {
        "train": [{"group_id": "a", "current_text": "explain that policy again"}],
        "test": [{"group_id": "b", "current_text": "explain that policy again"}],
    }
    assert secondary_field_leaks(splits, "current_text", min_tokens=10) == {}
    assert secondary_field_leaks(splits, "current_text", min_tokens=3) != {}


def test_secondary_field_leaks_row_predicate_scopes_which_rows_count() -> None:
    splits = {
        "train": [
            {"group_id": "a", "current_text": "explain that policy again", "example_kind": "k1"}
        ],
        "test": [
            {"group_id": "b", "current_text": "explain that policy again", "example_kind": "k2"}
        ],
    }
    only_k1 = secondary_field_leaks(
        splits, "current_text", row_predicate=lambda row: row["example_kind"] == "k1"
    )
    assert only_k1 == {}  # the test-split row is filtered out, so no cross-split pair remains

    both = secondary_field_leaks(
        splits, "current_text", row_predicate=lambda row: True
    )
    assert both != {}


BANNED = re.compile(r"\b(shown in the app|tap the freeze button)\b")


def _message(role: str, content: str | None) -> dict[str, Any]:
    return {"role": role, "content": content}


def test_banned_wording_leaks_flags_trainable_text_with_first_match_as_term() -> None:
    splits = {
        "train": [
            {
                "group_id": "g1",
                "text": "you can tap the freeze button, it is shown in the app",
            }
        ],
    }
    result = banned_wording_leaks(splits, BANNED)
    assert result["banned_wording_leak_count"] == 1
    (entry,) = result["banned_wording_leaks"]
    assert entry == {
        "split": "train",
        "group_id": "g1",
        "field": "text",
        "term": "tap the freeze button",  # first match in the string, not every match
    }


def test_banned_wording_leaks_exempts_frozen_splits_by_construction() -> None:
    """A frozen (non-trainable) split may deliberately contain banned wording
    as held-out evaluation material; only `trainable_splits` are scanned."""
    splits = {
        "train": [{"group_id": "g1", "text": "clean wording"}],
        "test": [{"group_id": "g2", "text": "it is shown in the app"}],
    }
    result = banned_wording_leaks(splits, BANNED)
    assert result["banned_wording_leaks"] == []
    assert result["banned_wording_leak_count"] == 0


def test_banned_wording_leaks_accepts_a_string_pattern() -> None:
    splits = {"train": [{"group_id": "g1", "text": "it is shown in the app"}]}
    result = banned_wording_leaks(splits, r"shown in the app")
    assert result["banned_wording_leak_count"] == 1


def test_banned_wording_leaks_scans_message_fields_by_role() -> None:
    splits = {
        "train": [
            {
                "group_id": "g1",
                "messages": [
                    _message("system", "never say shown in the app"),  # system excluded
                    _message("user", "how do I freeze it"),
                    _message("assistant", None),  # tool-call assistant: content is None
                    _message("tool", "shown in the app"),  # tool role excluded
                    _message("assistant", "it is shown in the app"),
                ],
            }
        ],
    }
    result = banned_wording_leaks(splits, BANNED, text_fields=(), message_fields=("messages",))
    assert result["banned_wording_leak_count"] == 1
    (entry,) = result["banned_wording_leaks"]
    assert entry["field"] == "messages"
    assert entry["term"] == "shown in the app"


def test_banned_wording_leaks_message_roles_are_configurable() -> None:
    splits = {
        "train": [
            {"group_id": "g1", "messages": [_message("system", "shown in the app")]},
        ],
    }
    default_roles = banned_wording_leaks(
        splits, BANNED, text_fields=(), message_fields=("messages",)
    )
    assert default_roles["banned_wording_leak_count"] == 0
    with_system = banned_wording_leaks(
        splits,
        BANNED,
        text_fields=(),
        message_fields=("messages",),
        message_roles=("system",),
    )
    assert with_system["banned_wording_leak_count"] == 1


def test_banned_wording_leaks_emits_one_entry_per_scanned_string() -> None:
    splits = {
        "train": [
            {
                "group_id": "g1",
                "messages": [
                    _message("user", "shown in the app"),
                    _message("assistant", "tap the freeze button"),
                ],
            }
        ],
    }
    result = banned_wording_leaks(splits, BANNED, text_fields=(), message_fields=("messages",))
    assert result["banned_wording_leak_count"] == 2
    assert [entry["term"] for entry in result["banned_wording_leaks"]] == [
        "shown in the app",
        "tap the freeze button",
    ]


def test_duplicate_text_leaks_buckets_globally_across_and_within_splits() -> None:
    splits = {
        "train": [
            {"group_id": "g1", "user_text": "Freeze my card!"},
            {"group_id": "g2", "user_text": "freeze my card"},
        ],
        "validation": [{"group_id": "g3", "user_text": "freeze  my  card"}],
        "test": [{"group_id": "g4", "user_text": "something else entirely"}],
    }
    result = duplicate_text_leaks(splits, "user_text")
    assert result["user_text_duplicate_leak_count"] == 1
    (bucket,) = result["user_text_duplicate_leaks"]
    assert bucket["normalized"] == "freeze my card"
    assert bucket["members"] == [
        {"split": "train", "group_id": "g1"},
        {"split": "train", "group_id": "g2"},
        {"split": "validation", "group_id": "g3"},
    ]


def test_duplicate_text_leaks_splits_in_scope_limits_which_splits_are_bucketed() -> None:
    splits = {
        "train": [{"group_id": "g1", "user_text": "freeze my card"}],
        "test": [{"group_id": "g2", "user_text": "freeze my card"}],
    }
    assert duplicate_text_leaks(splits, "user_text")["user_text_duplicate_leak_count"] == 1
    scoped = duplicate_text_leaks(splits, "user_text", splits_in_scope=("train",))
    assert scoped["user_text_duplicate_leak_count"] == 0


def test_duplicate_text_leaks_passes_bucket_rows_carrying_their_split_to_exempt() -> None:
    splits = {
        "train": [{"group_id": "g1", "user_text": "freeze my card"}],
        "validation": [{"group_id": "g2", "user_text": "freeze my card"}],
    }
    seen: list[list[str]] = []

    def record(bucket_rows: object) -> bool:
        assert isinstance(bucket_rows, list)
        seen.append([str(row["_split"]) for row in bucket_rows])
        return True

    result = duplicate_text_leaks(splits, "user_text", exempt=record)
    assert result["user_text_duplicate_leak_count"] == 0
    assert seen == [["train", "validation"]]
    # the copies handed to `exempt` must not mutate the caller's rows
    assert "_split" not in splits["train"][0]


def test_duplicate_text_leaks_ignores_missing_and_blank_values() -> None:
    splits = {
        "train": [{"group_id": "g1"}, {"group_id": "g2", "user_text": "   "}],
        "test": [{"group_id": "g3", "user_text": ""}],
    }
    assert duplicate_text_leaks(splits, "user_text")["user_text_duplicate_leak_count"] == 0


def _pair_row(group_id: str, pair_id: str, target: str, history: object, split: str) -> dict:
    return {
        "group_id": group_id,
        "pair_id": pair_id,
        "pair_target": target,
        "history": history,
        "_split": split,
        "user_text": "freeze my card",
    }


def test_paired_counterfactual_exemption_accepts_two_well_formed_pairs() -> None:
    exempt = paired_counterfactual_exemption()
    bucket = [
        _pair_row("g1", "p1", "freeze_card", ["a"], "train"),
        _pair_row("g2", "p1", "list_cards", ["b"], "train"),
        _pair_row("g3", "p2", "freeze_card", ["c"], "train"),
        _pair_row("g4", "p2", "list_cards", ["d"], "train"),
    ]
    assert exempt(bucket) is True


def test_paired_counterfactual_exemption_rejects_an_odd_or_singleton_bucket() -> None:
    exempt = paired_counterfactual_exemption()
    assert exempt([_pair_row("g1", "p1", "freeze_card", ["a"], "train")]) is False
    assert (
        exempt(
            [
                _pair_row("g1", "p1", "freeze_card", ["a"], "train"),
                _pair_row("g2", "p1", "list_cards", ["b"], "train"),
                _pair_row("g3", "p2", "freeze_card", ["c"], "train"),
            ]
        )
        is False
    )


def test_paired_counterfactual_exemption_rejects_a_pair_id_without_exactly_two_rows() -> None:
    exempt = paired_counterfactual_exemption()
    bucket = [
        _pair_row("g1", "p1", "freeze_card", ["a"], "train"),
        _pair_row("g2", "p1", "list_cards", ["b"], "train"),
        _pair_row("g3", "p1", "block_card", ["c"], "train"),
        _pair_row("g4", "p2", "list_cards", ["d"], "train"),
    ]
    assert exempt(bucket) is False


def test_paired_counterfactual_exemption_rejects_identical_targets_within_a_pair() -> None:
    exempt = paired_counterfactual_exemption()
    bucket = [
        _pair_row("g1", "p1", "freeze_card", ["a"], "train"),
        _pair_row("g2", "p1", "freeze_card", ["b"], "train"),
    ]
    assert exempt(bucket) is False


def test_paired_counterfactual_exemption_rejects_a_bucket_spanning_two_splits() -> None:
    """A counterfactual pair is a within-split construction; the same text in
    two splits is leakage, never an exemptible pair."""
    exempt = paired_counterfactual_exemption()
    bucket = [
        _pair_row("g1", "p1", "freeze_card", ["a"], "train"),
        _pair_row("g2", "p1", "list_cards", ["b"], "test"),
    ]
    assert exempt(bucket) is False


def test_paired_counterfactual_exemption_rejects_a_missing_pair_id() -> None:
    exempt = paired_counterfactual_exemption()
    bucket = [
        {"group_id": "g1", "pair_target": "freeze_card", "history": ["a"], "_split": "train"},
        {"group_id": "g2", "pair_target": "list_cards", "history": ["b"], "_split": "train"},
    ]
    assert exempt(bucket) is False


def test_paired_counterfactual_exemption_rejects_pairs_sharing_a_context_signature() -> None:
    """Two pairs with the same context are a copy of one pair, not two
    distinct counterfactual constructions (v9's distinct-history rule)."""
    exempt = paired_counterfactual_exemption()
    bucket = [
        _pair_row("g1", "p1", "freeze_card", ["a"], "train"),
        _pair_row("g2", "p1", "list_cards", ["b"], "train"),
        _pair_row("g3", "p2", "freeze_card", ["a"], "train"),
        _pair_row("g4", "p2", "list_cards", ["b"], "train"),
    ]
    assert exempt(bucket) is False


def test_paired_counterfactual_exemption_fields_are_configurable() -> None:
    exempt = paired_counterfactual_exemption(
        pair_field="cf_id", target_field="cf_target", context_fields=("prior_state",)
    )
    bucket = [
        {"cf_id": "p1", "cf_target": "a", "prior_state": {"s": 1}, "_split": "train"},
        {"cf_id": "p1", "cf_target": "b", "prior_state": {"s": 2}, "_split": "train"},
    ]
    assert exempt(bucket) is True


def test_duplicate_text_leaks_with_paired_counterfactual_exemption_end_to_end() -> None:
    splits = {
        "train": [
            _pair_row("g1", "p1", "freeze_card", ["a"], "unused"),
            _pair_row("g2", "p1", "list_cards", ["b"], "unused"),
        ],
    }
    for row in splits["train"]:
        del row["_split"]  # duplicate_text_leaks stamps it on the copies it hands to exempt
    result = duplicate_text_leaks(
        splits, "user_text", exempt=paired_counterfactual_exemption()
    )
    assert result["user_text_duplicate_leak_count"] == 0


def test_secondary_field_leaks_raises_when_the_field_key_is_absent_from_every_row() -> None:
    """A typo in `secondary_leak_fields` must fail loudly rather than pass
    vacuously with an empty result."""
    splits = {
        "train": [{"group_id": "g1", "current_text": "a"}],
        "test": [{"group_id": "g2", "current_text": "b"}],
    }
    with pytest.raises(ValueError, match="curent_text"):
        secondary_field_leaks(splits, "curent_text")


def test_secondary_field_leaks_accepts_a_key_present_only_with_a_blank_value() -> None:
    """Key absence is the error, not 'no usable value': a present-but-blank
    value still proves the field name is real."""
    splits = {
        "train": [{"group_id": "g1"}],
        "test": [{"group_id": "g2", "current_text": "   "}],
    }
    assert secondary_field_leaks(splits, "current_text") == {}


def test_secondary_field_leaks_on_entirely_empty_splits_is_not_an_error() -> None:
    assert secondary_field_leaks({"train": [], "test": []}, "current_text") == {}
