from __future__ import annotations

from dataforge.guards import (
    contains_heldout_ngram,
    count_pii_matches,
    heldout_leaks,
    is_heldout_text,
    leakage_report,
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
        "train": [{"group_id": "g1", "current_text": "same text"}],
        "test": [{"group_id": "g2", "current_text": "same text"}],
    }
    report = leakage_report(splits, secondary_leak_fields=("current_text",))
    assert report["current_text_split_leak_count"] == 1
    assert report["group_split_leak_count"] == 0  # groups are genuinely distinct
