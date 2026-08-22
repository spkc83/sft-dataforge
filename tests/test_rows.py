from __future__ import annotations

import pytest

from dataforge.rows import (
    DERIVED_FIELDS,
    canonical_json_bytes,
    make_row,
    normalize_text,
    normalize_text_ascii,
    rederive_text,
    render_context,
    validate_row_consistency,
)


def test_render_context_current_only() -> None:
    text, has_context = render_context("hello there")
    assert text == "[CURRENT_USER]\nhello there"
    assert has_context is False


def test_render_context_with_history_and_state() -> None:
    history = [
        {"role": "user", "content": "what's my balance"},
        {"role": "assistant", "content": "It's $100."},
    ]
    text, has_context = render_context(
        "freeze it",
        history,
        prior_state={"pending_servicing": "view_balance"},
    )
    assert has_context is True
    assert text.startswith("[PRIOR_STATE]\n")
    assert "[CURRENT_USER]\nfreeze it" in text
    assert "[PREVIOUS_ASSISTANT]\nIt's $100." in text
    assert "[PREVIOUS_USER]\nwhat's my balance" in text
    # most recent exchange rendered first
    assert text.index("[PREVIOUS_ASSISTANT]") < text.index("[PREVIOUS_USER]")


def test_render_context_meaningful_state_predicate_suppresses_empty_state() -> None:
    text, has_context = render_context(
        "hi",
        prior_state={"pending_servicing": None},
        is_meaningful_state=lambda state: bool(state and state.get("pending_servicing")),
    )
    assert has_context is False
    assert "[PRIOR_STATE]" not in text


def test_render_context_rejects_empty_current() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        render_context("   ")


def test_render_context_respects_max_exchanges() -> None:
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    text, _ = render_context("u3", history, max_exchanges=1)
    assert "u2" in text and "a2" in text
    assert "u1" not in text and "a1" not in text


def test_normalize_text_strips_punctuation_and_case() -> None:
    assert normalize_text("Freeze my Card, please!!") == "freeze my card please"


def test_make_row_multi_hot_relations() -> None:
    row = make_row(
        current="freeze it",
        labels={"intent": "freeze_card", "action_name": "clarify"},
        relation_dimension=("context_dependent", "topic_shift"),
        relation_names=["context_dependent"],
        example_kind="clarify",
        source="unit-test",
        source_split="train",
        group_id="g1",
    )
    assert row["relation_labels"] == [1, 0]
    assert row["group_id"] == "g1"
    assert row["trajectory_id"] == "g1"
    assert row["intent"] == "freeze_card"


def test_make_row_rejects_unknown_relation_name() -> None:
    with pytest.raises(ValueError, match="unknown relation"):
        make_row(
            current="hi",
            labels={},
            relation_dimension=("context_dependent",),
            relation_names=["not_a_relation"],
            example_kind="k",
            source="s",
            source_split="train",
            group_id="g1",
        )


def test_make_row_extra_fields_and_pair_metadata() -> None:
    row = make_row(
        current="hi",
        labels={},
        example_kind="k",
        source="s",
        source_split="train",
        group_id="g1",
        trajectory_id="t1",
        pair_id="p1",
        pair_target="card",
        pair_family="family-a",
        extra={"tool_names": ["list_accounts"]},
    )
    assert row["trajectory_id"] == "t1"
    assert row["pair_id"] == "p1"
    assert row["pair_target"] == "card"
    assert row["pair_family"] == "family-a"
    assert row["tool_names"] == ["list_accounts"]


def test_canonical_json_bytes_sorts_keys() -> None:
    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_derived_fields_declares_text_depends_on_current_text() -> None:
    assert "current_text" in DERIVED_FIELDS["text"]
    assert "history" in DERIVED_FIELDS["text"]
    assert "prior_state" in DERIVED_FIELDS["text"]


def test_rederive_text_resyncs_after_current_text_edit() -> None:
    row = make_row(
        current="what is the overdraft policy",
        labels={},
        example_kind="k",
        source="s",
        source_split="train",
        group_id="g1",
    )
    row["current_text"] = "how long is the overdraft grace period"
    rederive_text(row)
    assert row["text"] == "[CURRENT_USER]\nhow long is the overdraft grace period"


def test_rederive_text_respects_history_and_state() -> None:
    row = make_row(
        current="freeze it",
        history=[
            {"role": "user", "content": "what's my balance"},
            {"role": "assistant", "content": "It's $100."},
        ],
        prior_state={"pending_servicing": "view_balance"},
        labels={},
        example_kind="k",
        source="s",
        source_split="train",
        group_id="g1",
    )
    row["current_text"] = "cancel it instead"
    rederive_text(row)
    assert "[CURRENT_USER]\ncancel it instead" in row["text"]
    assert "[PREVIOUS_USER]\nwhat's my balance" in row["text"]
    assert row["has_context"] is True


def test_validate_row_consistency_passes_for_a_fresh_row() -> None:
    row = make_row(
        current="what is the overdraft policy",
        labels={},
        example_kind="k",
        source="s",
        source_split="train",
        group_id="g1",
    )
    validate_row_consistency(row)  # no raise


def test_validate_row_consistency_rejects_stale_text() -> None:
    row = make_row(
        current="what is the overdraft policy",
        labels={},
        example_kind="k",
        source="s",
        source_split="train",
        group_id="g1",
    )
    row["current_text"] = "a completely different question"  # text not re-rendered
    with pytest.raises(ValueError, match="inconsistent"):
        validate_row_consistency(row)


def test_normalize_text_ascii_collapses_punctuation_and_whitespace() -> None:
    assert normalize_text_ascii("  Freeze  my CARD, please!  ") == "freeze my card please"


def test_normalize_text_ascii_drops_non_ascii_letters_unlike_normalize_text() -> None:
    """`normalize_text_ascii` is the v9 predicate: `[^a-z0-9]+` -> space, so a
    non-ASCII letter is punctuation to it, while `normalize_text`'s Unicode
    `str.isalnum()` keeps it. The divergence is deliberate and documented."""
    assert normalize_text_ascii("Café Münster") == "caf m nster"
    assert normalize_text("Café Münster") == "café münster"


def test_normalize_text_ascii_of_blank_text_is_empty() -> None:
    assert normalize_text_ascii("   ...   ") == ""
