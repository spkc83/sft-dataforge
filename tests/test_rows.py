from __future__ import annotations

import pytest

from dataforge.rows import (
    CONVERSATION_DERIVED_FIELDS,
    DERIVED_FIELDS,
    canonical_json_bytes,
    make_conversation_row,
    make_row,
    normalize_text,
    normalize_text_ascii,
    rederive_conversation,
    rederive_text,
    render_context,
    render_conversation,
    render_conversation_text,
    validate_conversation_messages,
    validate_conversation_row,
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


def _tool_turn(
    name: str = "freeze_card",
    arguments: dict | None = None,
    result: dict | None = None,
) -> dict:
    return {
        "name": name,
        "arguments": {"last4": "1792"} if arguments is None else arguments,
        "result": {"ok": True, "result": {"status": "frozen"}} if result is None else result,
    }


def _context_tool_pair(record_id: str = "r1", index: int = 0, name: str = "list_cards") -> list:
    call_id = f"context_{record_id}_{index}"
    return [
        {
            "role": "assistant",
            "content": None,
            "loss": False,
            "tool_calls": [
                {
                    "id": call_id,
                    "index": 0,
                    "type": "function",
                    "function": {"name": name, "arguments": {}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "loss": False,
            "content": {"ok": True, "result": {"cards": []}},
        },
    ]


def _conversation_row(**overrides: object) -> dict:
    kwargs: dict = {
        "record_id": "r1",
        "context_messages": [],
        "user_text": "freeze my card ending 1792",
        "action_turns": [_tool_turn()],
        "final_response": "Done. The card ending 1792 is frozen.",
        "labels": {"intent": "freeze_card", "action_name": "execute_tool"},
        "example_kind": "tool_success",
        "source": "unit-test",
        "source_split": "train",
        "group_id": "g1",
    }
    kwargs.update(overrides)
    return make_conversation_row(**kwargs)


def test_conversation_derived_fields_declares_the_three_derived_fields() -> None:
    assert CONVERSATION_DERIVED_FIELDS == {
        "messages": ("context_messages", "user_text", "action_turns", "final_response"),
        "text": ("context_messages", "user_text"),
        "expected_tool_calls": ("action_turns",),
    }


def test_make_conversation_row_builds_a_tool_calling_transcript() -> None:
    row = _conversation_row()
    assert row["record_id"] == "r1"
    assert row["expected_tool_calls"] == [{"name": "freeze_card", "arguments": {"last4": "1792"}}]
    assert [message["role"] for message in row["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert row["messages"][0] == {
        "role": "user",
        "content": "freeze my card ending 1792",
        "loss": False,
    }
    assert row["messages"][-1] == {
        "role": "assistant",
        "content": "Done. The card ending 1792 is frozen.",
        "loss": True,
    }
    assert row["text"] == "[CURRENT_USER]\nfreeze my card ending 1792"
    assert row["has_context"] is False
    assert row["intent"] == "freeze_card"
    assert row["trajectory_id"] == "g1"
    assert row["pair_id"] is None


def test_make_conversation_row_without_tools_has_only_user_and_final_turns() -> None:
    row = _conversation_row(action_turns=[])
    assert row["expected_tool_calls"] == []
    assert [message["role"] for message in row["messages"]] == ["user", "assistant"]


def test_make_conversation_row_deep_copies_its_inputs() -> None:
    context = [{"role": "user", "content": "hello", "loss": False}]
    turns = [_tool_turn()]
    row = _conversation_row(context_messages=context, action_turns=turns)
    context[0]["content"] = "mutated"
    turns[0]["arguments"]["last4"] = "0000"
    assert row["context_messages"][0]["content"] == "hello"
    assert row["action_turns"][0]["arguments"] == {"last4": "1792"}
    assert row["expected_tool_calls"][0]["arguments"] == {"last4": "1792"}


def test_make_conversation_row_extra_is_applied_last() -> None:
    row = _conversation_row(extra={"example_kind": "override", "policy_citations": ["p1"]})
    assert row["example_kind"] == "override"
    assert row["policy_citations"] == ["p1"]


def test_render_conversation_mints_deterministic_ids_and_loss_flags() -> None:
    messages = render_conversation(
        record_id="r9",
        context_messages=[{"role": "system", "content": "you are a bank agent", "loss": False}],
        user_text="freeze it",
        action_turns=[
            _tool_turn(name="list_cards", arguments={}, result={"ok": True, "result": {"n": 1}}),
            _tool_turn(result={"ok": False, "error": {"code": "not_found", "message": "no card"}}),
        ],
        final_response="All set.",
    )
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[2]["tool_calls"] == [
        {
            "id": "call_r9_0",
            "index": 0,
            "type": "function",
            "function": {"name": "list_cards", "arguments": {}},
        }
    ]
    assert messages[2]["content"] is None
    assert messages[2]["loss"] is True
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "call_r9_0",
        "name": "list_cards",
        "loss": False,
        "content": {"ok": True, "result": {"n": 1}},
    }
    assert messages[4]["tool_calls"][0]["id"] == "call_r9_1"
    # the error envelope is passed through verbatim
    assert messages[5]["content"] == {
        "ok": False,
        "error": {"code": "not_found", "message": "no card"},
    }
    assert messages[6] == {"role": "assistant", "content": "All set.", "loss": True}


def test_render_conversation_keeps_context_tool_call_ids_verbatim() -> None:
    context = _context_tool_pair("r1", 0)
    messages = render_conversation(
        record_id="r1",
        context_messages=context,
        user_text="and freeze it",
        action_turns=[_tool_turn()],
        final_response="Done.",
    )
    assert messages[0]["tool_calls"][0]["id"] == "context_r1_0"
    assert messages[1]["tool_call_id"] == "context_r1_0"
    assert messages[3]["tool_calls"][0]["id"] == "call_r1_0"


def test_render_conversation_text_includes_every_prior_exchange() -> None:
    context: list[dict] = []
    for index in range(5):
        context.append({"role": "user", "content": f"u{index}", "loss": False})
        context.append({"role": "assistant", "content": f"a{index}", "loss": False})
    text = render_conversation_text(context, "the newest question")
    assert text.startswith("[CURRENT_USER]\nthe newest question")
    for index in range(5):
        assert f"[PREVIOUS_USER]\nu{index}" in text
        assert f"[PREVIOUS_ASSISTANT]\na{index}" in text
    # oldest exchange survives the render -- there is no exchange cap
    assert "u0" in text and "a0" in text
    # most recent exchange first
    assert text.index("a4") < text.index("a0")


def test_render_conversation_text_excludes_system_tool_and_tool_call_turns() -> None:
    context = [
        {"role": "system", "content": "SYSTEM PROMPT", "loss": False},
        {"role": "user", "content": "which cards do I have", "loss": False},
        *_context_tool_pair("r1", 0),
        {"role": "assistant", "content": "You have one debit card.", "loss": False},
    ]
    text = render_conversation_text(context, "freeze it")
    assert "SYSTEM PROMPT" not in text
    assert "list_cards" not in text
    assert '"cards"' not in text
    assert "[PREVIOUS_USER]\nwhich cards do I have" in text
    assert "[PREVIOUS_ASSISTANT]\nYou have one debit card." in text


def test_make_conversation_row_sets_has_context_from_the_context_turns() -> None:
    row = _conversation_row(
        context_messages=[
            {"role": "user", "content": "which cards do I have", "loss": False},
            {"role": "assistant", "content": "One debit card.", "loss": False},
        ]
    )
    assert row["has_context"] is True
    assert "[PREVIOUS_USER]\nwhich cards do I have" in row["text"]


def test_make_conversation_row_system_only_context_is_not_context() -> None:
    row = _conversation_row(
        context_messages=[{"role": "system", "content": "you are an agent", "loss": False}]
    )
    assert row["has_context"] is False


def test_rederive_conversation_resyncs_messages_after_a_final_response_edit() -> None:
    row = _conversation_row()
    text_before = row["text"]
    expected_before = list(row["expected_tool_calls"])
    row["final_response"] = "Your card is frozen; nothing new can be charged."
    rederive_conversation(row)
    assert row["messages"][-1]["content"] == "Your card is frozen; nothing new can be charged."
    assert row["messages"][0]["content"] == "freeze my card ending 1792"
    assert row["text"] == text_before
    assert row["expected_tool_calls"] == expected_before
    validate_conversation_row(row)


def test_rederive_conversation_resyncs_text_and_calls_after_source_edits() -> None:
    row = _conversation_row()
    row["user_text"] = "please freeze the card ending 1792"
    row["action_turns"][0]["arguments"]["last4"] = "3600"
    rederive_conversation(row)
    assert row["text"] == "[CURRENT_USER]\nplease freeze the card ending 1792"
    assert row["messages"][0]["content"] == "please freeze the card ending 1792"
    assert row["expected_tool_calls"] == [{"name": "freeze_card", "arguments": {"last4": "3600"}}]
    validate_conversation_row(row)


def test_validate_conversation_row_rejects_stale_messages() -> None:
    row = _conversation_row()
    row["final_response"] = "a different closing line"  # messages not re-rendered
    with pytest.raises(ValueError, match="messages does not match its rendered"):
        validate_conversation_row(row)


def test_validate_conversation_row_rejects_stale_text() -> None:
    row = _conversation_row()
    row["text"] = "[CURRENT_USER]\nsomething else entirely"
    with pytest.raises(ValueError, match="text does not match its rendered"):
        validate_conversation_row(row)


def test_validate_conversation_row_rejects_stale_expected_tool_calls() -> None:
    row = _conversation_row()
    row["expected_tool_calls"] = [{"name": "replace_card", "arguments": {"last4": "1792"}}]
    with pytest.raises(ValueError, match="expected_tool_calls does not match its rendered"):
        validate_conversation_row(row)


def test_validate_conversation_row_rejects_stale_has_context() -> None:
    row = _conversation_row()
    row["has_context"] = True
    with pytest.raises(ValueError, match="has_context does not match its rendered"):
        validate_conversation_row(row)


_BAD_CONTEXT_CASES: list[tuple[str, list, str]] = [
    (
        "unknown role",
        [{"role": "narrator", "content": "hi", "loss": False}],
        "has invalid role",
    ),
    (
        "trainable user turn",
        [{"role": "user", "content": "hi", "loss": True}],
        "context message is labeled",
    ),
    (
        "trainable system turn",
        [{"role": "system", "content": "hi", "loss": True}],
        "context message is labeled",
    ),
    (
        "tool-call assistant with content",
        [
            {
                "role": "assistant",
                "content": "calling a tool",
                "loss": False,
                "tool_calls": [
                    {
                        "id": "context_r1_0",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "list_cards", "arguments": {}},
                    }
                ],
            }
        ],
        "tool-call assistant has content",
    ),
    (
        "two calls in one assistant message",
        [
            {
                "role": "assistant",
                "content": None,
                "loss": False,
                "tool_calls": [
                    {
                        "id": "context_r1_0",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "list_cards", "arguments": {}},
                    },
                    {
                        "id": "context_r1_1",
                        "index": 1,
                        "type": "function",
                        "function": {"name": "list_accounts", "arguments": {}},
                    },
                ],
            }
        ],
        "must contain exactly one call",
    ),
    (
        "non-boolean loss on a tool-call assistant",
        [
            {
                "role": "assistant",
                "content": None,
                "loss": "no",
                "tool_calls": [
                    {
                        "id": "context_r1_0",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "list_cards", "arguments": {}},
                    }
                ],
            }
        ],
        "assistant call has invalid loss label",
    ),
    (
        "call index not restarted",
        [
            {
                "role": "assistant",
                "content": None,
                "loss": False,
                "tool_calls": [
                    {
                        "id": "context_r1_0",
                        "index": 1,
                        "type": "function",
                        "function": {"name": "list_cards", "arguments": {}},
                    }
                ],
            }
        ],
        "tool call index must restart at zero",
    ),
    (
        "unstable context call id",
        [
            {
                "role": "assistant",
                "content": None,
                "loss": False,
                "tool_calls": [
                    {
                        "id": "context_r1_7",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "list_cards", "arguments": {}},
                    }
                ],
            }
        ],
        "has unstable context tool call id",
    ),
    (
        "duplicate call id",
        [*_context_tool_pair("r1", 0), *_context_tool_pair("r1", 0)],
        "has duplicate tool call id",
    ),
    (
        "arguments not an object",
        [
            {
                "role": "assistant",
                "content": None,
                "loss": False,
                "tool_calls": [
                    {
                        "id": "context_r1_0",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "list_cards", "arguments": '{"last4": "1792"}'},
                    }
                ],
            }
        ],
        "tool arguments must be object",
    ),
    (
        "labeled tool result",
        [
            _context_tool_pair("r1", 0)[0],
            {**_context_tool_pair("r1", 0)[1], "loss": True},
        ],
        "tool result is labeled",
    ),
    (
        "tool content not an object",
        [
            _context_tool_pair("r1", 0)[0],
            {**_context_tool_pair("r1", 0)[1], "content": "ok"},
        ],
        "tool content must be object",
    ),
    (
        "success envelope with an extra key",
        [
            _context_tool_pair("r1", 0)[0],
            {
                **_context_tool_pair("r1", 0)[1],
                "content": {"ok": True, "result": {}, "latency_ms": 3},
            },
        ],
        "success envelope is invalid",
    ),
    (
        "error envelope with the wrong keys",
        [
            _context_tool_pair("r1", 0)[0],
            {**_context_tool_pair("r1", 0)[1], "content": {"ok": False, "result": {}}},
        ],
        "error envelope is invalid",
    ),
    (
        "envelope without ok",
        [
            _context_tool_pair("r1", 0)[0],
            {**_context_tool_pair("r1", 0)[1], "content": {"result": {}}},
        ],
        "tool envelope missing ok",
    ),
    (
        "tool result correlated to the wrong call",
        [
            _context_tool_pair("r1", 0)[0],
            {**_context_tool_pair("r1", 0)[1], "tool_call_id": "context_r1_3"},
        ],
        "tool result correlation mismatch",
    ),
    (
        "call with no tool result at all",
        [_context_tool_pair("r1", 0)[0]],
        "tool result correlation mismatch",
    ),
    (
        "second call while one is pending",
        [_context_tool_pair("r1", 0)[0], *_context_tool_pair("r1", 1)],
        "tool result correlation mismatch",
    ),
]


@pytest.mark.parametrize(
    ("context_messages", "expected_message"),
    [(case[1], case[2]) for case in _BAD_CONTEXT_CASES],
    ids=[case[0] for case in _BAD_CONTEXT_CASES],
)
def test_make_conversation_row_rejects_malformed_context(
    context_messages: list, expected_message: str
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        _conversation_row(context_messages=context_messages)


def test_validate_conversation_row_rejects_an_unknown_tool() -> None:
    row = _conversation_row()
    with pytest.raises(ValueError, match="uses unknown tool 'freeze_card'"):
        validate_conversation_row(row, tool_arguments={"list_cards": set()})


def test_validate_conversation_row_rejects_unsupported_arguments() -> None:
    row = _conversation_row()
    with pytest.raises(ValueError, match=r"unsupported arguments for freeze_card: \['last4'\]"):
        validate_conversation_row(row, tool_arguments={"freeze_card": set()})


def test_validate_conversation_row_accepts_a_whitelisted_call() -> None:
    row = _conversation_row()
    validate_conversation_row(row, tool_arguments={"freeze_card": {"last4"}})  # no raise


def test_validate_conversation_messages_rejects_an_unstable_trainable_call_id() -> None:
    messages = render_conversation(
        record_id="r1",
        context_messages=[],
        user_text="freeze it",
        action_turns=[_tool_turn()],
        final_response="Done.",
    )
    messages[1]["tool_calls"][0]["id"] = "call_r1_7"
    messages[2]["tool_call_id"] = "call_r1_7"
    with pytest.raises(ValueError, match="has unstable tool call id"):
        validate_conversation_messages("r1", messages)


def test_validate_conversation_messages_rejects_an_untrainable_final_response() -> None:
    messages = render_conversation(
        record_id="r1",
        context_messages=[],
        user_text="freeze it",
        action_turns=[],
        final_response="Done.",
    )
    messages[-1]["loss"] = False
    with pytest.raises(ValueError, match="final assistant response must be trainable"):
        validate_conversation_messages("r1", messages)


def test_validate_conversation_messages_rejects_a_transcript_not_ending_in_the_final() -> None:
    messages = render_conversation(
        record_id="r1",
        context_messages=[],
        user_text="freeze it",
        action_turns=[_tool_turn()],
        final_response="Done.",
    )
    with pytest.raises(ValueError, match="final assistant response must be trainable"):
        validate_conversation_messages("r1", messages[:-1])


def test_validate_conversation_messages_rejects_a_final_that_is_not_the_final_response() -> None:
    messages = render_conversation(
        record_id="r1",
        context_messages=[],
        user_text="freeze it",
        action_turns=[],
        final_response="Done.",
    )
    with pytest.raises(ValueError, match="final assistant response must be trainable"):
        validate_conversation_messages("r1", messages, final_response="Something else.")


def test_validate_conversation_messages_rejects_calls_that_disagree_with_expected() -> None:
    messages = render_conversation(
        record_id="r1",
        context_messages=[],
        user_text="freeze it",
        action_turns=[_tool_turn()],
        final_response="Done.",
    )
    with pytest.raises(ValueError, match="expected tool_calls mismatch"):
        validate_conversation_messages(
            "r1",
            messages,
            expected_tool_calls=[{"name": "replace_card", "arguments": {"last4": "1792"}}],
        )


def test_validate_conversation_messages_ignores_context_calls_in_expected() -> None:
    row = _conversation_row(context_messages=_context_tool_pair("r1", 0))
    assert row["expected_tool_calls"] == [{"name": "freeze_card", "arguments": {"last4": "1792"}}]
    validate_conversation_row(row)  # no raise
