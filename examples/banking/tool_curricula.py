"""Curricula for the tool-calling worked example (``freeze_card``/``list_cards``).

Deliberately separate from :mod:`examples.banking.curricula`: those are flat
classifier rows (:func:`dataforge.rows.make_row`), these are conversation rows
(:func:`dataforge.rows.make_conversation_row`) whose hashed surface is a whole
transcript -- context turns, the current user turn, the tool calls and their
result envelopes -- and whose only editable text is the final assistant turn.
:mod:`examples.banking.build_tool_calls` composes them into their own dataset;
the classifier example is untouched.

The rows here are chosen to exercise the v9 mechanisms rather than to cover a
domain: a multi-turn row whose context contains its own tool-call pair, a
governed counterfactual pair that shares one utterance across two different
decisions, an error envelope, and a frozen test row that deliberately contains
banned wording.
"""

from __future__ import annotations

import re
from typing import Any

from dataforge.curricula import Registry
from dataforge.rows import make_conversation_row, validate_conversation_row
from examples.banking.taxonomy import TAXONOMY

#: Wording a trainable turn must never contain. ``app``/``apps`` keep the model
#: from deflecting to a surface it cannot see; the rest keep the corpus from
#: telling the model how the corpus was made. Frozen splits are exempt by
#: construction -- see :func:`dataforge.guards.banned_wording_leaks`.
BANNED_WORDING = re.compile(r"\b(?:apps?|mobile app|demo|synthetic|mock|sandbox|test)\b")

#: The argument whitelist enforced by :func:`validate_conversation_row`, so a
#: curriculum that invents an argument name fails at construction time.
TOOL_ARGUMENTS: dict[str, frozenset[str]] = {
    "freeze_card": frozenset({"card_id"}),
    "list_cards": frozenset({"status_filter"}),
}

#: The one train row whose context is scrubbed after export, and the literal
#: substitution applied to it. The phrase is banned wording that reached the
#: context through an upstream transcript, not through the teacher -- which is
#: exactly the case ``scrub_fields`` exists for: it is not editable text, so it
#: cannot be fixed by asking the teacher to rewrite it.
SCRUB_RECORD_ID = "train-multiturn-freeze-0"
SCRUB_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("while I am checking the mobile app", "while I am going through my accounts"),
)

_ACTIVE_CARDS = {
    "cards": [
        {"card_id": "card-4821", "label": "travel debit"},
        {"card_id": "card-7715", "label": "everyday credit"},
    ]
}

REGISTRY = Registry()


def _system_turn() -> dict[str, Any]:
    return {
        "role": "system",
        "content": "You are a retail banking servicing assistant with access to the card tools.",
        "loss": False,
    }


def _labels(**kwargs: Any) -> dict[str, Any]:
    return TAXONOMY.labels_for_example(**kwargs)


def _conversation_row(**kwargs: Any) -> dict[str, Any]:
    """``make_conversation_row`` plus this example's tool-argument whitelist.

    ``make_conversation_row`` validates the finished row, but it has no tool
    registry to check calls against; re-validating with ``tool_arguments`` is
    how a domain binds one. The teacher entry points get the same whitelist via
    ``functools.partial`` in ``build_tool_calls``.
    """
    row = make_conversation_row(**kwargs)
    validate_conversation_row(row, tool_arguments=TOOL_ARGUMENTS)
    return row


def _freeze_turn(card_id: str) -> dict[str, Any]:
    return {
        "name": "freeze_card",
        "arguments": {"card_id": card_id},
        "result": {"ok": True, "result": {"card_id": card_id, "status": "frozen"}},
    }


def _list_turn(status_filter: str) -> dict[str, Any]:
    return {
        "name": "list_cards",
        "arguments": {"status_filter": status_filter},
        "result": {"ok": True, "result": _ACTIVE_CARDS},
    }


def _execute_labels() -> dict[str, Any]:
    return _labels(
        intent="freeze_card",
        action="execute_tool",
        entity_resolution="resolved",
        tool_names=["freeze_card"],
    )


def _list_labels() -> dict[str, Any]:
    return _labels(
        intent="freeze_card",
        action="execute_tool",
        entity_resolution="resolved",
        tool_names=["list_cards"],
    )


def _clarify_labels() -> dict[str, Any]:
    return _labels(intent="freeze_card", action="clarify", entity_resolution="missing")


@REGISTRY.register("single_turn_tools", splits=("train", "validation"))
def single_turn_tool_rows(split: str) -> list[dict[str, Any]]:
    """One ``freeze_card`` and one ``list_cards`` row per trainable split."""
    if split == "train":
        return [
            _conversation_row(
                record_id="train-freeze-execute-0",
                context_messages=[_system_turn()],
                user_text="please freeze my travel debit card ending in 4821",
                action_turns=[_freeze_turn("card-4821")],
                final_response=(
                    "Your travel debit card ending in 4821 is frozen, so nothing new can be "
                    "charged to it."
                ),
                labels=_execute_labels(),
                example_kind="freeze_card_execute",
                source="synthetic-tool-freeze",
                source_split=split,
                group_id="tool-freeze-execute|train|0",
            ),
            _conversation_row(
                record_id="train-list-cards-0",
                context_messages=[_system_turn()],
                user_text="which of my cards are still active right now",
                action_turns=[_list_turn("active")],
                final_response=(
                    "You have two active cards right now: the travel debit ending in 4821 and "
                    "the everyday credit ending in 7715."
                ),
                labels=_list_labels(),
                example_kind="list_cards_execute",
                source="synthetic-tool-list",
                source_split=split,
                group_id="tool-list-cards|train|0",
            ),
        ]
    return [
        _conversation_row(
            record_id="validation-freeze-execute-0",
            context_messages=[_system_turn()],
            user_text="please put a freeze on my everyday credit card ending in 7715",
            action_turns=[_freeze_turn("card-7715")],
            final_response=(
                "Your everyday credit card ending in 7715 is frozen, and no further purchases "
                "will go through on it."
            ),
            labels=_execute_labels(),
            example_kind="freeze_card_execute",
            source="synthetic-tool-freeze",
            source_split=split,
            group_id="tool-freeze-execute|validation|0",
        ),
        _conversation_row(
            record_id="validation-list-cards-0",
            context_messages=[_system_turn()],
            user_text="can you tell me which cards are on my account",
            action_turns=[_list_turn("all")],
            final_response=(
                "There are two cards on your account: the travel debit ending in 4821 and the "
                "everyday credit ending in 7715."
            ),
            labels=_list_labels(),
            example_kind="list_cards_execute",
            source="synthetic-tool-list",
            source_split=split,
            group_id="tool-list-cards|validation|0",
        ),
    ]


@REGISTRY.register("multi_turn_freeze", splits=("train",))
def multi_turn_freeze_rows(split: str) -> list[dict[str, Any]]:
    """One multi-turn row: an earlier exchange **and** a context tool-call pair.

    The context tool call carries its own ``context_{record_id}_{n}`` id and
    ``loss: False``: it already happened, it is not what this row trains. The
    trainable call the row does mint is ``call_{record_id}_0``, on the other
    id lane. The context assistant turn is also this build's scrub target --
    it arrived carrying banned wording that no teacher rewrite can reach.
    """
    record_id = SCRUB_RECORD_ID
    context_call_id = f"context_{record_id}_0"
    return [
        _conversation_row(
            record_id=record_id,
            context_messages=[
                _system_turn(),
                {
                    "role": "user",
                    "content": "can you show me the cards on my account",
                    "loss": False,
                },
                {
                    "role": "assistant",
                    "content": None,
                    "loss": False,
                    "tool_calls": [
                        {
                            "id": context_call_id,
                            "index": 0,
                            "type": "function",
                            "function": {
                                "name": "list_cards",
                                "arguments": {"status_filter": "active"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": context_call_id,
                    "name": "list_cards",
                    "loss": False,
                    "content": {"ok": True, "result": _ACTIVE_CARDS},
                },
                {
                    "role": "assistant",
                    "content": (
                        "You have a travel debit ending in 4821 and an everyday credit ending "
                        "in 7715. Give me a moment while I am checking the mobile app for "
                        "anything pending."
                    ),
                    "loss": False,
                },
            ],
            user_text="freeze the travel debit one for now",
            action_turns=[_freeze_turn("card-4821")],
            final_response=(
                "I have frozen the travel debit card ending in 4821 for you; the everyday "
                "credit card is untouched."
            ),
            labels=_execute_labels(),
            example_kind="freeze_card_multiturn",
            source="synthetic-tool-multiturn",
            source_split=split,
            group_id="tool-multiturn|train|0",
        )
    ]


@REGISTRY.register("counterfactual_pair", splits=("train",))
def counterfactual_pair_rows(split: str) -> list[dict[str, Any]]:
    """The governed pair: one utterance, two contexts, two different decisions.

    Both rows share ``user_text`` verbatim -- that is the teaching signal, and
    it is also exactly what the global uniqueness check flags. They are exempt
    only because they are a *structurally proven* pair: same ``pair_id``,
    distinct ``pair_target``, one split, distinct contexts. Break any of those
    and ``compose`` fails the build. Their ``text`` still differs (the context
    differs), so ``compose``'s own dedup keeps both.
    """
    shared_user_text = "freeze it before anyone else uses it"
    return [
        _conversation_row(
            record_id="train-pair-execute-0",
            context_messages=[
                _system_turn(),
                {
                    "role": "user",
                    "content": "i lost my travel debit card ending in 4821 at the airport",
                    "loss": False,
                },
                {
                    "role": "assistant",
                    "content": "I am sorry that happened. I can secure that card for you.",
                    "loss": False,
                },
            ],
            user_text=shared_user_text,
            action_turns=[_freeze_turn("card-4821")],
            final_response=(
                "I have frozen the travel debit card ending in 4821, so whoever picked it up "
                "cannot use it."
            ),
            labels=_execute_labels(),
            example_kind="freeze_card_counterfactual",
            source="synthetic-tool-counterfactual",
            source_split=split,
            group_id="tool-pair-execute|train|0",
            pair_id="pair-freeze-wallet",
            pair_target="execute",
            pair_family="freeze_ambiguity",
        ),
        _conversation_row(
            record_id="train-pair-clarify-0",
            context_messages=[
                _system_turn(),
                {
                    "role": "user",
                    "content": "someone may have walked off with my wallet and both cards",
                    "loss": False,
                },
                {
                    "role": "assistant",
                    "content": "That sounds stressful. I can help you secure your cards.",
                    "loss": False,
                },
            ],
            user_text=shared_user_text,
            action_turns=[],
            final_response=(
                "Which card should I freeze first, the travel debit ending in 4821 or the "
                "everyday credit ending in 7715?"
            ),
            labels=_clarify_labels(),
            example_kind="freeze_card_counterfactual",
            source="synthetic-tool-counterfactual",
            source_split=split,
            group_id="tool-pair-clarify|train|0",
            pair_id="pair-freeze-wallet",
            pair_target="clarify",
            pair_family="freeze_ambiguity",
        ),
    ]


@REGISTRY.register("tool_error_envelope", splits=("validation",))
def tool_error_envelope_rows(split: str) -> list[dict[str, Any]]:
    """A row whose tool call came back as an error envelope, not a result.

    The envelope is part of the hashed, non-editable surface, so the teacher
    can rephrase the apology but cannot promote a failure into a success.
    """
    return [
        _conversation_row(
            record_id="validation-freeze-error-0",
            context_messages=[_system_turn()],
            user_text="freeze my old student card ending in 3390",
            action_turns=[
                {
                    "name": "freeze_card",
                    "arguments": {"card_id": "card-3390"},
                    "result": {
                        "ok": False,
                        "error": {
                            "code": "card_not_found",
                            "message": "No card ending in 3390 is open on this account.",
                        },
                    },
                }
            ],
            final_response=(
                "I could not find a card ending in 3390 on your account, so nothing was "
                "frozen; the cards I can see end in 4821 and 7715."
            ),
            labels=_execute_labels(),
            example_kind="freeze_card_error",
            source="synthetic-tool-error",
            source_split=split,
            group_id="tool-freeze-error|validation|0",
        )
    ]


@REGISTRY.register("frozen_regression", splits=("test",))
def frozen_regression_rows(split: str) -> list[dict[str, Any]]:
    """Frozen evaluation rows, never sent to the teacher.

    The first one deliberately contains "shown in the app" -- banned wording
    the model must never be *trained* to produce, which is precisely why the
    regression that watches for it has to be allowed to say it. It stays legal
    because ``banned_wording_leaks`` only scans trainable splits.
    """
    return [
        _conversation_row(
            record_id="test-frozen-pending-0",
            context_messages=[_system_turn()],
            user_text="why does my frozen card still have a pending charge on it",
            action_turns=[_list_turn("frozen")],
            final_response=(
                "That charge was authorized before the freeze, so it is still shown in the "
                "app until the merchant settles it."
            ),
            labels=_list_labels(),
            example_kind="frozen_regression",
            source="heldout-tool-regression",
            source_split=split,
            group_id="tool-frozen-pending|test|0",
        ),
        _conversation_row(
            record_id="test-freeze-both-0",
            context_messages=[_system_turn()],
            user_text="can you freeze both of my cards while i travel next week",
            action_turns=[_freeze_turn("card-4821"), _freeze_turn("card-7715")],
            final_response=(
                "Both cards are frozen for your trip: the travel debit ending in 4821 and the "
                "everyday credit ending in 7715."
            ),
            labels=_execute_labels(),
            example_kind="freeze_cards_batch",
            source="heldout-tool-regression",
            source_split=split,
            group_id="tool-freeze-both|test|0",
        ),
    ]
