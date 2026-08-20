"""A handful of tiny curricula demonstrating every dataforge feature.

This intentionally does NOT reproduce hello-SLM's full banking curricula --
just enough rows, in enough shapes, to exercise: null-intent refusal,
execute_tool, clarify, retrieve_policy, multiturn context rendering, and a
held-out regression guard.
"""

from __future__ import annotations

from typing import Any

from dataforge.curricula import Registry
from dataforge.rows import make_row
from examples.banking.taxonomy import RELATION_LABELS, TAXONOMY

REGISTRY = Registry()

# Held out verbatim in test only; a leak guard fails the build if any of
# these phrases (or a long shared n-gram) show up in train/validation.
HELD_OUT_TEXTS = (
    "why is my card declined at the airport lounge",
    "explain the overdraft grace period one more time slowly",
)


def _labels(**kwargs: Any) -> dict[str, Any]:
    return TAXONOMY.labels_for_example(**kwargs)


@REGISTRY.register("out_of_scope", splits=("train", "validation", "test"))
def out_of_scope_rows(split: str) -> list[dict[str, Any]]:
    prompts = {
        "train": ("what's the weather like today", "tell me a joke", "set a timer for ten minutes"),
        "validation": ("play some music", "what time is it in tokyo"),
        "test": ("remind me to buy milk", "translate hello to spanish"),
    }[split]
    rows = []
    for index, current in enumerate(prompts):
        rows.append(
            make_row(
                current=current,
                labels=_labels(intent=None),
                relation_dimension=RELATION_LABELS,
                example_kind="out_of_scope",
                source="synthetic-out-of-scope",
                source_split=split,
                group_id=f"out-of-scope|{split}|{index}",
            )
        )
    return rows


@REGISTRY.register("balance_lookup", splits=("train", "validation", "test"))
def balance_lookup_rows(split: str) -> list[dict[str, Any]]:
    resolved_prompts = {
        "train": ("what's my checking account balance", "show me my account balances"),
        "validation": ("how much is in my savings account",),
        "test": ("what's my current account balance",),
    }[split]
    clarify_prompts = {
        "train": ("freeze my card",),
        "validation": ("please freeze that card",),
        "test": ("freeze the card",),
    }[split]
    rows = []
    for index, current in enumerate(resolved_prompts):
        rows.append(
            make_row(
                current=current,
                labels=_labels(
                    intent="view_balance",
                    action="execute_tool",
                    entity_resolution="resolved",
                    tool_names=["list_accounts"],
                ),
                relation_dimension=RELATION_LABELS,
                example_kind="balance_lookup_resolved",
                source="synthetic-balance-lookup",
                source_split=split,
                group_id=f"balance-resolved|{split}|{index}",
                extra={
                    "tool_names": ["list_accounts"],
                    "assistant_response": "I found your account balances.",
                },
            )
        )
    for index, current in enumerate(clarify_prompts):
        rows.append(
            make_row(
                current=current,
                labels=_labels(
                    intent="freeze_card",
                    action="clarify",
                    entity_resolution="missing",
                ),
                relation_dimension=RELATION_LABELS,
                example_kind="freeze_card_clarify",
                source="synthetic-balance-lookup",
                source_split=split,
                group_id=f"freeze-clarify|{split}|{index}",
            )
        )
    return rows


@REGISTRY.register("policy_faq", splits=("train", "validation", "test"))
def policy_faq_rows(split: str) -> list[dict[str, Any]]:
    prompts = {
        "train": ("what is the overdraft policy", "when do overdraft fees start applying"),
        "validation": ("what are the overdraft fees",),
        "test": ("can you explain the overdraft grace period",),
    }[split]
    rows = []
    for index, current in enumerate(prompts):
        rows.append(
            make_row(
                current=current,
                labels=_labels(
                    intent="policy_faq",
                    action="retrieve_policy",
                    entity_resolution="not_required",
                ),
                relation_dimension=RELATION_LABELS,
                example_kind="policy_faq",
                source="synthetic-policy-faq",
                source_split=split,
                group_id=f"policy-faq|{split}|{index}",
                extra={"assistant_response": "I can answer that banking policy question."},
            )
        )
    return rows


@REGISTRY.register("context_followup", splits=("train", "validation", "test"))
def context_followup_rows(split: str) -> list[dict[str, Any]]:
    """One multiturn row per split, exercising render_context's
    [PRIOR_STATE]/[PREVIOUS_*] flattening and the relation multi-hot vector."""
    history = [
        {"role": "user", "content": "what's my checking account balance"},
        {"role": "assistant", "content": "Your checking balance is $1,204.18."},
    ]
    return [
        make_row(
            current="freeze that account's card instead",
            history=history,
            labels=_labels(
                intent="freeze_card",
                action="clarify",
                entity_resolution="missing",
            ),
            relation_dimension=RELATION_LABELS,
            relation_names=["context_dependent"],
            example_kind="context_followup",
            source="synthetic-context-followup",
            source_split=split,
            group_id=f"context-followup|{split}",
            prior_state={"pending_servicing": "view_balance"},
        )
    ]


@REGISTRY.register("regression_heldout", splits=("test",))
def regression_heldout_rows(split: str) -> list[dict[str, Any]]:
    """Verbatim held-out regression prompts that must only ever live in test."""
    rows = []
    for index, current in enumerate(HELD_OUT_TEXTS):
        intent = "policy_faq" if "overdraft" in current else None
        if intent is None:
            labels = _labels(intent=None)
        else:
            labels = _labels(
                intent=intent, action="retrieve_policy", entity_resolution="not_required"
            )
        rows.append(
            make_row(
                current=current,
                labels=labels,
                relation_dimension=RELATION_LABELS,
                example_kind="heldout_regression",
                source="heldout-regression",
                source_split=split,
                group_id=f"heldout|{split}|{index}",
            )
        )
    return rows
