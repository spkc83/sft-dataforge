"""A small synthetic banking taxonomy, built purely from dataforge.taxonomy."""

from __future__ import annotations

from dataforge.taxonomy import IntentSpec, Taxonomy

DOMAIN_LABELS = ("out_of_domain", "banking")
LANE_LABELS = ("out_of_domain", "servicing", "policy")
ACTION_LABELS = ("refuse_out_of_scope", "execute_tool", "clarify", "retrieve_policy")
ENTITY_RESOLUTION_LABELS = ("not_required", "resolved", "missing")

RELATION_LABELS = ("context_dependent", "clarification_answer")

TAXONOMY = Taxonomy(
    dimensions={
        "domain": DOMAIN_LABELS,
        "lane": LANE_LABELS,
        "action": ACTION_LABELS,
        "entity_resolution": ENTITY_RESOLUTION_LABELS,
    },
    hierarchy_dimensions=("domain", "lane"),
    null_hierarchy=("out_of_domain", "out_of_domain"),
    intents={
        "view_balance": IntentSpec(
            hierarchy=("banking", "servicing"),
            tools=frozenset({"list_accounts"}),
        ),
        "freeze_card": IntentSpec(
            hierarchy=("banking", "servicing"),
            tools=frozenset({"list_cards", "freeze_card"}),
        ),
        "policy_faq": IntentSpec(hierarchy=("banking", "policy")),
    },
    null_action="refuse_out_of_scope",
    null_entity_resolution="not_required",
    gate_dimension="lane",
    gated_actions={"execute_tool": "servicing"},
    legal_action_entity_pairs={
        "servicing": frozenset(
            {
                ("execute_tool", "resolved"),
                ("clarify", "missing"),
            }
        ),
        "policy": frozenset({("retrieve_policy", "not_required")}),
        "out_of_domain": frozenset({("refuse_out_of_scope", "not_required")}),
    },
)
