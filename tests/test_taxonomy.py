from __future__ import annotations

import pytest

from dataforge.taxonomy import IntentSpec, Taxonomy, TaxonomyError


def _taxonomy() -> Taxonomy:
    return Taxonomy(
        dimensions={
            "domain": ("out_of_domain", "banking"),
            "lane": ("out_of_domain", "servicing", "policy"),
            "action": ("refuse_out_of_scope", "execute_tool", "clarify", "retrieve_policy"),
            "entity_resolution": ("not_required", "resolved", "missing"),
        },
        hierarchy_dimensions=("domain", "lane"),
        null_hierarchy=("out_of_domain", "out_of_domain"),
        intents={
            "view_balance": IntentSpec(
                hierarchy=("banking", "servicing"), tools=frozenset({"list_accounts"})
            ),
            "policy_faq": IntentSpec(hierarchy=("banking", "policy")),
        },
        gated_actions={"execute_tool": "servicing"},
        legal_action_entity_pairs={
            "servicing": frozenset({("execute_tool", "resolved"), ("clarify", "missing")}),
            "policy": frozenset({("retrieve_policy", "not_required")}),
            "out_of_domain": frozenset({("refuse_out_of_scope", "not_required")}),
        },
    )


def test_labels_for_example_null_intent() -> None:
    taxonomy = _taxonomy()
    labels = taxonomy.labels_for_example(intent=None)
    assert labels["domain_name"] == "out_of_domain"
    assert labels["lane_name"] == "out_of_domain"
    assert labels["action_name"] == "refuse_out_of_scope"
    assert labels["entity_resolution_name"] == "not_required"


def test_labels_for_example_execute_tool() -> None:
    taxonomy = _taxonomy()
    labels = taxonomy.labels_for_example(
        intent="view_balance",
        action="execute_tool",
        entity_resolution="resolved",
        tool_names=["list_accounts"],
    )
    assert labels["lane_name"] == "servicing"
    assert labels["action_index"] == taxonomy.index_of("action", "execute_tool")


def test_hierarchy_mismatch_rejected() -> None:
    taxonomy = _taxonomy()
    with pytest.raises(TaxonomyError):
        taxonomy.validate_hierarchical_labels(
            {
                "domain_name": "out_of_domain",  # wrong: view_balance is banking/servicing
                "lane_name": "servicing",
                "action_name": "execute_tool",
                "entity_resolution_name": "resolved",
            },
            intent="view_balance",
        )


def test_execute_tool_requires_servicing_lane() -> None:
    taxonomy = _taxonomy()
    with pytest.raises(TaxonomyError):
        taxonomy.labels_for_example(
            intent="policy_faq",
            action="execute_tool",
            entity_resolution="resolved",
        )


def test_illegal_action_entity_pair_rejected() -> None:
    taxonomy = _taxonomy()
    with pytest.raises(TaxonomyError):
        taxonomy.labels_for_example(
            intent="view_balance",
            action="execute_tool",
            entity_resolution="missing",  # not a legal pair for servicing
        )


def test_incompatible_tool_rejected() -> None:
    taxonomy = _taxonomy()
    with pytest.raises(TaxonomyError):
        taxonomy.labels_for_example(
            intent="view_balance",
            action="execute_tool",
            entity_resolution="resolved",
            tool_names=["freeze_card"],  # not compatible with view_balance
        )


def test_execute_tool_requires_a_tool() -> None:
    taxonomy = _taxonomy()
    with pytest.raises(TaxonomyError):
        taxonomy.labels_for_example(
            intent="view_balance",
            action="execute_tool",
            entity_resolution="resolved",
            tool_names=[],
        )


def test_refuse_out_of_scope_only_valid_for_null_intent() -> None:
    taxonomy = _taxonomy()
    with pytest.raises(TaxonomyError):
        taxonomy.labels_for_example(
            intent="view_balance",
            action="refuse_out_of_scope",
            entity_resolution="not_required",
        )


def test_null_intent_requires_null_action_and_entity() -> None:
    taxonomy = _taxonomy()
    with pytest.raises(TaxonomyError):
        taxonomy.validate_hierarchical_labels(
            {
                "domain_name": "out_of_domain",
                "lane_name": "out_of_domain",
                "action_name": "clarify",
                "entity_resolution_name": "missing",
            },
            intent=None,
        )


def test_round_trip_to_dict_from_dict() -> None:
    taxonomy = _taxonomy()
    restored = Taxonomy.from_dict(taxonomy.to_dict())
    labels = restored.labels_for_example(
        intent="view_balance",
        action="execute_tool",
        entity_resolution="resolved",
        tool_names=["list_accounts"],
    )
    assert labels["lane_name"] == "servicing"


def test_unknown_intent_rejected() -> None:
    taxonomy = _taxonomy()
    with pytest.raises(TaxonomyError):
        taxonomy.labels_for_example(
            intent="nonexistent", action="clarify", entity_resolution="missing"
        )
