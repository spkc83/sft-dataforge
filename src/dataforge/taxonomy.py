"""Declarative label taxonomy for governed SFT dataset construction.

A :class:`Taxonomy` is built from a plain dict (or JSON document) rather than
hard-coded constants. It owns two things:

* the hierarchy that an ``intent`` implies across one or more ordered
  dimensions (e.g. ``domain`` -> ``lane`` -> ``family``), and
* which ``(action, entity_resolution)`` pairs are legal for each value of a
  designated "gate" dimension (e.g. only the ``servicing`` lane may carry
  ``execute_tool``).

Deriving *which* action/entity-resolution a given example should carry from
raw signals (coreference targets, entity counts, and so on) is domain
business logic and stays in the caller's curriculum code. The taxonomy's job
is to validate that a fully-specified label set is internally consistent,
and to compute the label dict (names + indexes) for a given intent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class TaxonomyError(ValueError):
    """Raised when a taxonomy definition or a label set is invalid."""


@dataclass(frozen=True)
class IntentSpec:
    """The declared hierarchy and tool compatibility for one intent."""

    hierarchy: tuple[str, ...]
    tools: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Taxonomy:
    """A declarative, JSON-serializable label taxonomy.

    Attributes:
        dimensions: every label dimension (hierarchy dimensions plus the
            action and entity-resolution dimensions), each mapped to its
            ordered tuple of label names.
        hierarchy_dimensions: the ordered subset of ``dimensions`` that an
            intent determines, e.g. ``("domain", "lane", "family")``.
        null_hierarchy: the hierarchy values used when there is no intent
            (an out-of-scope / out-of-domain example).
        intents: intent name -> :class:`IntentSpec`.
        action_dimension: the name of the action dimension.
        entity_dimension: the name of the entity-resolution dimension.
        null_action: the action required when intent is ``None``.
        null_entity_resolution: the entity-resolution value required when
            intent is ``None``.
        gate_dimension: the hierarchy dimension whose value gates certain
            actions (e.g. ``"lane"``).
        gated_actions: action name -> the ``gate_dimension`` value required
            for that action to be legal (e.g. ``{"execute_tool":
            "servicing"}``).
        legal_action_entity_pairs: ``gate_dimension`` value -> the set of
            legal ``(action, entity_resolution)`` pairs for examples whose
            intent resolves to that gate value.
    """

    dimensions: Mapping[str, tuple[str, ...]]
    hierarchy_dimensions: tuple[str, ...]
    null_hierarchy: tuple[str, ...]
    intents: Mapping[str, IntentSpec]
    action_dimension: str = "action"
    entity_dimension: str = "entity_resolution"
    null_action: str = "refuse_out_of_scope"
    null_entity_resolution: str = "not_required"
    gate_dimension: str = "lane"
    gated_actions: Mapping[str, str] = field(default_factory=dict)
    legal_action_entity_pairs: Mapping[str, frozenset[tuple[str, str]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if len(self.null_hierarchy) != len(self.hierarchy_dimensions):
            raise TaxonomyError("null_hierarchy must match hierarchy_dimensions length")
        for dimension in (*self.hierarchy_dimensions, self.action_dimension, self.entity_dimension):
            if dimension not in self.dimensions:
                raise TaxonomyError(f"dimension {dimension!r} is not declared")
        if self.gate_dimension not in self.hierarchy_dimensions:
            raise TaxonomyError("gate_dimension must be one of hierarchy_dimensions")
        for value, name in zip(self.null_hierarchy, self.hierarchy_dimensions, strict=True):
            self._check_label(name, value)
        self._check_label(self.action_dimension, self.null_action)
        self._check_label(self.entity_dimension, self.null_entity_resolution)
        for intent, spec in self.intents.items():
            if len(spec.hierarchy) != len(self.hierarchy_dimensions):
                raise TaxonomyError(f"intent {intent!r} hierarchy length mismatch")
            for value, name in zip(spec.hierarchy, self.hierarchy_dimensions, strict=True):
                self._check_label(name, value)
        for action, gate_value in self.gated_actions.items():
            self._check_label(self.action_dimension, action)
            self._check_label(self.gate_dimension, gate_value)
        for gate_value, pairs in self.legal_action_entity_pairs.items():
            self._check_label(self.gate_dimension, gate_value)
            for action, entity_resolution in pairs:
                self._check_label(self.action_dimension, action)
                self._check_label(self.entity_dimension, entity_resolution)

    def _check_label(self, dimension: str, value: str) -> None:
        if value not in self.dimensions.get(dimension, ()):
            raise TaxonomyError(f"{value!r} is not a legal {dimension} label")

    def index_of(self, dimension: str, value: str) -> int:
        try:
            return self.dimensions[dimension].index(value)
        except (KeyError, ValueError) as error:
            raise TaxonomyError(f"unsupported {dimension} label: {value}") from error

    def hierarchy_for_intent(self, intent: str | None) -> tuple[str, ...]:
        if intent is None:
            return self.null_hierarchy
        try:
            return self.intents[intent].hierarchy
        except KeyError as error:
            raise TaxonomyError(f"unsupported intent: {intent}") from error

    def gate_value_for_intent(self, intent: str | None) -> str:
        index = self.hierarchy_dimensions.index(self.gate_dimension)
        return self.hierarchy_for_intent(intent)[index]

    def tools_for_intent(self, intent: str | None) -> frozenset[str]:
        if intent is None:
            return frozenset()
        try:
            return self.intents[intent].tools
        except KeyError as error:
            raise TaxonomyError(f"unsupported intent: {intent}") from error

    def labels_for_example(
        self,
        *,
        intent: str | None,
        action: str | None = None,
        entity_resolution: str | None = None,
        tool_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Build the full label dict (names + indexes) for one example.

        ``action``/``entity_resolution`` default to :attr:`null_action` /
        :attr:`null_entity_resolution` only when ``intent`` is ``None``;
        otherwise the caller must supply them explicitly (the taxonomy does
        not infer intent-specific business logic).
        """
        hierarchy = self.hierarchy_for_intent(intent)
        if intent is None:
            action = action or self.null_action
            entity_resolution = entity_resolution or self.null_entity_resolution
        if action is None or entity_resolution is None:
            raise TaxonomyError("action and entity_resolution are required for a non-null intent")

        labels: dict[str, Any] = {"intent": intent}
        for dimension, value in zip(self.hierarchy_dimensions, hierarchy, strict=True):
            labels[f"{dimension}_name"] = value
            labels[f"{dimension}_index"] = self.index_of(dimension, value)
        labels[f"{self.action_dimension}_name"] = action
        labels[f"{self.action_dimension}_index"] = self.index_of(self.action_dimension, action)
        labels[f"{self.entity_dimension}_name"] = entity_resolution
        labels[f"{self.entity_dimension}_index"] = self.index_of(
            self.entity_dimension, entity_resolution
        )
        self.validate_hierarchical_labels(labels, intent=intent, tool_names=tool_names)
        return labels

    def validate_hierarchical_labels(
        self,
        labels: Mapping[str, Any],
        *,
        intent: str | None = None,
        tool_names: Sequence[str] | None = None,
    ) -> None:
        """Re-validate a label dict for internal + hierarchy consistency."""
        intent = intent if intent is not None else labels.get("intent")
        expected_hierarchy = self.hierarchy_for_intent(intent)
        for dimension, expected in zip(self.hierarchy_dimensions, expected_hierarchy, strict=True):
            actual = str(labels.get(f"{dimension}_name", ""))
            if actual != expected:
                subject = f"intent {intent!r}" if intent is not None else "null-intent examples"
                raise TaxonomyError(f"{subject} requires {dimension}={expected!r}, got {actual!r}")
            index = labels.get(f"{dimension}_index")
            if index is not None and int(index) != self.index_of(dimension, actual):
                raise TaxonomyError(f"{dimension} index does not match {actual!r}")

        action = str(labels.get(f"{self.action_dimension}_name", ""))
        entity_resolution = str(labels.get(f"{self.entity_dimension}_name", ""))
        self.index_of(self.action_dimension, action)
        self.index_of(self.entity_dimension, entity_resolution)
        action_index = labels.get(f"{self.action_dimension}_index")
        if action_index is not None and int(action_index) != self.index_of(
            self.action_dimension, action
        ):
            raise TaxonomyError(f"{self.action_dimension} index does not match {action!r}")
        entity_index = labels.get(f"{self.entity_dimension}_index")
        if entity_index is not None and int(entity_index) != self.index_of(
            self.entity_dimension, entity_resolution
        ):
            raise TaxonomyError(
                f"{self.entity_dimension} index does not match {entity_resolution!r}"
            )

        if intent is None:
            if action != self.null_action or entity_resolution != self.null_entity_resolution:
                raise TaxonomyError(
                    f"null-intent examples require {self.action_dimension}={self.null_action!r} "
                    f"and {self.entity_dimension}={self.null_entity_resolution!r}"
                )
        elif action == self.null_action:
            raise TaxonomyError(f"{self.null_action!r} is only valid for null-intent examples")

        gate_value = expected_hierarchy[self.hierarchy_dimensions.index(self.gate_dimension)]
        required_gate = self.gated_actions.get(action)
        if required_gate is not None and gate_value != required_gate:
            raise TaxonomyError(
                f"{action!r} requires {self.gate_dimension}={required_gate!r}, got {gate_value!r}"
            )

        legal_pairs = self.legal_action_entity_pairs.get(gate_value)
        if legal_pairs is not None and (action, entity_resolution) not in legal_pairs:
            raise TaxonomyError(
                f"({action!r}, {entity_resolution!r}) is not legal for "
                f"{self.gate_dimension}={gate_value!r}"
            )

        if action in self.gated_actions and tool_names is not None:
            if not tool_names:
                raise TaxonomyError(f"{action!r} requires at least one compatible tool")
            compatible = self.tools_for_intent(intent)
            for tool_name in tool_names:
                if tool_name not in compatible:
                    raise TaxonomyError(
                        f"tool {tool_name!r} is incompatible with intent {intent!r}"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": {name: list(values) for name, values in self.dimensions.items()},
            "hierarchy_dimensions": list(self.hierarchy_dimensions),
            "null_hierarchy": list(self.null_hierarchy),
            "action_dimension": self.action_dimension,
            "entity_dimension": self.entity_dimension,
            "null_action": self.null_action,
            "null_entity_resolution": self.null_entity_resolution,
            "gate_dimension": self.gate_dimension,
            "gated_actions": dict(self.gated_actions),
            "legal_action_entity_pairs": {
                gate_value: sorted(list(pair) for pair in pairs)
                for gate_value, pairs in self.legal_action_entity_pairs.items()
            },
            "intents": {
                name: {"hierarchy": list(spec.hierarchy), "tools": sorted(spec.tools)}
                for name, spec in self.intents.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Taxonomy:
        dimensions = {
            name: tuple(values) for name, values in payload.get("dimensions", {}).items()
        }
        intents = {
            name: IntentSpec(
                hierarchy=tuple(spec["hierarchy"]),
                tools=frozenset(spec.get("tools", ())),
            )
            for name, spec in payload.get("intents", {}).items()
        }
        legal_action_entity_pairs = {
            gate_value: frozenset((action, entity) for action, entity in pairs)
            for gate_value, pairs in payload.get("legal_action_entity_pairs", {}).items()
        }
        kwargs: dict[str, Any] = dict(
            dimensions=dimensions,
            hierarchy_dimensions=tuple(payload["hierarchy_dimensions"]),
            null_hierarchy=tuple(payload["null_hierarchy"]),
            intents=intents,
            gated_actions=dict(payload.get("gated_actions", {})),
            legal_action_entity_pairs=legal_action_entity_pairs,
        )
        for key in (
            "action_dimension",
            "entity_dimension",
            "null_action",
            "null_entity_resolution",
            "gate_dimension",
        ):
            if key in payload:
                kwargs[key] = payload[key]
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str | Path) -> Taxonomy:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
