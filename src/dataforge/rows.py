"""Generic row contract: context rendering, multi-hot relations, provenance."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

PRIOR_STATE_TAG = "[PRIOR_STATE]"
CURRENT_USER_TAG = "[CURRENT_USER]"
PREVIOUS_ASSISTANT_TAG = "[PREVIOUS_ASSISTANT]"
PREVIOUS_USER_TAG = "[PREVIOUS_USER]"


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation to spaces, and collapse whitespace.

    Used both to render a dedup/leak-guard key and to keep those checks
    resilient to cosmetic differences (casing, punctuation, spacing).
    """
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in text).split()
    )


def render_context(
    current: str,
    history: Sequence[Mapping[str, Any]] | None = None,
    *,
    max_exchanges: int = 3,
    prior_state: Mapping[str, Any] | None = None,
    is_meaningful_state: Callable[[Mapping[str, Any] | None], bool] | None = None,
) -> tuple[str, bool]:
    """Render one flattened text block: prior state, current turn, then history.

    ``history`` is a flat list of ``{"role": "user"|"assistant", "content":
    str}`` turns preceding ``current``, oldest first. Up to ``max_exchanges``
    complete (user, assistant) exchanges are rendered, most recent first,
    each as a ``[PREVIOUS_ASSISTANT]``/``[PREVIOUS_USER]`` pair.

    Returns the rendered text and whether any context (state or history)
    was actually included, so callers can label examples as
    context-dependent.
    """
    if not isinstance(current, str) or not current.strip():
        raise ValueError("current must be a non-empty string")
    if max_exchanges < 0:
        raise ValueError("max_exchanges must be non-negative")

    include_state = bool(prior_state) if is_meaningful_state is None else is_meaningful_state(
        prior_state
    )
    parts = []
    if include_state and prior_state:
        state_json = json.dumps(prior_state, sort_keys=True, separators=(",", ":"))
        parts.append(f"{PRIOR_STATE_TAG}\n{state_json}")
    parts.append(f"{CURRENT_USER_TAG}\n{current.strip()}")
    exchanges = _visible_complete_exchanges(history or [])
    selected = exchanges[-max_exchanges:] if max_exchanges else []
    for previous_user, previous_assistant in reversed(selected):
        parts.append(f"{PREVIOUS_ASSISTANT_TAG}\n{previous_assistant}")
        parts.append(f"{PREVIOUS_USER_TAG}\n{previous_user}")
    return "\n".join(parts), bool(selected or include_state)


def visible_history(history: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Filter a raw history sequence down to non-empty user/assistant turns."""
    return [
        {"role": str(item["role"]), "content": str(item["content"]).strip()}
        for item in history
        if item.get("role") in {"user", "assistant"} and str(item.get("content", "")).strip()
    ]


def _visible_complete_exchanges(
    history: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    exchanges: list[tuple[str, str]] = []
    pending_user: str | None = None
    for item in history:
        role = item.get("role")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user":
            pending_user = content.strip()
        elif role == "assistant" and pending_user is not None:
            exchanges.append((pending_user, content.strip()))
            pending_user = None
    return exchanges


def make_row(
    *,
    current: str,
    history: Sequence[Mapping[str, Any]] = (),
    labels: Mapping[str, Any],
    relation_dimension: Sequence[str] = (),
    relation_names: Sequence[str] = (),
    example_kind: str,
    source: str,
    source_split: str,
    group_id: str,
    trajectory_id: str | None = None,
    prior_state: Mapping[str, Any] | None = None,
    pair_id: str | None = None,
    pair_target: str | None = None,
    pair_family: str | None = None,
    max_exchanges: int = 3,
    is_meaningful_state: Callable[[Mapping[str, Any] | None], bool] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one governed row: rendered text, labels, relations, provenance.

    ``labels`` is normally the dict returned by
    :meth:`dataforge.taxonomy.Taxonomy.labels_for_example` and is spread
    into the row as-is. ``relation_dimension`` is the full ordered list of
    relation label names (e.g. all discourse-relation tags this dataset
    supports); ``relation_names`` are the ones that apply to this row, and
    are encoded as a multi-hot vector in ``relation_dimension`` order.
    """
    unknown = set(relation_names) - set(relation_dimension)
    if unknown:
        raise ValueError(f"unknown relation names: {sorted(unknown)}")
    relation_index = {name: index for index, name in enumerate(relation_dimension)}
    relation_labels = [0] * len(relation_dimension)
    for relation in relation_names:
        relation_labels[relation_index[relation]] = 1

    cleaned_history = visible_history(history)
    text, has_context = render_context(
        current,
        cleaned_history,
        max_exchanges=max_exchanges,
        prior_state=prior_state,
        is_meaningful_state=is_meaningful_state,
    )
    row: dict[str, Any] = {
        "text": text,
        "current_text": current.strip(),
        "history": cleaned_history,
        "has_context": has_context,
        **dict(labels),
        "relation_labels": relation_labels,
        "example_kind": example_kind,
        "source": source,
        "source_split": source_split,
        "group_id": group_id,
        "trajectory_id": trajectory_id or group_id,
        "prior_state": dict(prior_state or {}),
        "pair_id": pair_id,
        "pair_target": pair_target,
        "pair_family": pair_family,
    }
    if extra:
        row.update(extra)
    return row
