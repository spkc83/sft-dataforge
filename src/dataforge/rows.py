"""Generic row contract: context rendering, multi-hot relations, provenance."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Mapping, MutableMapping, Sequence
from copy import deepcopy
from typing import Any

PRIOR_STATE_TAG = "[PRIOR_STATE]"
CURRENT_USER_TAG = "[CURRENT_USER]"
PREVIOUS_ASSISTANT_TAG = "[PREVIOUS_ASSISTANT]"
PREVIOUS_USER_TAG = "[PREVIOUS_USER]"

#: Fields a make_row() row derives from other fields, mapped to the fields
#: they're derived from. Anything editable by a teacher that overlaps one of
#: these dependency sets must be paired with a `rederive` callback (see
#: :func:`rederive_text`) -- otherwise the derived field goes stale silently.
#: This is the default `derived_fields` for :mod:`dataforge.teacher`.
DERIVED_FIELDS: dict[str, tuple[str, ...]] = {
    "text": ("current_text", "history", "prior_state"),
}

#: The same contract for :func:`make_conversation_row` rows: the rendered
#: transcript, its flattened `text`, and the `expected_tool_calls` projection
#: are all derived from the four source fields. Pass this as `derived_fields`
#: (with `rederive=rederive_conversation` and
#: `validate=validate_conversation_row`) to the :mod:`dataforge.teacher` entry
#: points -- it is deliberately *not* the default map, so a caller who edits a
#: conversation row's text must say so explicitly.
CONVERSATION_DERIVED_FIELDS: dict[str, tuple[str, ...]] = {
    "messages": ("context_messages", "user_text", "action_turns", "final_response"),
    "text": ("context_messages", "user_text"),
    "expected_tool_calls": ("action_turns",),
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation to spaces, and collapse whitespace.

    Used both to render a dedup/leak-guard key and to keep those checks
    resilient to cosmetic differences (casing, punctuation, spacing).
    """
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in text).split()
    )


def normalize_text_ascii(text: str) -> str:
    """Lowercase, then map every run of non-``[a-z0-9]`` characters to a single
    space, and strip.

    Deliberately **not** the same predicate as :func:`normalize_text`: that one
    uses Unicode-aware ``str.isalnum()``, so ``"Café"`` normalizes to ``"café"``,
    while this one treats any non-ASCII character as punctuation and yields
    ``"caf"``. This is the exact predicate the v9 uniqueness checks used, so
    duplicate detection ported from there keeps its measured behaviour; it is
    the default ``normalize`` for
    :func:`dataforge.guards.duplicate_text_leaks`. Use :func:`normalize_text`
    for anything that should respect non-ASCII alphabets.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


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


def rederive_text(
    row: MutableMapping[str, Any],
    *,
    max_exchanges: int = 3,
    is_meaningful_state: Callable[[Mapping[str, Any] | None], bool] | None = None,
) -> None:
    """Recompute ``row["text"]``/``row["has_context"]`` from ``current_text``,
    ``history``, and ``prior_state``.

    Pass this as ``rederive`` to
    :func:`dataforge.teacher.export_teacher_requests` /
    :func:`dataforge.teacher.import_teacher_responses` whenever
    ``current_text``, ``history``, or ``prior_state`` is an editable field, so
    the rendered ``text`` a model actually trains on stays in sync with the
    edit instead of going stale (see :data:`DERIVED_FIELDS`).
    """
    text, has_context = render_context(
        str(row["current_text"]),
        row.get("history") or (),
        max_exchanges=max_exchanges,
        prior_state=row.get("prior_state") or None,
        is_meaningful_state=is_meaningful_state,
    )
    row["text"] = text
    row["has_context"] = has_context


def validate_row_consistency(
    row: Mapping[str, Any],
    *,
    max_exchanges: int = 3,
    is_meaningful_state: Callable[[Mapping[str, Any] | None], bool] | None = None,
) -> None:
    """Raise ``ValueError`` if ``text``/``has_context`` disagree with a fresh
    render of ``current_text``/``history``/``prior_state``.

    A structural post-edit check, analogous to hello-SLM's
    ``validate_records``: pass as ``validate`` to
    :func:`dataforge.teacher.import_teacher_responses` for defense in depth
    even when a field's derivation dependency isn't declared in
    :data:`DERIVED_FIELDS` (e.g. because a caller opted out of the
    dependency guard).
    """
    expected_text, expected_has_context = render_context(
        str(row["current_text"]),
        row.get("history") or (),
        max_exchanges=max_exchanges,
        prior_state=row.get("prior_state") or None,
        is_meaningful_state=is_meaningful_state,
    )
    if row.get("text") != expected_text:
        raise ValueError("row text is inconsistent with current_text/history/prior_state")
    if row.get("has_context") != expected_has_context:
        raise ValueError("row has_context is inconsistent with current_text/history/prior_state")


def _conversation_context_turns(
    context_messages: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    """Context turns that belong in the rendered ``text``: ``user``/``assistant``
    turns carrying non-blank ``str`` content, oldest first.

    Tool-call assistants have ``content: None`` and are skipped by the ``str``
    test; ``system`` and ``tool`` turns are excluded by role, exactly as
    :func:`render_context` excludes everything but user/assistant history.
    """
    turns: list[tuple[str, str]] = []
    for message in context_messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        if not content.strip():
            continue
        turns.append((str(role), content.strip()))
    return turns


def render_conversation_text(
    context_messages: Sequence[Mapping[str, Any]],
    user_text: str,
) -> str:
    """Render the flattened ``text`` field of a conversation row.

    Same tag vocabulary as :func:`render_context` -- ``[CURRENT_USER]`` then the
    prior turns most-recent-first as ``[PREVIOUS_ASSISTANT]``/``[PREVIOUS_USER]``
    -- but over the **whole** context with no ``max_exchanges`` cap, and turn by
    turn rather than by complete (user, assistant) exchange. For a
    conventionally alternating context the two render byte-identically; the
    difference is that nothing here is ever dropped, so two rows that differ
    only in their earliest turns cannot collapse to the same ``text`` and be
    deduplicated away by :func:`dataforge.curricula.compose`.
    """
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a non-empty string")
    parts = [f"{CURRENT_USER_TAG}\n{user_text.strip()}"]
    for role, content in reversed(_conversation_context_turns(context_messages)):
        tag = PREVIOUS_USER_TAG if role == "user" else PREVIOUS_ASSISTANT_TAG
        parts.append(f"{tag}\n{content}")
    return "\n".join(parts)


def render_conversation(
    *,
    record_id: str,
    context_messages: Sequence[Mapping[str, Any]],
    user_text: str,
    action_turns: Sequence[Mapping[str, Any]],
    final_response: str,
) -> list[dict[str, Any]]:
    """Render the full ``messages`` transcript of a conversation row.

    Layout: the context verbatim (ids and ``loss`` labels included -- context
    tool-call ids are the caller's, conventionally ``context_{record_id}_{n}``,
    and are never rewritten here), the current user turn, then one
    ``assistant`` tool-call message plus its ``tool`` result per entry in
    ``action_turns``, then the trainable final assistant turn.

    Trainable tool-call ids are minted deterministically as
    ``call_{record_id}_{n}``, so the transcript is a pure function of the
    row's source fields and any drift is detectable
    (:func:`validate_conversation_row`).
    """
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record_id must be a non-empty string")
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("user_text must be a non-empty string")
    if not isinstance(final_response, str) or not final_response.strip():
        raise ValueError("final_response must be a non-empty string")

    messages: list[dict[str, Any]] = [dict(deepcopy(message)) for message in context_messages]
    messages.append({"role": "user", "content": user_text.strip(), "loss": False})
    for index, turn in enumerate(action_turns):
        missing = {"name", "arguments", "result"} - set(turn)
        if missing:
            raise ValueError(
                f"{record_id} action turn {index} is missing {sorted(missing)}: an action turn is "
                '{"name": str, "arguments": dict, "result": envelope}'
            )
        call_id = f"call_{record_id}_{index}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "loss": True,
                "tool_calls": [
                    {
                        "id": call_id,
                        "index": 0,
                        "type": "function",
                        "function": {
                            "name": turn["name"],
                            "arguments": deepcopy(turn["arguments"]),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": turn["name"],
                "loss": False,
                "content": deepcopy(turn["result"]),
            }
        )
    messages.append({"role": "assistant", "content": final_response.strip(), "loss": True})
    return messages


def _derive_expected_tool_calls(
    action_turns: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"name": turn["name"], "arguments": deepcopy(turn["arguments"])} for turn in action_turns
    ]


def make_conversation_row(
    *,
    record_id: str,
    context_messages: Sequence[Mapping[str, Any]],
    user_text: str,
    action_turns: Sequence[Mapping[str, Any]],
    final_response: str,
    labels: Mapping[str, Any],
    example_kind: str,
    source: str,
    source_split: str,
    group_id: str,
    trajectory_id: str | None = None,
    pair_id: str | None = None,
    pair_target: str | None = None,
    pair_family: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one governed tool-calling conversation row.

    The row's source of truth is four fields -- ``context_messages``,
    ``user_text``, ``action_turns``, ``final_response`` -- from which
    ``messages``, ``text``, ``has_context`` and ``expected_tool_calls`` are
    derived (:data:`CONVERSATION_DERIVED_FIELDS`). ``context_messages`` and
    ``action_turns`` are deep-copied, so later mutation of the caller's inputs
    cannot silently desynchronize the derived transcript.

    ``action_turns`` is the ordered, non-editable list of tool calls the
    skeleton decided on, each ``{"name": str, "arguments": dict, "result":
    {"ok": True, "result": ...} | {"ok": False, "error": {"code", "message"}}}``.
    ``expected_tool_calls`` is derived from it rather than passed in, so there
    is exactly one source of truth for what the row trains the model to call.

    Every context message must already carry its own ``loss`` label (``False``)
    and, for context tool calls, its own ids -- they are rendered verbatim. The
    finished row is checked with :func:`validate_conversation_row` before it is
    returned, so a malformed context fails at construction rather than at
    training time.
    """
    text = render_conversation_text(context_messages, user_text)
    row: dict[str, Any] = {
        "record_id": record_id,
        "context_messages": [dict(deepcopy(message)) for message in context_messages],
        "user_text": user_text.strip(),
        "action_turns": [dict(deepcopy(turn)) for turn in action_turns],
        "final_response": final_response.strip(),
        "expected_tool_calls": _derive_expected_tool_calls(action_turns),
        "messages": render_conversation(
            record_id=record_id,
            context_messages=context_messages,
            user_text=user_text,
            action_turns=action_turns,
            final_response=final_response,
        ),
        "text": text,
        "has_context": bool(_conversation_context_turns(context_messages)),
        **dict(labels),
        "example_kind": example_kind,
        "source": source,
        "source_split": source_split,
        "group_id": group_id,
        "trajectory_id": trajectory_id or group_id,
        "pair_id": pair_id,
        "pair_target": pair_target,
        "pair_family": pair_family,
    }
    if extra:
        row.update(extra)
    validate_conversation_row(row)
    return row


def rederive_conversation(row: MutableMapping[str, Any]) -> None:
    """Recompute ``expected_tool_calls``/``messages``/``text``/``has_context``
    from a conversation row's source fields.

    Pass as ``rederive`` (with ``validate=validate_conversation_row`` and
    ``derived_fields=CONVERSATION_DERIVED_FIELDS``) to
    :func:`dataforge.teacher.export_teacher_requests` /
    :func:`dataforge.teacher.import_teacher_responses` whenever ``user_text``
    or ``final_response`` is editable, so the transcript the model actually
    trains on reflects the teacher's rewrite instead of going stale.
    """
    context_messages = row.get("context_messages") or []
    action_turns = row.get("action_turns") or []
    row["expected_tool_calls"] = _derive_expected_tool_calls(action_turns)
    row["messages"] = render_conversation(
        record_id=str(row["record_id"]),
        context_messages=context_messages,
        user_text=str(row["user_text"]),
        action_turns=action_turns,
        final_response=str(row["final_response"]),
    )
    row["text"] = render_conversation_text(context_messages, str(row["user_text"]))
    row["has_context"] = bool(_conversation_context_turns(context_messages))


def validate_conversation_messages(
    record_id: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    final_response: str | None = None,
    expected_tool_calls: Sequence[Mapping[str, Any]] | None = None,
    tool_arguments: Mapping[str, Collection[str]] | None = None,
) -> None:
    """Run the transcript state machine over ``messages``; raise ``ValueError``.

    A port of hello-SLM's ``validate_records`` message pass, with the
    domain-specific parts made optional: ``tool_arguments`` is the per-tool
    argument whitelist (and doubles as the tool-name registry) when the caller
    has one, and ``final_response``/``expected_tool_calls`` are cross-checked
    against the transcript when given. Every message keeps v9's
    ``"{record_id} ..."`` prefix so an error names the offending row.

    Checked, in one forward pass: roles; exactly one call per tool-call
    assistant, with ``content: None``, a boolean ``loss``, ``index`` 0, and a
    deterministic id per lane (``call_{record_id}_{n}`` for trainable calls,
    ``context_{record_id}_{n}`` for context ones, counted independently);
    object arguments inside the whitelist; each call answered by the
    immediately following ``tool`` message with a matching ``tool_call_id``, an
    unlabeled ``loss``, and an envelope whose key set is exactly ``{ok, result}``
    or ``{ok, error}``; unlabeled ``system``/``user`` turns; no call left
    pending; and a trainable ``assistant`` final turn last.
    """
    pending_call_id: str | None = None
    seen_call_ids: set[str] = set()
    trainable_call_ids: list[str] = []
    context_call_ids: list[str] = []
    canonical_calls: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            if pending_call_id is not None:
                raise ValueError(f"{record_id} tool result correlation mismatch")
            tool_calls = message["tool_calls"]
            if not isinstance(tool_calls, list) or len(tool_calls) != 1:
                raise ValueError(f"{record_id} tool-call assistant must contain exactly one call")
            if message.get("content") is not None:
                raise ValueError(f"{record_id} tool-call assistant has content")
            loss = message.get("loss")
            if loss is not True and loss is not False:
                raise ValueError(f"{record_id} assistant call has invalid loss label")
            call = tool_calls[0]
            if not isinstance(call, Mapping):
                raise ValueError(f"{record_id} tool-call assistant must contain exactly one call")
            call_id = call.get("id")
            if not isinstance(call_id, str):
                raise ValueError(f"{record_id} tool call id must be a string")
            if call.get("index") != 0:
                raise ValueError(
                    f"{record_id} tool call index must restart at zero per assistant message"
                )
            if call_id in seen_call_ids:
                raise ValueError(f"{record_id} has duplicate tool call id")
            seen_call_ids.add(call_id)
            if loss is True:
                if call_id != f"call_{record_id}_{len(trainable_call_ids)}":
                    raise ValueError(f"{record_id} has unstable tool call id")
                trainable_call_ids.append(call_id)
            else:
                if call_id != f"context_{record_id}_{len(context_call_ids)}":
                    raise ValueError(f"{record_id} has unstable context tool call id")
                context_call_ids.append(call_id)
            function = call.get("function")
            if not isinstance(function, Mapping):
                function = {}
            name = function.get("name")
            arguments = function.get("arguments")
            if tool_arguments is not None and name not in tool_arguments:
                raise ValueError(f"{record_id} uses unknown tool {name!r}")
            if not isinstance(arguments, dict):
                raise ValueError(f"{record_id} tool arguments must be object")
            if tool_arguments is not None:
                extras = set(arguments) - set(tool_arguments[str(name)])
                if extras:
                    raise ValueError(
                        f"{record_id} unsupported arguments for {name}: {sorted(extras)}"
                    )
            if loss is True:
                canonical_calls.append({"name": name, "arguments": dict(arguments)})
            pending_call_id = call_id
        elif role == "tool":
            if message.get("loss") is not False:
                raise ValueError(f"{record_id} tool result is labeled")
            content = message.get("content")
            if not isinstance(content, Mapping):
                raise ValueError(f"{record_id} tool content must be object")
            ok = content.get("ok")
            if ok is True and set(content) != {"ok", "result"}:
                raise ValueError(f"{record_id} success envelope is invalid")
            if ok is False and set(content) != {"ok", "error"}:
                raise ValueError(f"{record_id} error envelope is invalid")
            if ok is not True and ok is not False:
                raise ValueError(f"{record_id} tool envelope missing ok")
            if pending_call_id is None or message.get("tool_call_id") != pending_call_id:
                raise ValueError(f"{record_id} tool result correlation mismatch")
            pending_call_id = None
        elif role == "assistant":
            if pending_call_id is not None:
                raise ValueError(f"{record_id} tool result correlation mismatch")
            loss = message.get("loss")
            if loss is not True and loss is not False:
                raise ValueError(f"{record_id} assistant message has invalid loss label")
        elif role in {"system", "user"}:
            if pending_call_id is not None:
                raise ValueError(f"{record_id} tool result correlation mismatch")
            if message.get("loss") is not False:
                raise ValueError(f"{record_id} context message is labeled")
        else:
            raise ValueError(f"{record_id} has invalid role {role!r}")
    if pending_call_id is not None:
        raise ValueError(f"{record_id} tool result correlation mismatch")

    final = messages[-1] if messages else None
    if (
        final is None
        or final.get("role") != "assistant"
        or final.get("tool_calls")
        or not isinstance(final.get("content"), str)
        or final.get("loss") is not True
        or (final_response is not None and final.get("content") != final_response)
    ):
        raise ValueError(f"{record_id} final assistant response must be trainable")
    if expected_tool_calls is not None and canonical_calls != [
        dict(call) for call in expected_tool_calls
    ]:
        raise ValueError(f"{record_id} expected tool_calls mismatch")


def validate_conversation_row(
    row: Mapping[str, Any],
    *,
    tool_arguments: Mapping[str, Collection[str]] | None = None,
) -> None:
    """Raise ``ValueError`` unless a conversation row is internally consistent.

    Two layers, in order. First the derivation check: ``messages``, ``text``,
    ``expected_tool_calls`` and ``has_context`` must equal a fresh render of
    ``context_messages``/``user_text``/``action_turns``/``final_response`` --
    this is what catches a teacher edit whose ``rederive`` never ran, or ran
    and was wrong. Then :func:`validate_conversation_messages` runs the
    transcript state machine, which is what catches a malformed *context* (the
    part this module copies through verbatim rather than mints).

    Pass as ``validate`` to the :mod:`dataforge.teacher` entry points together
    with ``rederive=rederive_conversation``; ``tool_arguments`` (a
    ``{tool_name: allowed_argument_names}`` map) can be bound with
    :func:`functools.partial` when the caller has a tool registry to enforce.
    """
    record_id = str(row["record_id"])
    context_messages = row.get("context_messages") or []
    action_turns = row.get("action_turns") or []
    user_text = str(row["user_text"])
    final_response = str(row["final_response"])
    if row.get("messages") != render_conversation(
        record_id=record_id,
        context_messages=context_messages,
        user_text=user_text,
        action_turns=action_turns,
        final_response=final_response,
    ):
        raise ValueError(f"{record_id} messages does not match its rendered conversation")
    if row.get("text") != render_conversation_text(context_messages, user_text):
        raise ValueError(f"{record_id} text does not match its rendered context")
    if row.get("expected_tool_calls") != _derive_expected_tool_calls(action_turns):
        raise ValueError(
            f"{record_id} expected_tool_calls does not match its rendered action turns"
        )
    if row.get("has_context") != bool(_conversation_context_turns(context_messages)):
        raise ValueError(f"{record_id} has_context does not match its rendered context")
    validate_conversation_messages(
        record_id,
        row["messages"],
        final_response=final_response.strip(),
        expected_tool_calls=row["expected_tool_calls"],
        tool_arguments=tool_arguments,
    )
