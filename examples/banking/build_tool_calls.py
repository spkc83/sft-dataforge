"""Build the example banking tool-calling dataset end-to-end.

Run with ``python -m examples.banking.build_tool_calls [output_dir]``. Like
``examples.banking.build`` it is deterministic: two runs produce byte-identical
files, given the fixed ``created_at`` and a stub teacher with no randomness.

This is the second, separate build in the examples package -- conversation rows
rather than classifier rows -- and it exists to wire the v9 mechanisms end to
end. The order of the pipeline is the point, so it is worth stating once:

1. ``compose`` runs the curricula, then the **pre-dedup** duplicate-``user_text``
   check on the raw rows and fails the build if a duplicate is not a governed
   counterfactual pair. This has to happen before dedup, because dedup on
   ``text`` would otherwise silently drop the evidence.
2. Teacher requests are exported over ``final_response`` only, pinning a hash
   of everything else -- context, user turn, tool calls, tool results, labels.
3. The stub teacher rewrites the finals.
4. ``scrub_fields`` fixes banned wording in one train row's **context**, which
   is not editable text and therefore cannot be fixed by the teacher. That
   moves the row's immutable hash and stamps the pre-scrub hash into
   provenance, which is why the import below has to accept it.
5. ``check_teacher_batch`` audits the whole batch at once and the build raises
   on any finding.
6. ``import_teacher_responses`` applies the finals, re-deriving the transcript.
7. The governance report is **rebuilt** on the realized rows -- the report that
   gates emission must describe the bytes actually written, and both the
   teacher pass and the scrub changed them -- and only then is the dataset
   written and locked.

Step 4 is why the report from step 1 is not the report that gates: at compose
time the banned-wording check legitimately still sees "mobile app" in that
context turn. The scrub happens later, and the rebuilt report is clean.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from dataforge.checks import (
    banned_pattern,
    check_teacher_batch,
    fields_present,
    format_findings,
    hash_pinned,
    min_words,
    no_extra_keys,
    opening_ngram_cap,
    unique_normalized,
)
from dataforge.curricula import build_report, compose
from dataforge.emit import write_dataset, write_source_lock
from dataforge.guards import (
    banned_wording_leaks,
    duplicate_text_leaks,
    paired_counterfactual_exemption,
)
from dataforge.rows import (
    CONVERSATION_DERIVED_FIELDS,
    rederive_conversation,
    validate_conversation_row,
)
from dataforge.teacher import (
    compute_teacher_prompt_hash,
    export_teacher_requests,
    import_teacher_responses,
    scrub_fields,
)
from examples.banking.taxonomy import ACTION_LABELS, ENTITY_RESOLUTION_LABELS, LANE_LABELS
from examples.banking.tool_curricula import (
    BANNED_WORDING,
    REGISTRY,
    SCRUB_RECORD_ID,
    SCRUB_SUBSTITUTIONS,
    TOOL_ARGUMENTS,
)

DEFAULT_OUTPUT_DIR = Path("dist/banking-tool-calls-example")
CREATED_AT = "2024-01-01T00:00:00+00:00"
TEACHER_MODEL = "stub-teacher-tools-v1"
SPLIT_ORDER = ("train", "validation", "test")
TRAINABLE_SPLITS = ("train", "validation")
EDITABLE_FIELDS = ("final_response",)
VOICE_SPEC_PATH = Path(__file__).resolve().parent / "voice_spec.md"

#: The one sentence the stub teacher adds, so a realized final is visibly a
#: rewrite. A real teacher must vary its wording; ``opening_ngram_cap`` below
#: is what notices when it stops doing so.
TEACHER_CLOSER = "Let me know if you would like anything else on the account."

Splits = Mapping[str, Sequence[Mapping[str, Any]]]

#: Reused three times -- as the pre-dedup exemption, and inside the checker's
#: cross-row uniqueness rule -- because "this duplicate is a governed pair" is
#: one predicate, not one per call site. ``context_fields`` names the
#: conversation row's context; the default (``history``) is the classifier
#: row's.
GOVERNED_PAIR_EXEMPTION = paired_counterfactual_exemption(context_fields=("context_messages",))

#: The transcript validator bound to this domain's tool registry, passed to
#: both teacher entry points so a rewrite that somehow disturbed a tool call
#: is caught structurally rather than trusted.
VALIDATE_ROW = partial(validate_conversation_row, tool_arguments=TOOL_ARGUMENTS)


def _banned_wording_check(splits: Splits) -> Mapping[str, Any]:
    """Banned wording anywhere in a trainable transcript.

    ``text_fields=()`` and ``message_fields=("messages",)``: the rendered
    transcript is the whole trainable surface, so scanning it covers context
    turns, the user turn and the final in one pass -- and ``messages`` is
    derived, so it cannot go stale relative to what is emitted.
    """
    return banned_wording_leaks(
        splits,
        BANNED_WORDING,
        text_fields=(),
        message_fields=("messages",),
        trainable_splits=TRAINABLE_SPLITS,
    )


def _duplicate_user_text_check(splits: Splits) -> Mapping[str, Any]:
    """Global normalized uniqueness of ``user_text``, pairs exempted.

    Wired through ``pre_dedup_checks``: ``compose`` deduplicates on ``text``
    before it builds the report, so by report time a within-split duplicate has
    already been dropped and a report-time check would pass vacuously.
    """
    return duplicate_text_leaks(splits, "user_text", exempt=GOVERNED_PAIR_EXEMPTION)


def _duplicate_final_response_check(splits: Splits) -> Mapping[str, Any]:
    """Global normalized uniqueness of ``final_response``, no exemption.

    Deliberately unexempted: the two rows of a counterfactual pair share an
    utterance on purpose, but they exist to be answered *differently*, so two
    identical finals are a defect even there.
    """
    return duplicate_text_leaks(splits, "final_response")


# Shared between compose() and the post-teacher, post-scrub build_report()
# rebuild, so the report that gates the release is computed exactly the way the
# pre-teacher one was -- just over the rows actually being emitted.
REPORT_KWARGS: dict[str, Any] = {
    "text_field": "text",
    "secondary_leak_fields": ("user_text",),
    "pii_fields": ("messages",),
    "extra_leak_checks": (_banned_wording_check, _duplicate_final_response_check),
}

#: The teacher-batch rules. Every one of them is generic; nothing here knows
#: about banking. They accumulate: one bad batch reports every affected row
#: instead of raising on the first.
CHECKER_RULES = (
    hash_pinned(),
    fields_present(EDITABLE_FIELDS),
    no_extra_keys({"record_id", "immutable_hash", "fields"}),
    min_words("final_response", 7),
    banned_pattern("final_response", BANNED_WORDING),
    unique_normalized(
        "final_response",
        record_value=lambda record: record.get("final_response"),
        exempt=GOVERNED_PAIR_EXEMPTION,
    ),
    opening_ngram_cap("final_response", lambda pair: str(pair.record.get("example_kind", ""))),
)


def _stub_teacher_rewrite(record_id: str, final_response: str) -> str:
    """A deterministic stand-in for an LLM teacher: wording only, no new facts.

    Takes ``record_id`` as well as the text so a caller (a test) can make one
    row misbehave and watch the checker catch it.
    """
    text = final_response if final_response.endswith(".") else f"{final_response}."
    return f"{text} {TEACHER_CLOSER}"


def build(
    output_dir: Path,
    *,
    seed_splits: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    teacher_rewrite: Callable[[str, str], str] = _stub_teacher_rewrite,
) -> dict[str, Any]:
    """Compose, realize, audit, scrub, and emit the tool-calling dataset.

    ``seed_splits`` and ``teacher_rewrite`` exist so the end-to-end test can
    inject a bad row or a bad teacher; the defaults are the real build.
    """
    seeds: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_ORDER}
    for split, rows in (seed_splits or {}).items():
        seeds[split] = [dict(row) for row in rows]

    splits, report = compose(
        seed_splits=seeds,
        registry=REGISTRY,
        split_order=SPLIT_ORDER,
        pre_dedup_checks=(_duplicate_user_text_check,),
        **REPORT_KWARGS,
    )

    # Frozen splits are never sent to a teacher: their wording is the
    # regression, and rewriting it would rewrite the test.
    teacher_records = [row for split in TRAINABLE_SPLITS for row in splits[split]]
    all_records = [row for split in SPLIT_ORDER for row in splits[split]]

    requests_path = output_dir / "teacher_requests.jsonl"
    responses_path = output_dir / "teacher_responses.jsonl"
    export_teacher_requests(
        teacher_records,
        requests_path,
        editable_fields=EDITABLE_FIELDS,
        derived_fields=CONVERSATION_DERIVED_FIELDS,
        rederive=rederive_conversation,
        validate=VALIDATE_ROW,
        instructions=(
            "Rewrite only final_response, in the voice of examples/banking/voice_spec.md. "
            "Do not change any fact, tool call, or decision."
        ),
    )
    _write_stub_teacher_responses(requests_path, responses_path, teacher_rewrite)

    _scrub_context_wording(splits)

    findings = check_teacher_batch(requests_path, responses_path, all_records, CHECKER_RULES)
    if findings:
        raise ValueError(
            f"teacher batch check failed ({len(findings)} findings):\n" + format_findings(findings)
        )

    realized = import_teacher_responses(
        teacher_records,
        responses_path,
        editable_fields=EDITABLE_FIELDS,
        teacher_model=TEACHER_MODEL,
        # The prompt spec *and* the exact request file it was sent with: change
        # either and every row realized afterwards records a different hash.
        teacher_prompt_hash=compute_teacher_prompt_hash(VOICE_SPEC_PATH, requests_path),
        # The requests were pinned before the scrub moved the hash. The stamp
        # scrub_fields left behind is what lets this file still apply -- and it
        # is a trust statement about this pipeline, not a proof (see README).
        accept_pre_scrub_hashes=True,
        derived_fields=CONVERSATION_DERIVED_FIELDS,
        rederive=rederive_conversation,
        validate=VALIDATE_ROW,
    )
    realized_by_id = {row["record_id"]: row for row in realized}
    for rows in splits.values():
        for index, row in enumerate(rows):
            if row["record_id"] in realized_by_id:
                rows[index] = realized_by_id[row["record_id"]]

    # The scrub and the teacher pass both changed row content since compose
    # built its report; gating on that stale report would gate on bytes that
    # are no longer being emitted (and write_dataset would reject it anyway on
    # the splits fingerprint).
    report = build_report(
        splits,
        cross_split_duplicates_removed=report["cross_split_duplicates_removed"],
        within_split_duplicates_removed=report["within_split_duplicates_removed"],
        **REPORT_KWARGS,
    )

    manifest = write_dataset(
        output_dir,
        splits,
        split_order=SPLIT_ORDER,
        manifest_extra={
            "format_version": 1,
            "row_shape": "conversation",
            "tools": sorted(TOOL_ARGUMENTS),
            "lane_labels": list(LANE_LABELS),
            "action_labels": list(ACTION_LABELS),
            "entity_resolution_labels": list(ENTITY_RESOLUTION_LABELS),
            "teacher_voice_spec": VOICE_SPEC_PATH.name,
        },
        report=report,
        created_at=CREATED_AT,
        data_card_lines=_data_card,
        allowed_use={
            "train": ["training"],
            "validation": ["evaluation"],
            "test": ["evaluation"],
        },
    )
    write_source_lock(output_dir / "source.lock.json", manifest)
    return manifest


def _scrub_context_wording(splits: Mapping[str, list[dict[str, Any]]]) -> None:
    """Scrub banned wording out of one train row's context, after export.

    ``context_messages`` is not editable, which is the whole reason this is a
    scrub and not a teacher instruction: ``scrub_fields`` refuses to touch a
    field the teacher owns. Because the field *is* hashed, the substitution
    moves the row's immutable hash and a pre-scrub stamp is written.
    """
    for row in splits["train"]:
        if row["record_id"] != SCRUB_RECORD_ID:
            continue
        changed = scrub_fields(
            row,
            SCRUB_SUBSTITUTIONS,
            fields=("context_messages",),
            editable_fields=EDITABLE_FIELDS,
            derived_fields=CONVERSATION_DERIVED_FIELDS,
            rederive=rederive_conversation,
            validate=VALIDATE_ROW,
        )
        if not changed:
            raise ValueError(f"{SCRUB_RECORD_ID}: scrub matched nothing -- has the row changed?")
        return
    raise ValueError(f"{SCRUB_RECORD_ID} is not in the train split")


def _write_stub_teacher_responses(
    requests_path: Path,
    responses_path: Path,
    teacher_rewrite: Callable[[str, str], str],
) -> None:
    with (
        requests_path.open(encoding="utf-8") as source,
        responses_path.open("w", encoding="utf-8", newline="\n") as sink,
    ):
        for line in source:
            request = json.loads(line)
            response = {
                "record_id": request["record_id"],
                "immutable_hash": request["immutable_hash"],
                "fields": {
                    field: teacher_rewrite(request["record_id"], value)
                    for field, value in request["fields"].items()
                },
            }
            sink.write(json.dumps(response, sort_keys=True) + "\n")


def _data_card(manifest: dict[str, Any]) -> list[str]:
    counts = manifest["report"]["split_counts"]
    return [
        "# Banking Tool-Calling Example Dataset",
        "",
        "A tiny, synthetic worked example of governed tool-calling conversation",
        "rows built with the dataforge framework. Not derived from, and not a",
        "substitute for, any production banking dataset.",
        "",
        f"- Train rows: {counts['train']}",
        f"- Validation rows: {counts['validation']}",
        f"- Test rows (frozen, never teacher-realized): {counts['test']}",
        f"- Tools: {', '.join(manifest['tools'])}",
        f"- Teacher: {TEACHER_MODEL} against {manifest['teacher_voice_spec']}",
        "",
        "Only `final_response` was ever editable. Context turns, tool calls and",
        "tool results are covered by each row's immutable hash.",
        "",
    ]


def main() -> int:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR
    manifest = build(output_dir)
    counts = manifest["report"]["split_counts"]
    print(f"wrote {output_dir} (train={counts['train']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
