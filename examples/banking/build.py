"""Build the example banking dataset end-to-end.

Run with ``python -m examples.banking.build [output_dir]``. Deterministic:
two runs produce byte-identical jsonl/manifest output (a fixed
``created_at`` and no randomness anywhere in the curricula or the stub
teacher).
"""

from __future__ import annotations

import sys
from pathlib import Path

from dataforge.curricula import build_report, compose
from dataforge.emit import write_dataset, write_source_lock
from dataforge.rows import validate_row_consistency
from dataforge.teacher import export_teacher_requests, import_teacher_responses
from examples.banking.curricula import HELD_OUT_TEXTS, REGISTRY
from examples.banking.taxonomy import (
    ACTION_LABELS,
    DOMAIN_LABELS,
    ENTITY_RESOLUTION_LABELS,
    LANE_LABELS,
)

DEFAULT_OUTPUT_DIR = Path("dist/banking-example")
CREATED_AT = "2024-01-01T00:00:00+00:00"
TEACHER_MODEL = "stub-teacher-v1"
TEACHER_PROMPT_HASH = "sha256:stub-prompt"
EDITABLE_FIELDS = ("assistant_response",)

# Shared between compose() and the post-teacher build_report() re-run, so the
# report gating the release is computed the same way the pre-teacher report
# was -- just over the rows actually being emitted.
REPORT_KWARGS: dict = {
    "held_out_texts": HELD_OUT_TEXTS,
    "heldout_excluded_splits": ("test",),
}


def _stub_teacher_rewrite(text: str) -> str:
    """A deterministic stand-in for an LLM teacher: wording only, no facts added."""
    return text if text.endswith(".") else f"{text}."


def build(output_dir: Path) -> dict:
    splits, report = compose(
        seed_splits={"train": [], "validation": [], "test": []},
        registry=REGISTRY,
        **REPORT_KWARGS,
    )

    for rows in splits.values():
        for row in rows:
            row["record_id"] = row["group_id"]

    teacher_records = [
        row for rows in splits.values() for row in rows if "assistant_response" in row
    ]
    requests_path = output_dir / "teacher_requests.jsonl"
    responses_path = output_dir / "teacher_responses.jsonl"
    export_teacher_requests(teacher_records, requests_path, editable_fields=EDITABLE_FIELDS)

    _write_stub_teacher_responses(requests_path, responses_path)
    realized = import_teacher_responses(
        teacher_records,
        responses_path,
        editable_fields=EDITABLE_FIELDS,
        teacher_model=TEACHER_MODEL,
        teacher_prompt_hash=TEACHER_PROMPT_HASH,
        # assistant_response doesn't feed any derived field here, but wiring
        # a structural validator is cheap and catches drift if that ever
        # changes (hello-SLM's validate_records analogue).
        validate=validate_row_consistency,
    )
    realized_by_id = {row["record_id"]: row for row in realized}
    for rows in splits.values():
        for index, row in enumerate(rows):
            if row["record_id"] in realized_by_id:
                rows[index] = realized_by_id[row["record_id"]]

    # The teacher pass above can change any editable field's content (here,
    # assistant_response wording) -- and, more generally, could introduce
    # PII or a held-out leak into a field the original (pre-teacher) report
    # never scanned as changed. Gating write_dataset on the STALE report
    # would describe bytes that are no longer what's being emitted; rebuild
    # it on the realized splits before writing anything.
    report = build_report(
        splits,
        cross_split_duplicates_removed=report["cross_split_duplicates_removed"],
        **REPORT_KWARGS,
    )

    manifest = write_dataset(
        output_dir,
        splits,
        manifest_extra={
            "format_version": 1,
            "domain_labels": list(DOMAIN_LABELS),
            "lane_labels": list(LANE_LABELS),
            "action_labels": list(ACTION_LABELS),
            "entity_resolution_labels": list(ENTITY_RESOLUTION_LABELS),
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


def _write_stub_teacher_responses(requests_path: Path, responses_path: Path) -> None:
    import json

    with requests_path.open(encoding="utf-8") as source, responses_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as sink:
        for line in source:
            request = json.loads(line)
            rewritten_fields = {
                field: _stub_teacher_rewrite(value) for field, value in request["fields"].items()
            }
            response = {
                "record_id": request["record_id"],
                "immutable_hash": request["immutable_hash"],
                "fields": rewritten_fields,
            }
            sink.write(json.dumps(response, sort_keys=True) + "\n")


def _data_card(manifest: dict) -> list[str]:
    counts = manifest["report"]["split_counts"]
    return [
        "# Banking Example Dataset",
        "",
        "A tiny, synthetic worked example built with the dataforge framework.",
        "Not derived from, and not a substitute for, any production banking dataset.",
        "",
        f"- Train rows: {counts['train']}",
        f"- Validation rows: {counts['validation']}",
        f"- Test rows: {counts['test']}",
        f"- Domain labels: {', '.join(manifest['domain_labels'])}",
        f"- Lane labels: {', '.join(manifest['lane_labels'])}",
        f"- Action labels: {', '.join(manifest['action_labels'])}",
        f"- Entity-resolution labels: {', '.join(manifest['entity_resolution_labels'])}",
        "",
    ]


def main() -> int:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR
    manifest = build(output_dir)
    print(f"wrote {output_dir} (train={manifest['report']['split_counts']['train']} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
