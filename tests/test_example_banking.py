from __future__ import annotations

import json
from pathlib import Path

from examples.banking.build import build


def test_banking_example_builds_end_to_end(tmp_path: Path) -> None:
    manifest = build(tmp_path / "run1")
    assert manifest["report"]["pii_matches"] == 0
    assert manifest["report"]["leakage"]["group_split_leak_count"] == 0
    assert manifest["report"]["leakage"]["heldout_exact_leaks"] == []
    assert manifest["report"]["leakage"]["heldout_ngram_leaks"] == []
    for split in ("train", "validation", "test"):
        assert (tmp_path / "run1" / f"{split}.jsonl").exists()
        assert manifest["report"]["split_counts"][split] > 0


def test_banking_example_is_deterministic(tmp_path: Path) -> None:
    build(tmp_path / "run1")
    build(tmp_path / "run2")
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json", "README.md"):
        assert (tmp_path / "run1" / name).read_bytes() == (tmp_path / "run2" / name).read_bytes()


def test_banking_example_teacher_realization_applied(tmp_path: Path) -> None:
    manifest = build(tmp_path / "run1")
    assert manifest is not None
    rows = [
        json.loads(line)
        for split in ("train", "validation", "test")
        for line in (tmp_path / "run1" / f"{split}.jsonl").read_text().splitlines()
    ]
    realized = [row for row in rows if "assistant_response" in row]
    assert realized
    for row in realized:
        assert row["assistant_response"].endswith(".")
        assert row["provenance"]["teacher_model"] == "stub-teacher-v1"
