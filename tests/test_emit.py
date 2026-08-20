from __future__ import annotations

from pathlib import Path

import pytest

from dataforge.emit import (
    default_gates,
    rows_jsonl_bytes,
    rows_sha256,
    verify_release_split_digests,
    write_dataset,
    write_source_lock,
)


def test_rows_jsonl_bytes_canonical_sorted_keys() -> None:
    rows = [{"b": 1, "a": 2}]
    payload = rows_jsonl_bytes(rows)
    assert payload == b'{"a":2,"b":1}\n'


def test_rows_sha256_matches_bytes_digest() -> None:
    import hashlib

    rows = [{"x": 1}]
    assert rows_sha256(rows) == hashlib.sha256(rows_jsonl_bytes(rows)).hexdigest()


def test_default_gates_raise_on_pii() -> None:
    with pytest.raises(ValueError, match="PII"):
        default_gates({"pii_matches": 3, "leakage": {}})


def test_default_gates_raise_on_leak_count() -> None:
    with pytest.raises(ValueError):
        default_gates({"pii_matches": 0, "leakage": {"group_split_leak_count": 1}})


def test_default_gates_raise_on_nonempty_leak_list() -> None:
    with pytest.raises(ValueError):
        default_gates({"pii_matches": 0, "leakage": {"heldout_exact_leaks": [{"x": 1}]}})


def test_default_gates_pass_when_clean() -> None:
    default_gates(
        {
            "pii_matches": 0,
            "leakage": {"group_split_leak_count": 0, "heldout_exact_leaks": []},
        }
    )


def test_write_dataset_manifest_and_digests_round_trip(tmp_path: Path) -> None:
    splits = {
        "train": [{"text": "a", "group_id": "g1"}],
        "validation": [{"text": "b", "group_id": "g2"}],
        "test": [{"text": "c", "group_id": "g3"}],
    }
    report = {"pii_matches": 0, "leakage": {"group_split_leak_count": 0}}
    manifest = write_dataset(
        tmp_path,
        splits,
        manifest_extra={"schema": "unit-test"},
        report=report,
        created_at="2024-01-01T00:00:00+00:00",
        data_card_lines=["# Dataset"],
    )
    for split, rows in splits.items():
        path = tmp_path / f"{split}.jsonl"
        assert path.read_bytes() == rows_jsonl_bytes(rows)
    entry = next(entry for entry in manifest["splits"] if entry["name"] == "train")
    assert entry["sha256"] == rows_sha256(splits["train"])
    assert (tmp_path / "README.md").read_text().startswith("# Dataset")


def test_write_dataset_aborts_on_gate_failure(tmp_path: Path) -> None:
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    report = {"pii_matches": 1, "leakage": {}}
    with pytest.raises(ValueError, match="PII"):
        write_dataset(
            tmp_path,
            splits,
            manifest_extra={},
            report=report,
            created_at="2024-01-01T00:00:00+00:00",
            data_card_lines=["# x"],
        )
    assert not (tmp_path / "train.jsonl").exists()


def test_source_lock_round_trip_and_digest_drift(tmp_path: Path) -> None:
    splits = {"train": [{"text": "a", "group_id": "g1"}], "validation": [], "test": []}
    manifest = write_dataset(
        tmp_path,
        splits,
        manifest_extra={},
        report={"pii_matches": 0, "leakage": {}},
        created_at="2024-01-01T00:00:00+00:00",
        data_card_lines=["# x"],
    )
    lock_path = tmp_path / "source.lock.json"
    write_source_lock(lock_path, manifest)

    import json

    lock = json.loads(lock_path.read_text())
    verify_release_split_digests(manifest["splits"], lock)  # no raise

    drifted = {"prepared_split_sha256": {"train": "0" * 64}}
    with pytest.raises(ValueError, match="digest drift"):
        verify_release_split_digests(manifest["splits"], drifted)
