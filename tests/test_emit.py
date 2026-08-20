from __future__ import annotations

from pathlib import Path

import pytest

from dataforge.curricula import REPORT_CONTRACT, build_report
from dataforge.emit import (
    default_gates,
    rows_jsonl_bytes,
    rows_sha256,
    verify_release_split_digests,
    write_dataset,
    write_source_lock,
)


def _clean_report(splits: dict) -> dict:
    return build_report(splits, secondary_leak_fields=())


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
    manifest = write_dataset(
        tmp_path,
        splits,
        manifest_extra={"schema": "unit-test"},
        report=_clean_report(splits),
        created_at="2024-01-01T00:00:00+00:00",
        data_card_lines=["# Dataset"],
    )
    for split, rows in splits.items():
        path = tmp_path / f"{split}.jsonl"
        assert path.read_bytes() == rows_jsonl_bytes(rows)
    entry = next(entry for entry in manifest["splits"] if entry["name"] == "train")
    assert entry["sha256"] == rows_sha256(splits["train"])
    assert (tmp_path / "README.md").read_text().startswith("# Dataset")
    assert manifest["row_format_version"] == 1


def test_write_dataset_row_format_version_is_configurable(tmp_path: Path) -> None:
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    manifest = write_dataset(
        tmp_path,
        splits,
        manifest_extra={},
        report=_clean_report(splits),
        created_at="2024-01-01T00:00:00+00:00",
        data_card_lines=["# x"],
        row_format_version=2,
    )
    assert manifest["row_format_version"] == 2


def test_write_dataset_aborts_on_gate_failure(tmp_path: Path) -> None:
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    report = {**_clean_report(splits), "pii_matches": 1}
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


def test_write_dataset_rejects_report_with_wrong_contract(tmp_path: Path) -> None:
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    with pytest.raises(ValueError, match="contract"):
        write_dataset(
            tmp_path,
            splits,
            manifest_extra={},
            report={"totally": "bogus"},
            created_at="2024-01-01T00:00:00+00:00",
            data_card_lines=["# x"],
        )
    assert not (tmp_path / "train.jsonl").exists()


def test_write_dataset_expected_contract_none_opts_out(tmp_path: Path) -> None:
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    manifest = write_dataset(
        tmp_path,
        splits,
        manifest_extra={},
        report={"pii_matches": 0, "leakage": {}},
        created_at="2024-01-01T00:00:00+00:00",
        data_card_lines=["# x"],
        expected_contract=None,
    )
    assert manifest["contract"] == "dataforge-dataset-manifest"


def test_write_dataset_rejects_matching_contract_missing_fingerprint(tmp_path: Path) -> None:
    """N6: a report whose contract matches but which is missing
    splits_fingerprint entirely must be refused, not silently treated as
    unverifiable-but-fine -- build_report always sets one, so its absence
    on an otherwise-conforming report is itself suspicious."""
    splits = {"train": [{"text": "a", "group_id": "g1"}], "validation": [], "test": []}
    report = _clean_report(splits)
    del report["splits_fingerprint"]
    with pytest.raises(ValueError, match="splits_fingerprint"):
        write_dataset(
            tmp_path,
            splits,
            manifest_extra={},
            report=report,
            created_at="2024-01-01T00:00:00+00:00",
            data_card_lines=["# x"],
        )
    assert not (tmp_path / "train.jsonl").exists()


def test_write_dataset_expected_contract_none_still_checks_present_fingerprint(
    tmp_path: Path,
) -> None:
    """Opting out of the contract check (expected_contract=None) doesn't
    disable the fingerprint check when a fingerprint IS present -- only its
    presence stops being mandatory."""
    splits = {"train": [{"text": "a", "group_id": "g1"}], "validation": [], "test": []}
    report = _clean_report(splits)
    splits["train"][0]["text"] = "mutated after the report was computed"
    with pytest.raises(ValueError, match="stale"):
        write_dataset(
            tmp_path,
            splits,
            manifest_extra={},
            report=report,
            created_at="2024-01-01T00:00:00+00:00",
            data_card_lines=["# x"],
            expected_contract=None,
        )


def test_write_dataset_rejects_stale_report_fingerprint(tmp_path: Path) -> None:
    splits = {"train": [{"text": "a", "group_id": "g1"}], "validation": [], "test": []}
    report = _clean_report(splits)
    # Mutate splits after the report was built (e.g. a teacher realization
    # pass) without rebuilding the report -- the report is now stale.
    splits["train"][0]["text"] = "a mutated after the report was computed"
    with pytest.raises(ValueError, match="stale"):
        write_dataset(
            tmp_path,
            splits,
            manifest_extra={},
            report=report,
            created_at="2024-01-01T00:00:00+00:00",
            data_card_lines=["# x"],
        )
    assert not (tmp_path / "train.jsonl").exists()


def test_write_dataset_accepts_freshly_rebuilt_report_after_mutation(tmp_path: Path) -> None:
    splits = {"train": [{"text": "a", "group_id": "g1"}], "validation": [], "test": []}
    splits["train"][0]["text"] = "a mutated before the report was computed"
    report = _clean_report(splits)  # built AFTER the mutation: fresh, not stale
    write_dataset(
        tmp_path,
        splits,
        manifest_extra={},
        report=report,
        created_at="2024-01-01T00:00:00+00:00",
        data_card_lines=["# x"],
    )
    assert (tmp_path / "train.jsonl").read_bytes() == rows_jsonl_bytes(splits["train"])


def test_source_lock_round_trip(tmp_path: Path) -> None:
    splits = {"train": [{"text": "a", "group_id": "g1"}], "validation": [], "test": []}
    manifest = write_dataset(
        tmp_path,
        splits,
        manifest_extra={},
        report=_clean_report(splits),
        created_at="2024-01-01T00:00:00+00:00",
        data_card_lines=["# x"],
    )
    lock_path = tmp_path / "source.lock.json"
    write_source_lock(lock_path, manifest)

    import json

    lock = json.loads(lock_path.read_text())
    verify_release_split_digests(manifest["splits"], lock)  # no raise


def test_verify_release_split_digests_detects_drift(tmp_path: Path) -> None:
    splits = {"train": [{"text": "a", "group_id": "g1"}], "validation": [], "test": []}
    manifest = write_dataset(
        tmp_path,
        splits,
        manifest_extra={},
        report=_clean_report(splits),
        created_at="2024-01-01T00:00:00+00:00",
        data_card_lines=["# x"],
    )
    lock = {
        "prepared_split_sha256": {
            "train": "0" * 64,
            "validation": rows_sha256(splits["validation"]),
            "test": rows_sha256(splits["test"]),
        }
    }
    with pytest.raises(ValueError, match="digest drift"):
        verify_release_split_digests(manifest["splits"], lock)


def test_verify_release_split_digests_rejects_missing_split() -> None:
    split_entries = [
        {"name": "train", "sha256": "a" * 64},
        {"name": "validation", "sha256": "b" * 64},
        {"name": "test", "sha256": "c" * 64},
    ]
    lock = {"prepared_split_sha256": {"train": "a" * 64, "validation": "b" * 64}}  # missing test
    with pytest.raises(ValueError, match="missing digests"):
        verify_release_split_digests(split_entries, lock)


def test_verify_release_split_digests_rejects_empty_digest_map() -> None:
    split_entries = [{"name": "train", "sha256": "a" * 64}]
    with pytest.raises(ValueError, match="non-empty"):
        verify_release_split_digests(split_entries, {"prepared_split_sha256": {}})


def test_verify_release_split_digests_uses_explicit_split_order() -> None:
    split_entries = [{"name": "train", "sha256": "a" * 64}]
    lock = {"prepared_split_sha256": {"train": "a" * 64}}  # missing validation/test
    with pytest.raises(ValueError, match="missing digests"):
        verify_release_split_digests(
            split_entries, lock, split_order=("train", "validation", "test")
        )


def test_report_contract_constant_matches_build_report() -> None:
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    assert _clean_report(splits)["contract"] == REPORT_CONTRACT
