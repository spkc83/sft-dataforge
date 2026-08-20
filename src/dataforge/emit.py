"""Dataset directory emission: jsonl, manifest, data card, source lock."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from dataforge.curricula import REPORT_CONTRACT, splits_fingerprint
from dataforge.rows import canonical_json_bytes

__all__ = [
    "canonical_json_bytes",
    "default_gates",
    "file_sha256",
    "rows_jsonl_bytes",
    "rows_sha256",
    "verify_release_split_digests",
    "write_dataset",
    "write_source_lock",
]


def rows_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(rows_jsonl_bytes(rows)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_gates(report: Mapping[str, Any]) -> None:
    """Abort on any nonzero PII match count or nonempty leakage finding.

    Any ``report["leakage"]`` key ending in ``_leak_count`` must be zero, and
    any key ending in ``_leaks`` must be empty. This lets new leakage checks
    added to :mod:`dataforge.guards` gate automatically without touching
    this function.
    """
    pii_matches = report.get("pii_matches", 0)
    if pii_matches:
        raise ValueError(f"dataset contains {pii_matches} PII-like matches")
    leakage = report.get("leakage", {})
    for key, value in leakage.items():
        if key.endswith("_leak_count") and value:
            raise ValueError(f"dataset has {key}={value}")
        if key.endswith("_leaks") and value:
            raise ValueError(f"dataset has nonempty {key}")


def write_dataset(
    output_dir: Path,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    manifest_extra: Mapping[str, Any],
    report: Mapping[str, Any],
    created_at: str,
    data_card_lines: Sequence[str] | Callable[[dict[str, Any]], Sequence[str]],
    split_order: Sequence[str] = ("train", "validation", "test"),
    gates: Sequence[Callable[[Mapping[str, Any]], None]] = (default_gates,),
    allowed_use: Mapping[str, list[str]] | None = None,
    expected_contract: str | None = REPORT_CONTRACT,
    row_format_version: int = 1,
) -> dict[str, Any]:
    """Write ``{split}.jsonl`` + ``manifest.json`` + ``README.md`` for a dataset.

    Before writing anything: if ``expected_contract`` is set (the default is
    :data:`dataforge.curricula.REPORT_CONTRACT`), ``report["contract"]`` must
    match it, so an unrecognized/malformed report can never gate a release
    vacuously. Whenever that contract check applies and passes, ``report``
    must also carry a ``splits_fingerprint`` -- :func:`dataforge.curricula.build_report`
    always sets one, so its *absence* on an otherwise-conforming report is
    itself treated as an error rather than silently skipping the check (a
    report shaped like a fresh one but missing just that one key must not be
    trusted as fresh). The fingerprint must match one freshly recomputed
    over the exact ``splits`` being written -- this is what makes gating on
    a stale report (e.g. one built before a later teacher-realization pass
    edited row content) a hard error rather than a silent bypass. With
    ``expected_contract=None`` (a deliberate opt-out for a bespoke report
    shape), a present ``splits_fingerprint`` is still verified but is no
    longer required. Finally every ``gates`` callable runs against
    ``report``; a gate raises to abort the release.

    ``created_at`` is required rather than defaulted to "now" so that
    callers control (and can make deterministic) the one field in the
    manifest that would otherwise vary run to run.
    """
    if expected_contract is not None:
        if report.get("contract") != expected_contract:
            raise ValueError(
                f"report contract {report.get('contract')!r} does not match "
                f"expected {expected_contract!r}"
            )
        if "splits_fingerprint" not in report:
            raise ValueError(
                "report contract matches but splits_fingerprint is missing -- "
                "dataforge.curricula.build_report always sets one, so its absence means this "
                "report cannot be trusted as fresh. Rebuild it with build_report(), or pass "
                "expected_contract=None if this report is deliberately not build_report-shaped."
            )
    expected_fingerprint = report.get("splits_fingerprint")
    if expected_fingerprint is not None and splits_fingerprint(splits) != expected_fingerprint:
        raise ValueError(
            "report is stale: its splits_fingerprint does not match the splits being "
            "written. Rebuild the report (e.g. dataforge.curricula.build_report) on the "
            "exact rows you are about to emit, including any post-teacher-realization edits."
        )
    for gate in gates:
        gate(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_entries = []
    for split in split_order:
        rows = splits[split]
        path = output_dir / f"{split}.jsonl"
        path.write_bytes(rows_jsonl_bytes(rows))
        split_entries.append(
            {
                "name": split,
                "path": path.name,
                "rows": len(rows),
                "sha256": rows_sha256(rows),
                "bytes": path.stat().st_size,
                "allowed_use": (allowed_use or {}).get(split, []),
            }
        )

    manifest = {
        "contract": "dataforge-dataset-manifest",
        "row_format_version": row_format_version,
        "created_at": created_at,
        "splits": split_entries,
        "report": report,
        **manifest_extra,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_json, encoding="utf-8")

    lines = data_card_lines(manifest) if callable(data_card_lines) else data_card_lines
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def write_source_lock(
    path: Path,
    manifest: Mapping[str, Any],
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    lock = {
        "contract": "dataforge-source-lock",
        "created_at": manifest["created_at"],
        "prepared_split_sha256": {entry["name"]: entry["sha256"] for entry in manifest["splits"]},
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_release_split_digests(
    split_entries: Sequence[Mapping[str, Any]],
    release_lock: Mapping[str, Any],
    *,
    split_order: Sequence[str] | None = None,
) -> None:
    """Verify every split's digest against a release lock.

    Iterates the canonical split list -- ``split_order`` if given, otherwise
    the names in ``split_entries`` -- rather than ``release_lock``'s own
    keys, so a lock that is silently missing a split's digest (or carries an
    empty ``prepared_split_sha256``) is rejected instead of vacuously
    "verifying" nothing for that split.
    """
    expected = release_lock.get("prepared_split_sha256")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("release lock is missing a non-empty prepared_split_sha256")
    actual = {str(entry["name"]): str(entry["sha256"]) for entry in split_entries}
    canonical_splits = (
        tuple(split_order) if split_order is not None else tuple(actual.keys())
    )
    missing = [split for split in canonical_splits if split not in expected]
    if missing:
        raise ValueError(f"release lock is missing digests for splits: {missing}")
    for split in canonical_splits:
        expected_digest = expected[split]
        if actual.get(split) != expected_digest:
            raise ValueError(
                f"{split} split digest drift: expected {expected_digest}, got {actual.get(split)}"
            )
