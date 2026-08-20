"""Dataset directory emission: jsonl, manifest, data card, source lock."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


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
) -> dict[str, Any]:
    """Write ``{split}.jsonl`` + ``manifest.json`` + ``README.md`` for a dataset.

    Runs every ``gates`` callable against ``report`` before writing anything;
    a gate raises to abort the release. ``created_at`` is required rather
    than defaulted to "now" so that callers control (and can make
    deterministic) the one field in the manifest that would otherwise vary
    run to run.
    """
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
) -> None:
    expected = release_lock.get("prepared_split_sha256")
    if not isinstance(expected, Mapping):
        raise ValueError("release lock is missing prepared_split_sha256")
    actual = {str(entry["name"]): str(entry["sha256"]) for entry in split_entries}
    for split, expected_digest in expected.items():
        if actual.get(split) != expected_digest:
            raise ValueError(
                f"{split} split digest drift: expected {expected_digest}, got {actual.get(split)}"
            )
