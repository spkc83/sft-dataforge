# dataforge

Governed instruction/SFT dataset construction for classifier and
tool-calling fine-tuning in customer-servicing domains: a small, declarative
framework for building label taxonomies, composing curricula into
train/validation/test splits, guarding against PII and cross-split leakage,
emitting a locked+hashed dataset directory, and letting an LLM teacher
improve wording without touching labels.

It is a generalization of the SFT-data-construction machinery used to build
a retail-banking conversation router: the taxonomy, row shape, leakage
guards, emission format, and teacher-hash discipline are the same; the
banking-specific constants (intents, tools, curricula) are not part of the
library and live only in `examples/banking` as a worked example.

## Design principles

- **The deterministic skeleton owns labels.** A `Taxonomy` is a declarative
  dict (dimensions, an intent-to-hierarchy mapping, and the legal
  `(action, entity_resolution)` pairs per lane) rather than a python module
  full of banking constants. Curricula compute *which* label a given example
  gets; the taxonomy only validates that the result is internally
  consistent. This keeps label logic auditable and testable independent of
  any specific domain.
- **An LLM teacher owns wording, never labels, under hash discipline.**
  `dataforge.teacher` lets a caller export a per-record hash over everything
  *except* a declared set of editable text fields, hand those fields to a
  teacher model, and reject any response whose reapplication changes that
  hash. A teacher can rephrase; it cannot relabel.
- **Eval splits win deduplication.** When curricula produce the same
  (normalized) text in more than one split, `dataforge.curricula.compose`
  keeps the copy in the split highest in `dedup_priority` -- by default,
  test beats validation beats train -- so a train/eval collision never
  silently inflates apparent performance.
- **Gates before emission, not after.** `dataforge.emit.write_dataset` runs
  its gate functions against the composed report *before* writing any file.
  The default gate aborts on any nonzero PII match count or nonempty
  leakage finding; new leakage checks in `dataforge.guards` gate
  automatically because the default gate walks the report by key suffix
  (`_leak_count`, `_leaks`) rather than an enumerated list.
- **Locks and manifests for provenance.** Every emitted split gets a
  canonical (sorted-key) jsonl encoding and a sha256 in `manifest.json`.
  `dataforge.emit.write_source_lock` / `verify_release_split_digests` let a
  pipeline pin a release's split digests and detect drift on the next
  build.

## Package map

| Module | Responsibility |
| --- | --- |
| `dataforge/taxonomy.py` | `Taxonomy` dataclass (built from a dict/JSON): intent hierarchy, legal action/entity-resolution pairs per lane, tool compatibility. `labels_for_example` / `validate_hierarchical_labels`. |
| `dataforge/rows.py` | The row contract: `render_context` (the `[PRIOR_STATE]` / `[CURRENT_USER]` / `[PREVIOUS_ASSISTANT]` / `[PREVIOUS_USER]` flattening), `make_row` (labels + multi-hot relations + provenance), `normalize_text`. |
| `dataforge/curricula.py` | `@curriculum` registration on a `Registry`; `compose` runs every curriculum per split, enforces group/trajectory/pair non-leakage, deduplicates with eval-wins ordering, and produces the governance report. |
| `dataforge/guards.py` | PII regexes, held-out exact + n-gram leak detection, `leakage_report`. |
| `dataforge/emit.py` | Canonical jsonl encoding, `write_dataset` (gates, split files, manifest, data card), `write_source_lock`, `verify_release_split_digests`. |
| `dataforge/teacher.py` | `immutable_hash`, `export_teacher_requests`, `import_teacher_responses` -- the wording-only teacher realization harness. |

## Quickstart

The worked example in `examples/banking` instantiates every piece of the
framework against a small synthetic taxonomy (`view_balance`, `freeze_card`,
`policy_faq`, and out-of-scope refusal) with five tiny curricula, including
one multiturn example and one held-out regression guard. Build it:

```bash
uv pip install -e ".[dev]"
uv run python -m examples.banking.build dist/banking-example
```

This writes `dist/banking-example/{train,validation,test}.jsonl`,
`manifest.json`, `README.md`, `source.lock.json`, and the teacher
request/response jsonl pair. The build is deterministic: running it twice
against different output directories produces byte-identical files.

Read `examples/banking/taxonomy.py` and `examples/banking/curricula.py` for
the shape of a real integration; `examples/banking/build.py` shows the full
compose -> teacher-realize -> emit pipeline, including a stubbed teacher.

## Development

```bash
uv pip install -e ".[dev]"
uv run ruff check .
uv run mypy
uv run pytest -q
```
