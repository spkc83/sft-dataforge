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
| `dataforge/teacher.py` | `immutable_hash`, `export_teacher_requests`, `import_teacher_responses`, `scrub_fields`, `compute_teacher_prompt_hash` -- the wording-only teacher realization harness. |
| `dataforge/checks.py` | The teacher-batch checker: `check_teacher_batch` over a `Batch` of request/response/record triples, and the generic rule factories (`hash_pinned`, `min_words`, `banned_pattern`, `unique_normalized`, `opening_ngram_cap`, ...). |

## v9 mechanisms

Five mechanisms carried over from the v9 banking build, each of them a place
where an earlier pipeline shipped a defect. They are library features, not
example code; `examples/banking/build_tool_calls.py` wires all five together
and `examples/banking/tool_curricula.py` supplies rows shaped to exercise them.

### Banned wording, with the frozen splits exempt

`guards.banned_wording_leaks(splits, pattern, ...)` flags banned phrasing in
trainable text. Two details matter. It scans only `trainable_splits`
(`("train", "validation")` by default), so a frozen evaluation row is exempt
**by construction** rather than by a special case at gate time -- a regression
row often has to contain exactly the phrasing the model must never learn to
produce. And `message_fields=("messages",)` scans the `content` of every
message in a rendered transcript whose role is user or assistant, so a
tool-call assistant turn (`content: None`) and the `system`/`tool` turns are
skipped while context turns, the current user turn and the final are all
covered. Its keys end in `_leaks`/`_leak_count`, so wiring it through
`extra_leak_checks` gates the build with no other change.

### Global normalized uniqueness, and the pre-dedup hook

`guards.duplicate_text_leaks(splits, field, ...)` buckets rows **globally** on
`rows.normalize_text_ascii` (lowercase, non-alphanumerics collapsed to spaces --
deliberately more aggressive than `normalize_text`), so a within-split
duplicate is flagged just as a cross-split one is. `exempt` may suppress a
bucket; `guards.paired_counterfactual_exemption` is the exemption this was
built for, and it accepts a bucket only when the bucket is a *structurally
proven* set of counterfactual pairs: even size, partitioning by `pair_id` into
pairs of exactly two rows with present and distinct `pair_target`s, all inside
one split, with a distinct context signature per pair. Repeating an utterance
is legitimate only when the dataset can prove why.

The check has to run at the right moment. `compose` deduplicates on
`text_field` *before* it builds the report, so a report-time check can never
observe a within-split `text` duplicate -- it was already dropped. That is what
`compose(pre_dedup_checks=...)` is for: those checks run on the raw accumulated
rows and **fail fast**, raising `ValueError("pre-dedup check failed: ...")`, so
a post-teacher report rebuild needs no re-injection and `write_dataset` can
never see a build they rejected. Their keys are still merged into
`report["leakage"]` for the record -- and because a rebuilt report cannot
recompute a pre-dedup finding, `build_report(extra_leakage=...)` is how a caller
carries those keys into the report it actually emits, so the manifest records
that the gate ran. `compose` also reports `within_split_duplicates_removed` as a
sub-count of the unchanged `cross_split_duplicates_removed`; neither gates.

### Teacher provenance: outside the hash, stamped, and pinned to a prompt

`immutable_hash` excludes the `provenance` field. It has to: provenance is
written *by* the pipeline the hash protects, so hashing it would move a
record's hash the moment the record was realized and invalidate every request
already in flight. That exclusion is what makes export -> import -> export
idempotent.

`teacher.scrub_fields(record, substitutions, fields=..., editable_fields=...)`
rewrites literal substrings in **non-editable** fields -- the case a teacher
cannot fix, because the text is not the teacher's to touch. It refuses any
field that is also editable, requires `rederive`/`validate` when a scrubbed
field feeds a derived one, and, when the hash moves, appends
`{"before_hash", "after_hash", "excluded_fields"}` to
`provenance["pre_scrub_immutable_hashes"]`. The tag is the **teacher's**
projection (`editable_fields` plus the derived fields they affect), because
that is the exclusion set the importer will recompute.

`import_teacher_responses(..., accept_pre_scrub_hashes=True)` then accepts a
request row pinned to either the live hash or any stamp's `before_hash` whose
tag matches -- which is how a teacher file exported before a scrub stays
usable. The widening is one-directional: the post-edit check still compares
against the live hash, so the record's own semantics still cannot move.
**The trust boundary is real and worth stating:** `provenance` is outside the
hash and `scrub_fields` is public, so a stamp certifies nothing by itself.
Anything that can write provenance can mint a stamp for any hash it likes.
`accept_pre_scrub_hashes=True` does not mean "the library verified a scrub
happened"; it means "I trust every step that could write provenance between
export and import". The default `False` keeps exactly one acceptable hash.

`teacher.compute_teacher_prompt_hash(prompt_path, request_path)` is an explicit
definition of the otherwise-opaque `teacher_prompt_hash`:
`"sha256:"` + sha256 over the canonical JSON of
`{"prompt_sha256": ..., "requests_sha256": ...}`. It moves when the prompt spec
or the exact set of requests moves, and neither file has to be kept to
re-verify the pairing later. The library never recomputes it -- it is
provenance, not a gate.

### The teacher-batch checker

`checks.check_teacher_batch(requests, responses, records, rules)` audits a whole
teacher batch and returns every `Finding` instead of raising on the first, so
one pass tells you everything to send back. `records` is required, so a
response for a record nobody knows about is itself a finding, and the records
no response claimed are available to the rules as `Batch.untouched`. The rule
factories are generic and stdlib-only: `hash_pinned`, `fields_present`,
`no_extra_keys`, `untouched_field`, `min_words`, `max_sentences`,
`banned_pattern`, `preserved_literals`, `unique_normalized` (whose buckets are
*records* carrying the rewritten value, so `paired_counterfactual_exemption`
applies unchanged, and which also catches a rewrite that collides with an
already-released untouched row) and `opening_ngram_cap` (a teacher left alone
converges on one opening; the cap is per family, and counts rewritten rows
only). Domain rules stay in the caller.

### Conversation rows and their hash projection

`rows.make_conversation_row` builds a tool-calling row whose source of truth is
four fields -- `context_messages`, `user_text`, `action_turns`,
`final_response` -- from which `messages`, `text`, `expected_tool_calls` and
`has_context` are derived (`rows.CONVERSATION_DERIVED_FIELDS`). `action_turns`
is the hashed, non-editable list of `{name, arguments, result}` triples, with
`result` an `{"ok": true, "result": ...}` / `{"ok": false, "error": {...}}`
envelope; the transcript renders each as an assistant tool-call message
(`content: None`, deterministic id `call_{record_id}_{n}`) plus its `tool`
result. Context tool calls keep their own `context_{record_id}_{n}` ids and are
never trainable. `validate_conversation_row` first re-renders and compares --
catching a teacher edit whose `rederive` never ran -- and then walks the
transcript state machine (roles, loss labels, id stability per lane, envelope
shape, call/result correlation, and an optional per-tool argument whitelist).

The projection is the point. With `editable_fields=("final_response",)` and
`CONVERSATION_DERIVED_FIELDS`, exactly `{final_response, messages}` sit outside
the hash: change a tool name, an argument, a result envelope, a context turn or
a label and the hash moves. Callers must pass `rederive=rederive_conversation`
and `validate=validate_conversation_row` to both teacher entry points -- the
derived-field map is non-default, so the library cannot infer them.
Note that `text` renders context and user turns only, never tool calls: two
rows sharing a context and a user turn but calling different tools render the
same `text`, which is why the example dedups on `user_text` through
`pre_dedup_checks` rather than leaning on `text`. One more knob a conversation
build must set: `compose`/`build_report` default `secondary_leak_fields` to
`("current_text",)`, which is the classifier row's field, and
`secondary_field_leaks` now raises on a field absent from every row -- so pass
`secondary_leak_fields=("user_text",)` (as the example does) rather than
letting the default fail the build.

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

The second, separate example builds tool-calling **conversation** rows and
wires every mechanism in the section above:

```bash
uv run python -m examples.banking.build_tool_calls dist/banking-tool-calls-example
```

`examples/banking/tool_curricula.py` holds the rows -- single-turn `freeze_card`
and `list_cards` calls, a multi-turn row whose context contains its own
tool-call pair, a governed counterfactual pair sharing one utterance across two
decisions, an error envelope, and a frozen test row that deliberately contains
banned wording. `examples/banking/build_tool_calls.py` is the pipeline:
pre-dedup uniqueness -> export over `final_response` only -> stub teacher ->
`scrub_fields` on a context turn -> `check_teacher_batch` -> import accepting
the pre-scrub hash -> report rebuild -> gated emission.
`examples/banking/voice_spec.md` is the teacher prompt spec that
`compute_teacher_prompt_hash` pins.

## Development

```bash
uv pip install -e ".[dev]"
uv run ruff check .
uv run mypy
uv run pytest -q
```
