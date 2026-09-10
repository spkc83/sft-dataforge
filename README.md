# dataforge

Governed instruction/SFT dataset construction for classifier and tool-calling
fine-tuning in customer-servicing domains: a small, declarative framework for
building label taxonomies, composing curricula into train/validation/test
splits, guarding against PII and cross-split leakage, emitting a locked and
hashed dataset directory, and letting an LLM teacher improve wording without
touching labels.

The library is domain-neutral. It was extracted from the machinery behind a
retail-banking conversation router, but no banking constant is part of it:
intents, tools and curricula for that domain live only under
`examples/banking`, as a worked example you can read end to end.

MIT licensed. Python 3.11+, no runtime dependencies outside the standard
library.

## Quickstart

```bash
uv pip install -e ".[dev]"
uv run python -m examples.banking.build dist/banking-example
```

That builds a classifier dataset from a small synthetic taxonomy
(`view_balance`, `freeze_card`, `policy_faq`, and out-of-scope refusal) over
five tiny curricula, including one multiturn example and one held-out
regression guard. It writes
`dist/banking-example/{train,validation,test}.jsonl`, a `manifest.json`, a
generated `README.md` data card, a `source.lock.json`, and the teacher
request/response jsonl pair. The build is deterministic: run it twice into
different directories and the files are byte-identical.

The second example builds tool-calling **conversation** rows and wires the
guards and checks a governed conversation build needs:

```bash
uv run python -m examples.banking.build_tool_calls dist/banking-tool-calls-example
```

To see the shape of a real integration, read `examples/banking/taxonomy.py`
and `examples/banking/curricula.py` for the declarations, and
`examples/banking/build.py` for the full compose to teacher-realize to emit
pipeline with a stubbed teacher.

## Design principles

- **The deterministic skeleton owns labels.** A `Taxonomy` is a frozen
  dataclass built from a plain dict or JSON file (dimensions, an
  intent-to-hierarchy mapping, and the legal `(action, entity_resolution)`
  pairs per value of the gate dimension) rather than a python module full of
  domain constants. Curricula compute *which* label a given example
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
  `default_gates` aborts on any nonzero PII match count or nonempty leakage
  finding; a new check in `dataforge.guards` gates automatically because that
  gate walks `report["leakage"]` by key suffix (`_leak_count`, `_leaks`)
  rather than an enumerated list. It walks that sub-report and nothing else,
  which is why a finding computed outside `compose` has to be carried in
  through `build_report(extra_leakage=...)` to gate at all.
- **Locks and manifests for provenance.** Every emitted split gets a
  canonical (sorted-key) jsonl encoding and a sha256 in `manifest.json`.
  `dataforge.emit.write_source_lock` / `verify_release_split_digests` let a
  pipeline pin a release's split digests and detect drift on the next
  build.

## Package map

| Module | Responsibility |
| --- | --- |
| `dataforge/taxonomy.py` | `Taxonomy`, a frozen dataclass with `from_dict`/`from_json`: intent hierarchy, legal action/entity-resolution pairs per value of the configurable `gate_dimension`, tool compatibility. `labels_for_example` / `validate_hierarchical_labels`. |
| `dataforge/rows.py` | The row contract: `render_context` (the `[PRIOR_STATE]` / `[CURRENT_USER]` / `[PREVIOUS_ASSISTANT]` / `[PREVIOUS_USER]` flattening), `make_row` (labels, multi-hot relations, and the governance identifiers), `make_conversation_row`, `normalize_text`. |
| `dataforge/curricula.py` | `@curriculum` registration on a `Registry`; `compose` runs every curriculum per split, enforces group/trajectory/pair non-leakage, deduplicates with eval-wins ordering, and produces the governance report. Also `BehaviourSeed`/`behaviour_rows` and the `uses` tag with `foreign_use_rows`. |
| `dataforge/guards.py` | PII regexes, held-out exact + n-gram leak detection, `leakage_report`, `secondary_field_leaks`, `banned_wording_leaks`; the duplicate and near-duplicate guards, and the build-time invariant guards (`field_invariant_leaks` and its primitives, `probe_exclusion_leaks`, `unsupported_claim_leaks`). |
| `dataforge/emit.py` | Canonical jsonl encoding, `write_dataset` (gates, staleness check, split files, manifest, data card), `default_gates`, `write_source_lock`, `verify_release_split_digests`. |
| `dataforge/teacher.py` | `immutable_hash`, `export_teacher_requests`, `import_teacher_responses`, `scrub_fields`, `compute_teacher_prompt_hash` -- the wording-only teacher realization harness. |
| `dataforge/checks.py` | The teacher-batch checker: `check_teacher_batch(requests, responses, records, rules)`, where the first two are jsonl paths, and the generic rule factories (`hash_pinned`, `min_words`, `banned_pattern`, `unique_normalized`, `opening_ngram_cap`, ...). Rules see a `Batch` of matched `Pair` triples. |

## Row shapes

`rows.make_row` builds the classifier row: hierarchical labels from the
taxonomy, multi-hot relations, and a `text` derived by `render_context` from
the prior dialogue state and the surrounding turns.

Every row also carries the identifiers the governance checks run on, and a
curriculum has to supply them:

| field | what it identifies |
| --- | --- |
| `group_id` | the family of rows generated together; a group may not straddle splits |
| `trajectory_id` | the conversation a multi-turn row belongs to; defaults to `group_id` |
| `pair_id` / `pair_target` / `pair_family` | the two halves of a counterfactual pair and what distinguishes them |
| `source` / `source_split` | where the row came from and which split its source assigned it |
| `example_kind` | the curriculum's own label for the shape of the example |

`compose` raises when one of these straddles a split, so an absent or reused
identifier is a build failure rather than a silent leak. None of them is the
`provenance` field discussed under the teacher boundary: that key does not
exist until `import_teacher_responses` or `scrub_fields` writes it.

`rows.make_conversation_row` builds a tool-calling row whose source of truth
is four fields -- `context_messages`, `user_text`, `action_turns`,
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

The hash projection is the point. `messages` is the only derived field that
lists `final_response` among its sources, so with
`editable_fields=("final_response",)` and `CONVERSATION_DERIVED_FIELDS` the
projection excludes exactly `{final_response, messages}` (`provenance` is
excluded unconditionally, for the reason given under the teacher boundary).
Change a tool name, an argument, a result envelope, a context turn or a label
and the hash moves.

Getting that projection requires passing
`derived_fields=CONVERSATION_DERIVED_FIELDS` to both teacher entry points,
together with `rederive=rederive_conversation` and
`validate=validate_conversation_row`. `derived_fields` defaults to the
classifier map, under which no derived field depends on `final_response`, so
omitting it silently narrows the exclusion set to `{final_response}` alone,
leaves `messages` inside the hash, and makes `rederive` move it. The import
then fails with "changed immutable semantics", which reads like a misbehaving
teacher rather than a wiring mistake.

## Curricula

A curriculum is a function registered on a `Registry` that returns rows for a
given split. `compose(seed_splits, registry, ...)` drives them. `seed_splits`
is the starting map of split name to rows; a build with no pre-existing rows
passes an empty list per split, which is what both examples do. Every key in
it, and every split any curriculum targets, must appear in `split_order` or
`compose` raises rather than dropping rows silently.
Rows accumulate per split, seed rows first and then curriculum rows, and seed
rows are never `use`-filtered because they did not come from a curriculum that
could declare a `uses` policy.

### Repeating one behaviour across frames, with subjects held back

Some model behaviours have to be taught at the weight level, and a single
well-written example does not reach the weights. `curricula.BehaviourSeed`
states such a behaviour once: `frames` are the ways a customer might raise it,
`finals` the matching responses, and `subjects` the things it is raised about,
keyed by split. `curricula.behaviour_rows(seeds, split, row_fn=...)` expands it
-- every frame of every train subject, and the first
`frames_per_validation_subject` frames (default 2) for any other split.

The subjects a non-train split uses must be disjoint from train's;
`behaviour_rows` raises rather than trusting it, because a behaviour scored on
the subjects it was trained on measures recall, not generalization. It also
fails fast on frames and finals of different lengths, a frame with no `"{s}"`
placeholder, two seeds sharing `(family, key)`, and a seed with no subjects for
the split being built. `row_fn` receives
`seed`/`split`/`subject`/`frame_index`/`variant`/`text`/`final` and returns the
row, so every schema decision stays in the caller and one seed can serve a
classifier build and a conversation build unchanged.

### `uses`: which consumers a curriculum's rows may reach

`Registry.register(name, splits, uses=("*",))` declares which consumers a
curriculum's rows are allowed to reach, and `Registry.build(split, use=...)` /
`compose(use=...)` honour it; `use=None`, the default, applies no filter. Some
families must reach SFT train/validation and never a secondary consumer (a
router's training export, a public sample), and the convention "everyone
remembers to filter that family out" rots the moment someone adds a family
without knowing. Declaring it at registration puts the policy next to the rows.
`curricula.foreign_use_rows(registry, rows, use=..., name_field=...)` is the
audit half: given rows that name their curriculum, it returns the ones a
consumer should never have received, so an export built elsewhere can still be
checked.

## Guards

Guards report findings under keys ending in `_leaks` / `_leak_count`, which is
what lets the default emission gate pick them up without being told about each
one. Where a guard has to run before deduplication can hide a violation, wire
it through `compose(pre_dedup_checks=...)` instead of `extra_leak_checks`.

### Banned wording, with the frozen splits exempt

`guards.banned_wording_leaks(splits, pattern, ...)` flags banned phrasing in
trainable text. Two details matter. It scans only `trainable_splits`
(`("train", "validation")` by default), so a frozen evaluation row is exempt
**by construction** rather than by a special case at gate time -- a regression
row often has to contain exactly the phrasing the model must never learn to
produce. And it scans plain fields (`text_fields`, `("text",)` by default) and
rendered transcripts separately: pass `message_fields=("messages",)` for
conversation rows and it reads the `content` of every message whose role is
user or assistant, so a tool-call assistant turn (`content: None`) and the
`system`/`tool` turns are skipped while context turns, the current user turn
and the final are all covered.

### Global normalized uniqueness, and the pre-dedup hook

`guards.duplicate_text_leaks(splits, field, ...)` buckets rows **globally** on
`rows.normalize_text_ascii` (lowercase, non-alphanumerics collapsed to spaces --
deliberately more aggressive than `normalize_text`), so a within-split
duplicate is flagged just as a cross-split one is. `exempt` may suppress a
bucket; `guards.paired_counterfactual_exemption` accepts a bucket only when the
bucket is a *structurally proven* set of counterfactual pairs: even size,
partitioning by `pair_id` into pairs of exactly two rows with present and
distinct `pair_target`s, all inside one split, with a distinct context
signature per pair. Repeating an utterance is legitimate only when the dataset
can prove why.

This check has to run at the right moment. `compose` deduplicates on
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
sub-count of `cross_split_duplicates_removed`; neither gates. The outer counter
covers removals of both kinds despite its name, which is held stable so that a
caller reading it does not see the number change meaning underneath them.

### Field invariants

`guards.field_invariant_leaks(splits, field=..., invariants=[...])` runs
per-row invariants and reports every violation. The invariant primitives are
`no_digits`, `no_questions`, `banned_patterns(patterns, label=...)`,
`forbidden_terms(terms, label=...)`, `required_markers(markers_by_tag,
tag_fn=...)` and `min_word_count(n)`; each returns a `(value, row) -> str |
None` callable, and any callable with a `__name__` works. `row_predicate` scopes
the check to the rows an invariant is about -- a hand-authored family normally
holds to rules the rest of the corpus does not.

Wire these through `pre_dedup_checks`: a row that breaks an invariant and
happens to duplicate a clean sibling would otherwise be dropped by dedup, and
the build would pass on the strength of the row that survived. An absent
`field` and an empty `invariants` both raise, for the same reason
`secondary_field_leaks` raises on a missing field -- a check that cannot fire
still reports a reassuring zero.

### The probe-exclusion gate

`guards.probe_exclusion_leaks(splits, probes=..., fragments=..., fields=...)`
flags trainable text that reproduces an evaluation probe: `probes` match a
normalized field value exactly, `fragments` match as substrings (a paraphrased
probe keeps its distinctive phrase). Probes held out of training are only held
out if a gate proves it; this is the check that keeps "the model passed the
probe" a claim about generalization rather than memorization. `splits_checked`
defaults to the trainable splits, so the probes' own frozen split is exempt by
construction, and passing neither probes nor fragments raises.

### Claims without evidence

`guards.unsupported_claim_leaks(splits, field=..., claim_patterns=...,
evidence_fn=...)` flags a row whose text asserts a completed account action
while `evidence_fn(row)` is false -- normally "this row carries no tool-call
turns". A final that says "I have frozen the card" on a row that froze nothing
trains the model to produce the sentence in place of the action, which is the
most expensive failure a servicing model has. What counts as evidence is the
caller's, because it is the row schema's business.

### Near-duplicates

`guards.fuzzy_duplicate_leaks(splits, field=..., threshold=0.995,
group_fn=...)` flags pairs of rows whose normalized values are nearly identical
but not equal -- pairs that normalize to the same value stay
`duplicate_text_leaks`' job. That division of labour is close to a partition but
not exactly one: this guard normalizes with `normalize_text` and that one
defaults to `normalize_text_ascii`, so a pair equal only under the ASCII
normalizer (`"café"`/`"cafe"`) is reported by both.

Comparison is within one split and, with `group_fn`, within one group. Both are
cost bounds on an O(n^2) pass rather than claims about where duplicates live:
the quadratic term ends up per family instead of per corpus, and the cheap
`real_quick_ratio`/`quick_ratio` bounds short-circuit before `ratio()`. A
cross-split near-duplicate is consequently covered by nothing here --
`secondary_field_leaks` buckets on exact normalized equality and will not see it
either. `splits_checked` also defaults to the trainable splits, so a frozen
evaluation split is not scanned for near-duplicates at all unless you name it.

### Wiring a conversation build

Four guard defaults are written for the classifier row and have to be
overridden when the rows are conversations. Each of them fails quietly rather
than loudly, so they are worth setting before the first build.

| setting | default | pass instead |
| --- | --- | --- |
| `secondary_leak_fields` on `compose`/`build_report` | `("current_text",)` | `("user_text",)`, or `secondary_field_leaks` raises on a field no row has |
| `context_fields` on `paired_counterfactual_exemption` | `("history",)` | `("context_messages",)`, or the structural proof compares two empty projections and the exemption cannot be earned |
| `text_fields` on `banned_wording_leaks` | `("text",)` | `()` alongside `message_fields=("messages",)`, since `text` is the flattened context and user turn and every hit there would otherwise be reported twice |
| the field you deduplicate on | `text` | `user_text` through `pre_dedup_checks`, because `text` omits tool calls and two rows calling different tools render identically |

## Emission

`emit.write_dataset` writes `{split}.jsonl` in a canonical sorted-key encoding,
a `manifest.json` carrying each split's sha256, and a `README.md` data card
from lines the caller supplies. Gates run first, against the report, and a
raising gate leaves no file behind.

Before the gates it checks that the report is not stale. `build_report` stamps
a `splits_fingerprint` over the rows it saw, `write_dataset` recomputes it over
the rows it is about to write, and a mismatch raises. A report that matches the
expected contract but carries no fingerprint at all raises as well, because
`build_report` always sets one and its absence means the report came from
somewhere that cannot vouch for freshness. This is what forces a rebuild after
a teacher pass: the realized rows are not the rows the first report described.

`write_source_lock` pins a release's split digests to a file, and
`verify_release_split_digests` compares a later build against it, which is how
a pipeline detects that a corpus moved without anyone saying so.

## The teacher boundary

### Provenance sits outside the hash

`immutable_hash` excludes the `provenance` field. It has to: provenance is
written *by* the pipeline the hash protects, so hashing it would move a
record's hash the moment the record was realized and invalidate every request
already in flight. That exclusion is what makes export, import and re-export
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
tag matches, which is how a teacher file exported before a scrub stays usable.
The widening is one-directional: the post-edit check still compares against the
live hash, so the record's own semantics still cannot move.

**The trust boundary is worth stating plainly.** `provenance` is outside the
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

## Worked examples

`examples/banking/curricula.py` and `examples/banking/build.py` cover the
classifier path. The tool-calling path is separate.

The examples are a working integration, not an exhaustive demonstration. The
near-duplicate guard, the `forbidden_terms`, `required_markers` and
`min_word_count` invariant primitives, and the `untouched_field`,
`max_sentences` and `preserved_literals` rules are exercised in `tests/`
rather than in a build. Read the tests for those.

`examples/banking/tool_curricula.py` holds the rows: single-turn `freeze_card`
and `list_cards` calls, a multi-turn row whose context contains its own
tool-call pair, a governed counterfactual pair sharing one utterance across two
decisions, an error envelope, a frozen test row that deliberately contains
banned wording, and a `refusal_honesty` behaviour curriculum built from one
seed with two train subjects and one held-back validation subject, tagged
`uses=("sft",)`, whose finals hold to invariants the tool-calling rows do not.

`examples/banking/build_tool_calls.py` is the pipeline: pre-dedup uniqueness,
field invariants, probe exclusion and unsupported-claim checks, then a `use`-
filtered router export audited with `foreign_use_rows`, then an export over
`final_response` only, a stub teacher, `scrub_fields` on a context turn,
`check_teacher_batch`, an import accepting the pre-scrub hash, a report
rebuild, and gated emission. `examples/banking/voice_spec.md` is the teacher
prompt spec that `compute_teacher_prompt_hash` pins.

## Development

```bash
uv pip install -e ".[dev]"
uv run ruff check .
uv run mypy
uv run pytest -q
```

## License

MIT. See [LICENSE](LICENSE).
