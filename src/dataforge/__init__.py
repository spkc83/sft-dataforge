"""dataforge: governed SFT/instruction dataset construction framework.

See the package README for the design principles. The public surface is
small and re-exported here for convenience:

* :mod:`dataforge.taxonomy` -- declarative label hierarchy + validation.
* :mod:`dataforge.rows` -- row contract, context rendering, provenance.
* :mod:`dataforge.curricula` -- curriculum registration and composition.
* :mod:`dataforge.guards` -- PII and cross-split leakage guards.
* :mod:`dataforge.emit` -- dataset directory emission and locking.
* :mod:`dataforge.teacher` -- wording-only LLM teacher realization.
* :mod:`dataforge.checks` -- teacher-batch checker (accumulating findings).
"""

from dataforge.checks import (
    Batch,
    CheckerInputError,
    Finding,
    Pair,
    Rule,
    banned_pattern,
    build_batch,
    check_teacher_batch,
    fields_present,
    format_findings,
    hash_pinned,
    max_sentences,
    min_words,
    no_extra_keys,
    opening_ngram_cap,
    preserved_literals,
    row_rule,
    summarize,
    unique_normalized,
    untouched_field,
)
from dataforge.curricula import (
    DEFAULT_DEDUP_PRIORITY,
    DEFAULT_SPLIT_ORDER,
    REPORT_CONTRACT,
    Curriculum,
    Registry,
    build_report,
    compose,
    curriculum,
    default_registry,
    splits_fingerprint,
)
from dataforge.emit import (
    default_gates,
    verify_release_split_digests,
    write_dataset,
    write_source_lock,
)
from dataforge.guards import count_pii_matches, heldout_leaks, leakage_report
from dataforge.rows import (
    CONVERSATION_DERIVED_FIELDS,
    DERIVED_FIELDS,
    make_conversation_row,
    make_row,
    normalize_text,
    normalize_text_ascii,
    rederive_conversation,
    rederive_text,
    render_context,
    render_conversation,
    render_conversation_text,
    validate_conversation_messages,
    validate_conversation_row,
    validate_row_consistency,
)
from dataforge.taxonomy import IntentSpec, Taxonomy, TaxonomyError
from dataforge.teacher import (
    TeacherRealizationError,
    compute_teacher_prompt_hash,
    export_teacher_requests,
    immutable_hash,
    import_teacher_responses,
    scrub_fields,
)

__all__ = [
    "Batch",
    "CONVERSATION_DERIVED_FIELDS",
    "CheckerInputError",
    "Curriculum",
    "DEFAULT_DEDUP_PRIORITY",
    "DEFAULT_SPLIT_ORDER",
    "DERIVED_FIELDS",
    "Finding",
    "IntentSpec",
    "Pair",
    "REPORT_CONTRACT",
    "Registry",
    "Rule",
    "Taxonomy",
    "TaxonomyError",
    "TeacherRealizationError",
    "banned_pattern",
    "build_batch",
    "build_report",
    "check_teacher_batch",
    "compose",
    "compute_teacher_prompt_hash",
    "count_pii_matches",
    "curriculum",
    "default_gates",
    "default_registry",
    "export_teacher_requests",
    "fields_present",
    "format_findings",
    "hash_pinned",
    "heldout_leaks",
    "immutable_hash",
    "import_teacher_responses",
    "leakage_report",
    "make_conversation_row",
    "make_row",
    "max_sentences",
    "min_words",
    "no_extra_keys",
    "normalize_text",
    "normalize_text_ascii",
    "opening_ngram_cap",
    "preserved_literals",
    "rederive_conversation",
    "rederive_text",
    "render_context",
    "render_conversation",
    "render_conversation_text",
    "row_rule",
    "scrub_fields",
    "splits_fingerprint",
    "summarize",
    "unique_normalized",
    "untouched_field",
    "validate_conversation_messages",
    "validate_conversation_row",
    "validate_row_consistency",
    "verify_release_split_digests",
    "write_dataset",
    "write_source_lock",
]
