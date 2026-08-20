"""dataforge: governed SFT/instruction dataset construction framework.

See the package README for the design principles. The public surface is
small and re-exported here for convenience:

* :mod:`dataforge.taxonomy` -- declarative label hierarchy + validation.
* :mod:`dataforge.rows` -- row contract, context rendering, provenance.
* :mod:`dataforge.curricula` -- curriculum registration and composition.
* :mod:`dataforge.guards` -- PII and cross-split leakage guards.
* :mod:`dataforge.emit` -- dataset directory emission and locking.
* :mod:`dataforge.teacher` -- wording-only LLM teacher realization.
"""

from dataforge.curricula import Curriculum, Registry, compose, curriculum, default_registry
from dataforge.emit import (
    default_gates,
    verify_release_split_digests,
    write_dataset,
    write_source_lock,
)
from dataforge.guards import count_pii_matches, heldout_leaks, leakage_report
from dataforge.rows import make_row, normalize_text, render_context
from dataforge.taxonomy import IntentSpec, Taxonomy, TaxonomyError
from dataforge.teacher import (
    TeacherRealizationError,
    export_teacher_requests,
    immutable_hash,
    import_teacher_responses,
)

__all__ = [
    "Curriculum",
    "IntentSpec",
    "Registry",
    "Taxonomy",
    "TaxonomyError",
    "TeacherRealizationError",
    "compose",
    "count_pii_matches",
    "curriculum",
    "default_gates",
    "default_registry",
    "export_teacher_requests",
    "heldout_leaks",
    "immutable_hash",
    "import_teacher_responses",
    "leakage_report",
    "make_row",
    "normalize_text",
    "render_context",
    "verify_release_split_digests",
    "write_dataset",
    "write_source_lock",
]
