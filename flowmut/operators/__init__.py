"""The FlowMuT mutation operator pool."""

from flowmut.operators.base import (  # noqa: F401
    OBJECTS,
    PRIMITIVES,
    Candidate,
    ConstraintSpec,
    FunctionalOperator,
    MutationOperator,
    NodeTemplate,
    OperatorPool,
    Pattern,
    Replacement,
    ReplacementError,
    Site,
    apply_replacement,
    register_site_predicate,
)
from flowmut.operators.constraints import (  # noqa: F401
    CONSTRAINT_KINDS,
    CONSTRAINT_VARIANTS,
    ConstraintChecker,
    ConstraintReport,
    filter_candidates,
    legality_rate,
)
from flowmut.operators.equivalences import (  # noqa: F401
    DERIVATION_OVERLAPS,
    PAPER_OVERLAP_PER_CELL,
    declared_overlap_pairs,
    validate_table,
)
from flowmut.operators.probes import probe_tfgs  # noqa: F401
from flowmut.operators.pool import (  # noqa: F401
    DedupReport,
    build_pool,
    deduplicate_pool,
    find_duplicate_groups,
    cg_operators,
    generate_candidates,
    pool_composition,
    pool_summary,
    tfg_operators,
    validate_pool,
)

__all__ = [
    "MutationOperator", "FunctionalOperator", "Pattern", "Site", "Candidate",
    "NodeTemplate", "Replacement", "ConstraintSpec", "OperatorPool",
    "apply_replacement", "ReplacementError", "register_site_predicate",
    "OBJECTS", "PRIMITIVES",
    "ConstraintChecker", "ConstraintReport", "CONSTRAINT_KINDS",
    "CONSTRAINT_VARIANTS", "filter_candidates", "legality_rate",
    "build_pool", "cg_operators", "tfg_operators", "generate_candidates",
    "validate_pool", "pool_composition", "pool_summary",
    "DedupReport", "deduplicate_pool", "find_duplicate_groups",
    "probe_tfgs", "DERIVATION_OVERLAPS", "PAPER_OVERLAP_PER_CELL",
    "declared_overlap_pairs", "validate_table",
]
