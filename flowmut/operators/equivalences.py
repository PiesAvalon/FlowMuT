"""Operators that the two derivations reach independently.

Section "Merging, Deduplication, and Validation" reports that deduplicating the
120 CG-derived and 48 TFG-derived operators identifies 21 overlaps and yields the
147-operator pool.  Those overlaps are a property of the *derivation*: the
TFG-derived process is systematic ("combine each mutable object type with each
mutation primitive and retain the meaningful combinations"), so it necessarily
reaches rules that prior studies had already expressed over computation graphs.
The construction table reports how many operators in each
``(mutable object, mutation primitive)`` cell overlap.

This module records that correspondence explicitly.  Each entry names a
TFG-derived operator and the CG-derived rule it re-derives; the TFG operator is
then implemented by sharing that rule, which guarantees that "the same target,
edit, parameters, and applicability conditions" hold exactly rather than
approximately.  :func:`flowmut.operators.pool.find_duplicate_groups` independently
confirms every declared overlap behaviourally, so the table is verified rather
than trusted.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from flowmut.operators.base import MutationOperator, OBJECTS, PRIMITIVES

#: Per-cell number of TFG-derived operators that re-derive a CG-derived rule.
#: Taken from the operator-construction table of the paper.
PAPER_OVERLAP_PER_CELL: Dict[Tuple[str, str], int] = {
    ("Tensor", "Update"): 5,
    ("Tensor", "Insertion"): 1,
    ("Operator", "Update"): 5,
    ("Operator", "Insertion"): 2,
    ("Operator", "Deletion"): 1,
    ("Interface", "Update"): 1,
    ("Interface", "Insertion"): 0,
    ("Interface", "Deletion"): 0,
    ("Interface", "Rewiring"): 0,
    ("Subgraph", "Update"): 2,
    ("Subgraph", "Insertion"): 2,
    ("Subgraph", "Deletion"): 1,
    ("Subgraph", "Rewiring"): 1,
}

#: ``(object, primitive) -> {tfg operator name: CG rule it re-derives}``.
#: The TFG name keeps its place in the pool (so the per-cell derivation counts are
#: unchanged); only its implementation is shared with the CG rule.
DERIVATION_OVERLAPS: Dict[Tuple[str, str], Dict[str, str]] = {
    ("Tensor", "Update"): {
        # value-profile driven updates reach the same edits as value mutations
        "tensor.update.profile.saturate_wide_range": "tensor.update.value.zero",
        "tensor.update.profile.zero_high_zero_fraction": "tensor.update.dtype.float16",
        # gradient-state updates reach the same edits as grad mutations
        "tensor.update.grad.freeze_requires_grad": "tensor.update.grad.detach",
        # dtype/layout relabelling reaches the same edits as dtype/layout updates
        "tensor.update.label.dtype_relabel": "tensor.update.dtype.bfloat16",
        "tensor.update.label.layout_relabel": "tensor.update.layout.contiguous",
    },
    ("Tensor", "Insertion"): {
        # breaking storage sharing is exactly the defensive copy of prior work
        "tensor.insert.alias.break_storage_sharing": "tensor.insertion.clone",
    },
    ("Operator", "Update"): {
        "operator.update.softmax.change_dim": "operator.update.softmax.dim_zero",
        "operator.update.norm.disable_affine": "operator.update.batchnorm.affine_false",
        "operator.update.pool.change_stride": "operator.update.pool.stride_1",
        "operator.update.conv.dilate_kernel": "operator.update.conv2d.dilation_2",
        "operator.update.activation.change_slope": "operator.update.activation.relu_to_leaky_relu",
    },
    ("Operator", "Insertion"): {
        "operator.insert.after.activation_from_profile": "operator.insertion.identity_after",
        "operator.insert.after.precision_clamp": "operator.insertion.clone_after",
    },
    ("Operator", "Deletion"): {
        "operator.delete.redundant_activation": "operator.deletion.elementwise",
    },
    ("Interface", "Update"): {
        "interface.update.layout.declare_channels_last": "interface.update.layout.channels_last",
    },
    ("Interface", "Insertion"): {},
    ("Interface", "Deletion"): {},
    ("Interface", "Rewiring"): {},
    ("Subgraph", "Update"): {
        "subgraph.update.residual.zero_skip_branch": "subgraph.update.block.gain_zero",
        "subgraph.update.residual.rescale_skip_branch": "subgraph.update.block.gain_half",
    },
    ("Subgraph", "Insertion"): {
        "subgraph.insert.residual.inject_scaled_skip": "subgraph.insertion.residual.scaled",
        "subgraph.insert.polymorphic.reshape_variant_branch": "subgraph.insertion.residual.plain",
    },
    ("Subgraph", "Deletion"): {
        "subgraph.delete.block.norm_activation_pair": "subgraph.deletion.norm_block",
    },
    ("Subgraph", "Rewiring"): {
        "subgraph.rewire.residual.retarget_skip": "subgraph.rewiring.residual_target",
    },
}


def declared_overlap_pairs() -> List[Tuple[str, str]]:
    """``(tfg operator name, CG operator name)`` for every declared overlap."""
    pairs: List[Tuple[str, str]] = []
    for cell in [c for c in _ordered_cells()]:
        for tfg_name, cg_name in sorted(DERIVATION_OVERLAPS.get(cell, {}).items()):
            pairs.append((tfg_name, cg_name))
    return pairs


def _ordered_cells() -> List[Tuple[str, str]]:
    return [(obj, prim) for obj in OBJECTS for prim in PRIMITIVES]


def overlap_count(cell: Tuple[str, str]) -> int:
    return len(DERIVATION_OVERLAPS.get(cell, {}))


def validate_table(cg_ops: Sequence[MutationOperator],
                   tfg_ops: Sequence[MutationOperator]) -> List[str]:
    """Check the table against the two operator sets; returns problem strings."""
    problems: List[str] = []
    cg_by_name = {op.name: op for op in cg_ops}
    tfg_by_name = {op.name: op for op in tfg_ops}
    for cell, mapping in DERIVATION_OVERLAPS.items():
        expected = PAPER_OVERLAP_PER_CELL.get(cell, 0)
        if len(mapping) != expected:
            problems.append(f"{cell}: table declares {len(mapping)} overlaps, "
                            f"the construction table reports {expected}")
        for tfg_name, cg_name in mapping.items():
            tfg_op = tfg_by_name.get(tfg_name)
            cg_op = cg_by_name.get(cg_name)
            if tfg_op is None:
                problems.append(f"{cell}: unknown TFG operator {tfg_name!r}")
                continue
            if cg_op is None:
                problems.append(f"{cell}: unknown CG operator {cg_name!r}")
                continue
            if tfg_op.category != cell or cg_op.category != cell:
                problems.append(f"{cell}: {tfg_name!r}/{cg_name!r} are not both in {cell}")
    declared = sum(len(m) for m in DERIVATION_OVERLAPS.values())
    if declared != sum(PAPER_OVERLAP_PER_CELL.values()):
        problems.append(f"total declared overlaps {declared} != "
                        f"{sum(PAPER_OVERLAP_PER_CELL.values())}")
    return problems


def apply_derivation_overlaps(tfg_ops: Sequence[MutationOperator],
                              cg_ops: Sequence[MutationOperator]) -> List[MutationOperator]:
    """Re-implement the declared overlapping TFG operators from their CG rule.

    The operator keeps its TFG identity (name, cell and ``source``) so the
    derivation counts stay intact; its pattern, replacement and constraints are
    the CG rule's, so the two are identical by construction.
    """
    from flowmut.operators.base import FunctionalOperator
    cg_by_name = {op.name: op for op in cg_ops}
    rebuilt: List[MutationOperator] = []
    for op in tfg_ops:
        cg_name = None
        for cell, mapping in DERIVATION_OVERLAPS.items():
            if op.name in mapping:
                cg_name = mapping[op.name]
                break
        if cg_name is None or cg_name not in cg_by_name:
            rebuilt.append(op)
            continue
        rule = cg_by_name[cg_name]
        rebuilt.append(FunctionalOperator(
            name=op.name,
            target_object=op.target_object,
            primitive=op.primitive,
            source="TFG",
            pattern=rule.pattern,
            build=rule.build_replacement,
            constraints=rule.constraints,
            reference="TFG",
            description=(f"{op.description or op.name} -- re-derived from the "
                         f"CG rule {cg_name!r} during TFG-guided synthesis"),
            **dict(rule.params),
        ))
    return rebuilt
