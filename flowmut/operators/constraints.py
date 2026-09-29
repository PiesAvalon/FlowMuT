"""The FlowMuT constraint system.

A mutation operator has *input*, *internal* and *output* constraints
(Section "Mutation as Constrained Subgraph Replacement"):

``input``    checks edges produced **outside** the matched subgraph and consumed
             **inside** it,
``internal`` checks the topology and data flow **within** the matched and
             replacement subgraphs,
``output``   checks edges produced **inside** the replacement and consumed by
             nodes **outside** it.

FlowMuT implements the three categories by materialising the replacement on a
copy of the graph, inferring tensor states for the new edges from ``F`` and the
static shape rules, and then re-evaluating ``R(v)`` on exactly those nodes whose
operand states changed.  Each violation is tagged with the category implied by
where its offending edge was produced, which is what RQ3 measures.

Disabling a category (the RQ3 ablation) is a first-class option so that the same
code path produces both the complete constraint system and its variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from flowmut.ir.graph import Graph
from flowmut.ir.ops import canonical_op, infer_shape
from flowmut.ir.specs import TensorSpec
from flowmut.operators.base import (
    BOOKKEEPING_ATTRS,
    ReplacementError,
    Candidate,
    Replacement,
    Site,
    apply_replacement,
)
from flowmut.tfg.predicates import ReqContext, evaluate as evaluate_predicate
from flowmut.tfg.tfg import TFG, Violation

#: All constraint categories, in the order the paper introduces them.
CONSTRAINT_KINDS: Tuple[str, ...] = ("input", "internal", "output")

#: Ablation configurations used by RQ3.
CONSTRAINT_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "full": ("input", "internal", "output"),
    "without_input": ("internal", "output"),
    "without_internal": ("input", "output"),
    "without_output": ("input", "internal"),
    "without_constraints": (),
}


@dataclass
class ConstraintReport:
    """The outcome of filtering one candidate."""

    accepted: bool
    violations: List[Violation] = field(default_factory=list)
    mutated_graph: Optional[Graph] = None
    #: First violated category, used for the RQ3 categorisation.
    first_violation_kind: str = ""
    error: str = ""

    def violations_of(self, kind: str) -> List[Violation]:
        return [v for v in self.violations if v.kind == kind]


# ---------------------------------------------------------------------------
# Spec inference for the mutated graph
# ---------------------------------------------------------------------------

def _dtype_for(op: str, attrs: Dict[str, Any], inputs: Sequence[TensorSpec]) -> str:
    op = canonical_op(op)
    if op == "cast":
        return str(attrs.get("dtype", inputs[0].dtype if inputs else "float32"))
    if op in ("argmax", "argmin"):
        return "int64"
    if op in ("embedding",) and len(inputs) >= 2:
        return inputs[1].dtype
    if not inputs:
        return str(attrs.get("dtype", "float32"))
    # Promotion when a constant operand was introduced by the replacement.
    dtypes = {s.dtype for s in inputs}
    if len(dtypes) > 1:
        order = ("float64", "float32", "float16", "bfloat16", "int64", "int32",
                 "int16", "int8", "uint8", "bool")
        for d in order:
            if d in dtypes:
                if d in ("int16", "int8", "uint8") and "float32" in dtypes:
                    return "float32"
                return d
    return inputs[0].dtype


def infer_specs_for_graph(mutated: Graph, known: Dict[str, Optional[TensorSpec]]
                          ) -> Dict[str, Optional[TensorSpec]]:
    """Infer ``F``-style states for edges created by a replacement.

    Edges that already exist keep their profiled annotation; new edges get a
    statically inferred shape and a propagated dtype/device/layout.  When the
    shape cannot be inferred, the annotation falls back to the first data input
    of the producing node, mirroring how an unknown-but-compatible tensor would
    be annotated.
    """
    specs: Dict[str, Optional[TensorSpec]] = dict(known)
    order = mutated.topological_order()
    for nid in order:
        node = mutated.nodes[nid]
        out_edges = mutated.out_edges(nid)
        if all(e in specs for e in out_edges):
            continue
        in_specs = [specs.get(e) for e in node.inputs]
        present = [s for s in in_specs if s is not None]
        shape = infer_shape(node.op, present, node.attrs)
        if shape is None and present:
            shape = tuple(present[0].shape)
        dtype = _dtype_for(node.op, node.attrs, present)
        base = present[0] if present else None
        spec = TensorSpec(
            shape=tuple(shape or ()),
            dtype=dtype,
            layout=base.layout if base else "contiguous",
            device=base.device if base else "cpu",
            requires_grad=any(s.requires_grad for s in present),
            alias_of=None,
            is_view=node.op in ("view", "permute", "transpose", "squeeze",
                                "unsqueeze", "expand", "broadcast_to", "getitem"),
            is_leaf=False,
            strides=None,
            value=base.value if base is not None else None,
            numel=int(_numel(shape)),
        )
        for e in out_edges:
            specs[e] = spec
    return specs


def _numel(shape) -> int:
    n = 1
    for d in shape or ():
        n *= int(d)
    return n


# ---------------------------------------------------------------------------
# The checker
# ---------------------------------------------------------------------------

@dataclass
class ConstraintChecker:
    """Filters candidates with the (possibly ablated) constraint system."""

    active: Tuple[str, ...] = CONSTRAINT_KINDS
    check_structure: bool = True
    check_operator_local: bool = True
    max_violations: int = 8

    @classmethod
    def for_variant(cls, variant: str = "full", **kwargs: Any) -> "ConstraintChecker":
        if variant not in CONSTRAINT_VARIANTS:
            raise KeyError(f"unknown constraint variant {variant!r}; "
                           f"expected one of {sorted(CONSTRAINT_VARIANTS)}")
        return cls(active=CONSTRAINT_VARIANTS[variant], **kwargs)

    # -- public API --------------------------------------------------------
    def filter(self, tfg: TFG, candidate: Candidate) -> ConstraintReport:
        op = candidate.operator
        site = candidate.site
        try:
            replacement = op.build_replacement(tfg, site)
        except ReplacementError as exc:
            return ConstraintReport(accepted=False, error=str(exc),
                                    first_violation_kind="structural")
        except Exception as exc:  # an operator bug must not abort the campaign
            return ConstraintReport(accepted=False, error=f"{type(exc).__name__}: {exc}",
                                    first_violation_kind="structural")
        if replacement is None:
            return ConstraintReport(accepted=False, error="pattern not applicable",
                                    first_violation_kind="pattern")
        return self.filter_replacement(tfg, candidate, replacement)

    def filter_replacement(self, tfg: TFG, candidate: Candidate,
                           replacement: Replacement) -> ConstraintReport:
        op = candidate.operator
        site = candidate.site
        base = tfg.graph

        try:
            mutated = apply_replacement(base, site, replacement, op.name)
        except ReplacementError as exc:
            return ConstraintReport(accepted=False, error=str(exc),
                                    first_violation_kind="structural")
        except Exception as exc:
            return ConstraintReport(accepted=False, error=f"{type(exc).__name__}: {exc}",
                                    first_violation_kind="structural")
        if mutated.name == base.name:
            mutated.name = base.name

        violations: List[Violation] = []

        # -- structural sanity --------------------------------------------
        if self.check_structure:
            problems = mutated.check()
            if problems:
                violations.append(Violation(kind="structural", predicate="graph_valid",
                                            node=site.consumer,
                                            message=problems[0],
                                            edge=site.anchor_edge))
                return self._finish(violations, mutated)

        # -- no-op detection ----------------------------------------------
        if not self._changed(base, mutated, site, replacement):
            return ConstraintReport(accepted=False, error="replacement is a no-op",
                                    first_violation_kind="pattern", mutated_graph=mutated)

        # -- infer tensor states and re-check the affected nodes -----------
        known = tfg.spec_map()
        internal_edges = self._replacement_edges(base, mutated, site, replacement)
        affected, external_consumers = self._affected_nodes(base, mutated, site,
                                                            replacement)
        # A state change at the site propagates along the data flow, so the
        # output constraints are evaluated over the *transitive* downstream cone
        # rather than only its immediate boundary: a float64 tensor injected at
        # the site may only surface as an operand mismatch several operators
        # later, and that is exactly the violation the framework would report.
        downstream = self._downstream_cone(mutated, affected)
        specs = infer_specs_for_graph(mutated, known)
        self._propagate_states(mutated, specs, internal_edges)

        checked = set(affected) | downstream | set(external_consumers)
        for nid in mutated.topological_order():
            if nid not in checked:
                continue
            node = mutated.nodes[nid]
            operand_specs = [specs.get(e) if e else None for e in node.inputs]
            is_boundary = nid not in affected
            for req in node.requirements:
                if req.kind not in ("input", "internal"):
                    continue
                if is_boundary:
                    if "output" not in self.active:
                        continue
                elif req.kind not in self.active:
                    continue
                # A requirement's own arguments define what it inspects, so they
                # are layered over the node's attributes for evaluation.
                ctx = ReqContext(node=node, attrs={**dict(node.attrs), **req.args},
                                 specs=operand_specs, spec_of=specs, graph=mutated)
                message = evaluate_predicate(req.predicate, ctx)
                if not message:
                    continue
                position = req.args.get("position")
                edge = node.inputs[position] if isinstance(position, int) and \
                    position < len(node.inputs) else ""
                if is_boundary:
                    kind = "output"
                elif req.kind == "internal":
                    # A requirement declared as ``internal`` stays internal: it
                    # constrains data flow *within* the replacement.
                    kind = "internal"
                else:
                    kind = "internal" if edge in internal_edges else "input"
                violations.append(Violation(kind=kind, predicate=req.predicate,
                                            node=nid, message=message, edge=edge))
                if len(violations) >= self.max_violations:
                    return self._finish(violations, mutated)

        # -- operator-local constraints ------------------------------------
        if self.check_operator_local:
            for spec in candidate.operator.constraints(tfg, site):
                if spec.kind not in self.active:
                    continue
                target = mutated.nodes.get(site.consumer)
                if target is None:
                    continue
                if spec.applies_to_ops:
                    targets = [mutated.nodes[n] for n in affected
                               if mutated.nodes[n].op in spec.applies_to_ops]
                else:
                    targets = [target]
                for node in targets:
                    operand_specs = [specs.get(e) if e else None for e in node.inputs]
                    ctx = ReqContext(node=node, attrs={**dict(node.attrs), **spec.args},
                                     specs=operand_specs, spec_of=specs, graph=mutated)
                    message = evaluate_predicate(spec.predicate, ctx)
                    if message:
                        violations.append(Violation(kind=spec.kind,
                                                    predicate=spec.predicate,
                                                    node=node.id, message=message,
                                                    edge=site.anchor_edge))
                        break

        # -- value-domain checks on profiled boundary tensors -------------
        return self._finish(violations, mutated)

    # -- helpers -----------------------------------------------------------
    def _finish(self, violations: List[Violation],
                mutated: Optional[Graph]) -> ConstraintReport:
        accepted = not violations
        first = ""
        for kind in CONSTRAINT_KINDS + ("structural",):
            if any(v.kind == kind for v in violations):
                first = kind
                break
        return ConstraintReport(accepted=accepted, violations=violations,
                                mutated_graph=mutated if accepted else None,
                                first_violation_kind=first)

    @staticmethod
    def _replacement_edges(base: Graph, mutated: Graph, site: Site,
                           replacement: Replacement) -> Set[str]:
        return {e for e in mutated.edges if e not in base.edges}


    @staticmethod
    def _downstream_cone(mutated: Graph, affected: Set[str],
                         max_nodes: int = 4096) -> Set[str]:
        """Operators reachable from the mutated region, excluding the region."""
        cone: Set[str] = set()
        stack = list(affected)
        while stack:
            nid = stack.pop()
            if nid not in mutated.nodes:
                continue
            for e in mutated.out_edges(nid):
                for (consumer, _pos) in mutated.edges[e].consumers:
                    if consumer in affected or consumer in cone:
                        continue
                    cone.add(consumer)
                    stack.append(consumer)
                    if len(cone) >= max_nodes:
                        return cone
        return cone

    @staticmethod
    def _propagate_states(mutated: Graph, specs: Dict[str, Optional[TensorSpec]],
                          internal_edges: Set[str]) -> None:
        """Re-infer tensor states between the site and its downstream boundary.

        ``infer_specs_for_graph`` fills in the *new* edges; this pass carries the
        resulting dtype/shape forward along the data flow so that a requirement
        evaluated further downstream sees the mutated state.
        """
        for nid in mutated.topological_order():
            node = mutated.nodes[nid]
            if node.op == "param":
                continue
            operand_specs = [specs.get(e) for e in node.inputs]
            present = [s for s in operand_specs if s is not None]
            if not present:
                continue
            shape = infer_shape(node.op, present, node.attrs)
            if shape is None:
                shape = tuple(present[0].shape)
            dtype = _dtype_for(node.op, node.attrs, present)
            base_spec = present[0]
            for e in mutated.out_edges(nid):
                if e in internal_edges or e not in specs or specs.get(e) is None:
                    continue
                current = specs[e]
                updated = current.with_(shape=tuple(shape or ()), dtype=dtype,
                                        numel=_numel(shape))
                if updated != current:
                    specs[e] = updated

    @staticmethod
    def _affected_nodes(base: Graph, mutated: Graph, site: Site,
                        replacement: Replacement) -> Tuple[Set[str], Set[str]]:
        """Nodes the mutation rewrote, and external consumers of its outputs.

        ``affected`` are the nodes created by the replacement plus the surviving
        nodes whose attributes were updated -- their own ``R(v)`` is re-checked
        as input/internal constraints.  ``external`` are pre-existing nodes that
        now read a replacement output -- their ``R(v)`` is re-checked as output
        constraints.
        """
        new_edges = {e for e in mutated.edges if e not in base.edges}
        affected: Set[str] = {nid for nid in mutated.nodes if nid not in base.nodes}
        affected |= {nid for nid in replacement.attr_updates if nid in mutated.nodes}
        external: Set[str] = set()
        for e in new_edges:
            for (consumer, _pos) in mutated.edges[e].consumers:
                if consumer in base.nodes and consumer not in affected:
                    external.add(consumer)
        for nid, node in mutated.nodes.items():
            if nid in base.nodes and nid not in affected:
                if base.nodes[nid].inputs != node.inputs:
                    external.add(nid)
        return affected, external

    @staticmethod
    def _changed(base: Graph, mutated: Graph, site: Site,
                 replacement: Replacement) -> bool:
        if len(base.nodes) != len(mutated.nodes) or len(base.edges) != len(mutated.edges):
            return True
        for nid, updates in replacement.attr_updates.items():
            if not updates:
                continue
            old = base.nodes.get(nid)
            if old is None:
                return True
            if any(k not in BOOKKEEPING_ATTRS and old.attrs.get(k) != v
                   for k, v in updates.items()):
                return True
        # Pure rewiring changes neither the node/edge count nor any attribute,
        # so the producer-consumer wiring has to be compared explicitly.
        for nid, node in mutated.nodes.items():
            old = base.nodes.get(nid)
            if old is not None and old.inputs != node.inputs:
                return True
        return False


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def filter_candidates(tfg: TFG, candidates: Sequence[Candidate],
                      checker: Optional[ConstraintChecker] = None
                      ) -> Tuple[List[Candidate], List[Tuple[Candidate, ConstraintReport]]]:
    """Split ``candidates`` into retained and rejected ones."""
    checker = checker or ConstraintChecker()
    kept: List[Candidate] = []
    rejected: List[Tuple[Candidate, ConstraintReport]] = []
    for c in candidates:
        report = checker.filter(tfg, c)
        if report.accepted:
            kept.append(c)
        else:
            rejected.append((c, report))
    return kept, rejected


def legality_rate(reports: Sequence[ConstraintReport]) -> float:
    """``LMR``: the fraction of retained mutants that satisfy the constraints."""
    if not reports:
        return 0.0
    return sum(1 for r in reports if r.accepted) / len(reports)
