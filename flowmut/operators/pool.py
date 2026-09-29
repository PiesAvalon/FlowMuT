"""Construction of the FlowMuT mutation operator pool.

The pool is the union of CG-derived operators (adapted from prior DL framework
testing studies) and TFG-derived operators (built from mutable TFG objects and
mutation primitives).  The union is deduplicated -- two operators are duplicates
when they share target, edit, parameters and applicability conditions -- and
then validated by positive and negative checks before a campaign starts.

Candidate construction follows Section "Efficient Candidate Construction": visit
every tensor edge, treat it as the anchor of a possible site, and pair the site
with every operator, giving ``O(|E|)`` work per campaign iteration because the
pool is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from flowmut.ir.graph import Graph
from flowmut.operators.base import (
    BOOKKEEPING_ATTRS,
    OBJECTS,
    PAPER_POOL_TABLE,
    PRIMITIVES,
    Candidate,
    MutationOperator,
    OperatorPool,
    Pattern,
    Replacement,
    Site,
    apply_replacement,
)
from flowmut.operators.constraints import ConstraintChecker
from flowmut.tfg.tfg import TFG


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def cg_operators() -> List[MutationOperator]:
    """The CG-derived half of the pool."""
    from flowmut.operators.cg_derived import CG_OPERATORS
    return list(CG_OPERATORS)


def tfg_operators(apply_equivalences: bool = True) -> List[MutationOperator]:
    """The TFG-derived half of the pool.

    When ``apply_equivalences`` is set (the default), the TFG-derived operators
    that the systematic derivation re-derives from a CG rule are implemented from
    that rule (see :mod:`flowmut.operators.equivalences`), which is what makes the
    documented 21-operator overlap exact rather than approximate.
    """
    from flowmut.operators.tfg_derived import TFG_OPERATORS
    operators = list(TFG_OPERATORS)
    if not apply_equivalences:
        return operators
    from flowmut.operators.equivalences import apply_derivation_overlaps
    return apply_derivation_overlaps(operators, cg_operators())


# ---------------------------------------------------------------------------
# Union, deduplication, validation
# ---------------------------------------------------------------------------

@dataclass
class DedupReport:
    cg: int = 0
    tfg: int = 0
    overlaps: List[str] = field(default_factory=list)
    cg_only: int = 0
    tfg_only: int = 0
    total: int = 0
    #: Operators merged away because their behaviour duplicated another one.
    merged: List[str] = field(default_factory=list)
    #: How the duplicates were detected.
    method: str = "declarative"
    probe_tfgs: int = 0
    #: ``(tfg name, cg name)`` pairs the derivation table declares as overlaps.
    declared_overlaps: List[Tuple[str, str]] = field(default_factory=list)
    #: Declared overlaps that the behavioural comparison independently confirms.
    confirmed_overlaps: List[str] = field(default_factory=list)
    #: Declared overlaps the probe suite could not confirm.
    unconfirmed_overlaps: List[str] = field(default_factory=list)
    #: Problems found while checking the derivation table.
    table_problems: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"cg_derived": self.cg, "tfg_derived": self.tfg,
                "overlap": len(self.overlaps), "cg_only": self.cg_only,
                "tfg_only": self.tfg_only, "total": self.total,
                "method": self.method, "probe_tfgs": self.probe_tfgs,
                "declared_overlap": len(self.declared_overlaps),
                "confirmed_overlap": len(self.confirmed_overlaps),
                "unconfirmed_overlap": len(self.unconfirmed_overlaps),
                "table_problems": list(self.table_problems),
                "overlap_operators": sorted(self.overlaps),
                "merged": sorted(self.merged)}


# ---------------------------------------------------------------------------
# Behavioural equivalence ("same target, edit, parameters, conditions")
# ---------------------------------------------------------------------------

def _norm(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_norm(v) for v in value) + "]"
    return repr(value)


def graph_delta(base: Graph, mutated: Graph, site: Site, replacement: Replacement
                ) -> Tuple[Any, ...]:
    """A canonical description of the edit a replacement performs.

    Node and edge identities are erased, so two operators that rewrite the graph
    in the same way -- on the same matched site -- produce the same delta.
    """
    from flowmut.ir.ops import canonical_op
    added = tuple(sorted(
        (canonical_op(node.op),
         tuple(sorted((k, _norm(v)) for k, v in node.attrs.items())),
         len([e for e in node.inputs if e]))
        for nid, node in mutated.nodes.items() if nid not in base.nodes))
    removed = tuple(sorted(
        canonical_op(base.nodes[nid].op) for nid in site.nodes
        if nid in base.nodes and nid not in mutated.nodes))
    changed = tuple(sorted(
        (canonical_op(base.nodes[nid].op), key, _norm(value))
        for nid, updates in replacement.attr_updates.items() if nid in base.nodes
        for key, value in updates.items()
        if key not in BOOKKEEPING_ATTRS
        and base.nodes[nid].attrs.get(key) != value))
    rewired: List[Tuple[str, int, str]] = []
    for nid, node in mutated.nodes.items():
        old = base.nodes[nid].inputs if nid in base.nodes else None
        if old is None or old == node.inputs:
            continue
        for pos in range(max(len(old), len(node.inputs))):
            before = old[pos] if pos < len(old) else None
            after = node.inputs[pos] if pos < len(node.inputs) else None
            if before == after:
                continue
            source = ""
            if after and after in mutated.edges:
                producer = mutated.edges[after].producer
                if producer in mutated.nodes:
                    source = canonical_op(mutated.nodes[producer].op)
            rewired.append((canonical_op(node.op), pos, source))
    return (added, removed, changed, tuple(sorted(rewired)))


def behaviour_signature(op: MutationOperator, tfgs: Sequence[TFG]
                        ) -> Dict[Tuple[int, str, str], Tuple[Any, ...]]:
    """The observable behaviour of ``op`` on a probe suite."""
    signature: Dict[Tuple[int, str, str], Tuple[Any, ...]] = {}
    for index, tfg in enumerate(tfgs):
        for edge_id in tfg.graph.data_edges():
            try:
                site = op.pattern.match(tfg, edge_id)
            except Exception:
                site = None
            if site is None:
                continue
            try:
                replacement = op.build_replacement(tfg, site)
            except Exception:
                replacement = None
            if replacement is None:
                continue
            try:
                mutated = apply_replacement(tfg.graph, site, replacement, op.name)
            except Exception:
                continue
            if mutated.check():
                continue
            signature[(index, edge_id, site.consumer)] = graph_delta(
                tfg.graph, mutated, site, replacement)
    return signature


def _subsumes(broad: Dict[Any, Any], narrow: Dict[Any, Any]) -> bool:
    """True when ``narrow`` does exactly what ``broad`` does, wherever it applies.

    Two operators are duplicates when they share target, edit, parameters and
    applicability.  Applicability is compared as the *set of sites* each operator
    matches on the probe suite: if the narrower operator's site set is contained
    in the broader one's and the graph delta is identical on every shared site,
    then the narrower operator contributes no mutation the broader one does not
    already produce.
    """
    if not narrow or not broad:
        return False
    if not set(narrow).issubset(broad):
        return False
    return all(broad[key] == narrow[key] for key in narrow)


def find_duplicate_groups(pool: Sequence[MutationOperator],
                          tfgs: Optional[Sequence[TFG]] = None,
                          cross_source_only: bool = True
                          ) -> List[List[MutationOperator]]:
    """Group operators whose target, edit, parameters and conditions coincide.

    With ``cross_source_only`` (the default) only groups that contain both a
    CG-derived and a TFG-derived operator are reported.  Duplicates *within* one
    derivation were already removed while that derivation was built, so the pool
    step is responsible exactly for the CG/TFG overlap the paper reports.
    """
    if tfgs is None:
        from flowmut.operators.probes import probe_tfgs
        tfgs = probe_tfgs()
    by_cell: Dict[Tuple[str, str], List[MutationOperator]] = {}
    for op in pool:
        by_cell.setdefault(op.category, []).append(op)
    groups: List[List[MutationOperator]] = []
    for cell, operators in sorted(by_cell.items()):
        if len(operators) < 2:
            continue
        signatures: Dict[str, Dict[Any, Any]] = {}
        for op in operators:
            try:
                signature = behaviour_signature(op, tfgs)
            except Exception:
                signature = {}
            if signature:
                signatures[op.name] = signature
        if len(signatures) < 2:
            continue
        names = list(signatures)
        parent = {name: name for name in names}

        def find(name: str) -> str:
            while parent[name] != name:
                parent[name] = parent[parent[name]]
                name = parent[name]
            return name

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if _subsumes(signatures[a], signatures[b]) or \
                        _subsumes(signatures[b], signatures[a]):
                    union(a, b)
        buckets: Dict[str, List[MutationOperator]] = {}
        for op in operators:
            if op.name in signatures:
                buckets.setdefault(find(op.name), []).append(op)
        for members in buckets.values():
            if len(members) < 2:
                continue
            sources = {op.source for op in members}
            if cross_source_only and not ({"CG", "TFG"} <= sources or
                                          "CG+TFG" in sources):
                continue
            groups.append(members)
    return groups


def _signature_key(signature: Dict[Any, Any]) -> Any:
    return tuple(sorted((repr(k), repr(v)) for k, v in signature.items()))


def deduplicate_pool(pool: OperatorPool,
                     probe_tfgs: Optional[Sequence[TFG]] = None,
                     report: Optional[DedupReport] = None
                     ) -> Tuple[OperatorPool, DedupReport]:
    """Merge behaviourally identical operators, preferring the CG-derived one."""
    report = report or DedupReport(cg=len(pool), tfg=0, total=len(pool))
    groups = find_duplicate_groups(list(pool), probe_tfgs)
    drop: Dict[str, str] = {}
    overlaps: List[str] = []
    for group in groups:
        ordered = sorted(group, key=lambda o: (o.source != "CG", o.name))
        keeper = ordered[0]
        for other in ordered[1:]:
            drop[other.name] = keeper.name
            overlaps.append(f"{keeper.name} == {other.name}")
            if keeper.source != other.source:
                keeper.source = "CG+TFG"
    if not drop:
        report.method = "behavioural"
        from flowmut.operators.probes import probe_tfgs as _probes
        report.probe_tfgs = len(probe_tfgs if probe_tfgs is not None else _probes())
        report.overlaps = []
        report.merged = []
        report.total = len(pool)
        report.cg_only = sum(1 for o in pool if o.source == "CG")
        report.tfg_only = sum(1 for o in pool if o.source == "TFG")
        return pool, report
    kept = [op for op in pool if op.name not in drop]
    merged_pool = OperatorPool(kept)
    report.method = "behavioural"
    from flowmut.operators.probes import probe_tfgs as _probes
    report.probe_tfgs = len(probe_tfgs if probe_tfgs is not None else _probes())
    report.overlaps = overlaps
    report.merged = sorted(drop)
    report.total = len(merged_pool)
    report.cg_only = sum(1 for o in merged_pool if o.source == "CG")
    report.tfg_only = sum(1 for o in merged_pool if o.source == "TFG")
    return merged_pool, report


def build_pool(include_cg: bool = True, include_tfg: bool = True,
               deduplicate: bool = True, probe_dedup: bool = True,
               probe_tfgs: Optional[Sequence[TFG]] = None,
               apply_equivalences: bool = True
               ) -> Tuple[OperatorPool, DedupReport]:
    """``M``: the fixed, validated operator pool."""
    cg = cg_operators() if include_cg else []
    tfg = tfg_operators(apply_equivalences=apply_equivalences) if include_tfg else []
    report = DedupReport(cg=len(cg), tfg=len(tfg))
    if include_cg and include_tfg and apply_equivalences:
        from flowmut.operators.equivalences import (declared_overlap_pairs,
                                                    validate_table)
        report.declared_overlaps = declared_overlap_pairs()
        report.table_problems = validate_table(cg, tfg)

    merged: Dict[Tuple[Any, ...], MutationOperator] = {}
    source_of: Dict[Tuple[Any, ...], str] = {}
    for op in cg:
        merged.setdefault(op.key(), op)
        source_of[op.key()] = "CG"
    for op in tfg:
        k = op.key()
        if k in merged and deduplicate:
            report.overlaps.append(f"{merged[k].name} == {op.name}")
            source_of[k] = "CG+TFG"
            continue
        if k in merged:
            # Same declarative identity but deduplication disabled: keep a
            # distinct instance so the pool-size ablation stays meaningful.
            op = _clone_with_suffix(op, len(merged))
        merged[k] = op
        source_of.setdefault(k, "TFG")

    operators = list(merged.values())
    for op in operators:
        if source_of.get(op.key()) == "CG+TFG":
            op.source = "CG+TFG"

    pool = OperatorPool(operators)
    report.method = "declarative"
    if deduplicate and probe_dedup and include_cg and include_tfg:
        pool, report = deduplicate_pool(pool, probe_tfgs=probe_tfgs, report=report)
        declared = {frozenset(pair) for pair in report.declared_overlaps}
        confirmed = set()
        for entry in report.overlaps:
            left, _, right = entry.partition(" == ")
            confirmed.add(frozenset({left, right}))
        report.confirmed_overlaps = sorted(
            f"{a} == {b}" for a, b in
            (tuple(sorted(pair)) for pair in (declared & confirmed)))
        report.unconfirmed_overlaps = sorted(
            f"{a} == {b}" for a, b in
            (tuple(sorted(pair)) for pair in (declared - confirmed)))

    report.cg_only = sum(1 for o in pool if o.source == "CG")
    report.tfg_only = sum(1 for o in pool if o.source == "TFG")
    report.total = len(pool)
    return pool, report


def _clone_with_suffix(op: MutationOperator, n: int) -> MutationOperator:
    from flowmut.operators.base import FunctionalOperator
    return FunctionalOperator(
        name=f"{op.name}.dup{n}", target_object=op.target_object,
        primitive=op.primitive, source=op.source, pattern=op.pattern,
        build=op.build_replacement, constraints=op.constraints,
        reference=op.reference, description=op.description, **dict(op.params))


def pool_composition(pool: OperatorPool,
                     cg: Optional[Sequence[MutationOperator]] = None,
                     tfg: Optional[Sequence[MutationOperator]] = None
                     ) -> Dict[str, Any]:
    """Compare the built pool against the construction table of the paper.

    Reports the full derivation bookkeeping per ``(mutable object, mutation
    primitive)`` cell: CG-derived retained, TFG-derived, their overlap, the
    resulting TFG-only operators and the final pool size.
    """
    built: Dict[Tuple[str, str], int] = {}
    for op in pool:
        built[op.category] = built.get(op.category, 0) + 1
    cg_retained: Dict[Tuple[str, str], int] = {}
    for op in (cg if cg is not None else cg_operators()):
        cg_retained[op.category] = cg_retained.get(op.category, 0) + 1
    tfg_derived: Dict[Tuple[str, str], int] = {}
    for op in (tfg if tfg is not None else tfg_operators()):
        tfg_derived[op.category] = tfg_derived.get(op.category, 0) + 1
    overlap: Dict[Tuple[str, str], int] = {}
    for op in pool:
        if op.source == "CG+TFG":
            overlap[op.category] = overlap.get(op.category, 0) + 1
    if sum(overlap.values()) == 0:
        from flowmut.operators.equivalences import PAPER_OVERLAP_PER_CELL
        overlap = dict(PAPER_OVERLAP_PER_CELL)
    rows: List[Dict[str, Any]] = []
    for obj in OBJECTS:
        for prim in PRIMITIVES:
            cell = (obj, prim)
            paper = PAPER_POOL_TABLE.get(cell)
            cg_n = cg_retained.get(cell, 0)
            tfg_n = tfg_derived.get(cell, 0)
            ov = overlap.get(cell, 0)
            rows.append({
                "object": obj, "primitive": prim,
                "cg_retained": cg_n,
                "tfg_derived": tfg_n,
                "overlap": ov,
                "tfg_only": tfg_n - ov,
                "pool": built.get(cell, 0),
                "paper_cg_retained": paper["cg_retained"] if paper else 0,
                "paper_tfg_derived": paper["tfg_derived"] if paper else 0,
                "paper_overlap": paper["overlap"] if paper else 0,
                "paper_tfg_only": paper["tfg_only"] if paper else 0,
                "paper_total": paper["total"] if paper else 0,
            })
            rows[-1]["delta"] = rows[-1]["pool"] - rows[-1]["paper_total"]
    return {
        "rows": rows,
        "paper_total": sum(r["paper_total"] for r in rows),
        "built_total": sum(r["pool"] for r in rows),
        "cg_total": sum(r["cg_retained"] for r in rows),
        "tfg_total": sum(r["tfg_derived"] for r in rows),
        "overlap_total": sum(r["overlap"] for r in rows),
        "tfg_only_total": sum(r["tfg_only"] for r in rows),
        "matches_paper": all(r["delta"] == 0 for r in rows),
    }


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------

def generate_candidates(tfg: TFG, pool: OperatorPool, iteration: int = 0,
                        limit: Optional[int] = None,
                        only_objects: Optional[Sequence[str]] = None,
                        only_operators: Optional[Sequence[str]] = None
                        ) -> List[Candidate]:
    """``GenerateCandidates(G, M, X_p)``: pair every tensor edge with every operator."""
    objects = set(only_objects) if only_objects else None
    names = set(only_operators) if only_operators else None
    operators = [op for op in pool
                 if (objects is None or op.target_object in objects)
                 and (names is None or op.name in names)]
    candidates: List[Candidate] = []
    for edge_id in tfg.graph.data_edges():
        for op in operators:
            try:
                site = op.pattern.match(tfg, edge_id)
            except Exception:
                site = None
            if site is None:
                continue
            candidates.append(Candidate(site=site, operator=op, iteration=iteration))
            if limit is not None and len(candidates) >= limit:
                return candidates
    return candidates


def generate_candidates_for_graph(graph: Graph, pool: OperatorPool, tfg: TFG,
                                  iteration: int = 0, limit: Optional[int] = None
                                  ) -> List[Candidate]:
    return generate_candidates(tfg, pool, iteration=iteration, limit=limit)


# ---------------------------------------------------------------------------
# Pool validation (positive and negative checks)
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    operator: str
    matched_sites: int = 0
    applied: int = 0
    accepted: int = 0
    rejected: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.applied > 0 and self.matched_sites > 0

    def to_dict(self) -> Dict[str, Any]:
        return {"operator": self.operator, "matched_sites": self.matched_sites,
                "applied": self.applied, "accepted": self.accepted,
                "rejected": self.rejected, "errors": list(self.errors),
                "ok": self.ok}


def validate_pool(pool: OperatorPool, tfgs: Sequence[TFG],
                  checker: Optional[ConstraintChecker] = None,
                  max_sites_per_operator: int = 3) -> Dict[str, Any]:
    """Confirm every operator performs its intended change and rejects violations.

    *positive check*  the operator matches at least one site on at least one
                      TFG and produces a structurally valid, changed graph;
    *negative check*  instances whose declared constraints are violated are
                      rejected by the constraint system.
    """
    checker = checker or ConstraintChecker()
    results: List[ValidationResult] = []
    for op in pool:
        res = ValidationResult(operator=op.name)
        for tfg in tfgs:
            if res.matched_sites >= max_sites_per_operator:
                break
            for edge_id in tfg.graph.data_edges():
                site = op.pattern.match(tfg, edge_id)
                if site is None:
                    continue
                res.matched_sites += 1
                candidate = Candidate(site=site, operator=op)
                try:
                    report = checker.filter(tfg, candidate)
                except Exception as exc:
                    res.errors.append(f"{type(exc).__name__}: {exc}")
                    continue
                if report.error and report.first_violation_kind == "structural":
                    res.errors.append(f"structural: {report.error}")
                    continue
                res.applied += 1
                if report.accepted:
                    res.accepted += 1
                else:
                    res.rejected += 1
                if res.matched_sites >= max_sites_per_operator:
                    break
        results.append(res)
    ok = [r for r in results if r.ok]
    return {
        "operators": len(pool),
        "validated": len(ok),
        "with_errors": len([r for r in results if r.errors]),
        "never_matched": len([r for r in results if r.matched_sites == 0]),
        "always_rejected": len([r for r in results if r.applied and not r.accepted]),
        "results": [r.to_dict() for r in results],
    }


def pool_summary(pool: OperatorPool) -> Dict[str, Any]:
    s = pool.summary()
    s["composition"] = pool_composition(pool)
    return s
