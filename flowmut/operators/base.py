"""Mutation operator abstraction: constrained subgraph replacement.

Every FlowMuT mutation is expressed as the replacement of one *bounded subgraph*
with another.  A :class:`MutationOperator` combines three things:

* a :class:`Pattern` stating **where** the operator may be applied,
* a :class:`Replacement` stating **how** the matched subgraph is changed,
* a set of :class:`ConstraintSpec` records stating **when** the change stays
  legal (input, internal and output constraints).

A :class:`Site` pairs an anchor tensor edge ``e`` with the matched subgraph.  A
:class:`Candidate` is a ``(site, operator)`` pair; :mod:`flowmut.operators.pool`
constructs candidates for every edge in ``O(|E|)`` and
:mod:`flowmut.operators.constraints` filters them.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from flowmut.ir.graph import Graph, Node
from flowmut.ir.ops import attach_requirements, canonical_op
from flowmut.ir.specs import TensorSpec
from flowmut.tfg.tfg import TFG

# ---------------------------------------------------------------------------
# Mutable object / primitive taxonomy (Table: operator construction)
# ---------------------------------------------------------------------------

OBJECTS: Tuple[str, ...] = ("Tensor", "Operator", "Interface", "Subgraph")
PRIMITIVES: Tuple[str, ...] = ("Update", "Insertion", "Deletion", "Rewiring")

#: Attributes an operator may write purely so that the *no-op detector* can see a
#: graph edit that changes neither the node/edge count nor any executable
#: configuration (a pure producer-consumer rewiring, for instance).  They carry
#: no execution semantics, so they are ignored when deciding whether two
#: operators perform "the same edit".
BOOKKEEPING_ATTRS: frozenset = frozenset({"rewired_from", "rewired_to"})

#: ``(object, primitive) -> number of operators in the paper's final pool``.
#: Used by :func:`flowmut.operators.pool.pool_composition` for reporting.
PAPER_POOL_TABLE: Dict[Tuple[str, str], Dict[str, int]] = {
    ("Tensor", "Update"): {"cg_extracted": 19, "cg_removed": 14, "cg_retained": 5,
                           "tfg_derived": 8, "overlap": 5, "tfg_only": 3, "total": 8},
    ("Tensor", "Insertion"): {"cg_extracted": 1, "cg_removed": 0, "cg_retained": 1,
                              "tfg_derived": 4, "overlap": 1, "tfg_only": 3, "total": 4},
    ("Operator", "Update"): {"cg_extracted": 46, "cg_removed": 9, "cg_retained": 37,
                             "tfg_derived": 6, "overlap": 5, "tfg_only": 1, "total": 38},
    ("Operator", "Insertion"): {"cg_extracted": 6, "cg_removed": 3, "cg_retained": 3,
                                "tfg_derived": 3, "overlap": 2, "tfg_only": 1, "total": 4},
    ("Operator", "Deletion"): {"cg_extracted": 2, "cg_removed": 1, "cg_retained": 1,
                               "tfg_derived": 1, "overlap": 1, "tfg_only": 0, "total": 1},
    ("Interface", "Update"): {"cg_extracted": 49, "cg_removed": 6, "cg_retained": 43,
                              "tfg_derived": 2, "overlap": 1, "tfg_only": 1, "total": 44},
    ("Interface", "Insertion"): {"cg_extracted": 0, "cg_removed": 0, "cg_retained": 0,
                                 "tfg_derived": 6, "overlap": 0, "tfg_only": 6, "total": 6},
    ("Interface", "Deletion"): {"cg_extracted": 1, "cg_removed": 0, "cg_retained": 1,
                                "tfg_derived": 2, "overlap": 0, "tfg_only": 2, "total": 3},
    ("Interface", "Rewiring"): {"cg_extracted": 0, "cg_removed": 0, "cg_retained": 0,
                                "tfg_derived": 3, "overlap": 0, "tfg_only": 3, "total": 3},
    ("Subgraph", "Update"): {"cg_extracted": 16, "cg_removed": 1, "cg_retained": 15,
                             "tfg_derived": 4, "overlap": 2, "tfg_only": 2, "total": 17},
    ("Subgraph", "Insertion"): {"cg_extracted": 9, "cg_removed": 0, "cg_retained": 9,
                                "tfg_derived": 3, "overlap": 2, "tfg_only": 1, "total": 10},
    ("Subgraph", "Deletion"): {"cg_extracted": 1, "cg_removed": 0, "cg_retained": 1,
                               "tfg_derived": 2, "overlap": 1, "tfg_only": 1, "total": 2},
    ("Subgraph", "Rewiring"): {"cg_extracted": 5, "cg_removed": 1, "cg_retained": 4,
                               "tfg_derived": 4, "overlap": 1, "tfg_only": 3, "total": 7},
}


# ---------------------------------------------------------------------------
# Sites and patterns
# ---------------------------------------------------------------------------

@dataclass
class Site:
    """A matched mutation site anchored at one tensor edge."""

    anchor_edge: str
    #: The node that consumes the anchor edge.
    consumer: str
    #: All nodes belonging to the matched bounded subgraph.
    nodes: List[str] = field(default_factory=list)
    #: Edges crossing into the subgraph from outside.
    input_edges: List[str] = field(default_factory=list)
    #: Edges crossing out of the subgraph.
    output_edges: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.anchor_edge}@{'+'.join(self.nodes)}"


@dataclass
class Pattern:
    """Declarative site matcher evaluated against a TFG."""

    #: Canonical op name the anchor consumer must have.
    op: Optional[str] = None
    op_in: Tuple[str, ...] = ()
    #: Canonical op name of the anchor producer.
    producer_op: Optional[str] = None
    producer_op_in: Tuple[str, ...] = ()
    #: Scope regex the anchor consumer's scope must match.
    scope_regex: Optional[str] = None
    #: Attrs the anchor consumer must (not) carry, exact match on the listed keys.
    require_attrs: Dict[str, Any] = field(default_factory=dict)
    forbid_attrs: Dict[str, Any] = field(default_factory=dict)
    #: Anchor tensor rank / rank set.
    rank: Optional[Tuple[int, ...]] = None
    #: Anchor tensor dtype family: ``"float"``, ``"integral"``, ``"bool"``.
    dtype_family: Optional[str] = None
    #: Bounds on the number of data consumers of the anchor edge.
    min_consumers: Optional[int] = None
    max_consumers: Optional[int] = None
    #: Require the anchor edge to be (not) a view / aliased storage.
    require_view: Optional[bool] = None
    require_alias: Optional[bool] = None
    #: Require the anchor consumer to be (not) a parameter producer.
    require_param_input: Optional[bool] = None
    #: Name of an extra site predicate registered in :data:`SITE_PREDICATES`.
    predicate: Optional[str] = None
    #: Human-readable statement of what the pattern matches.
    description: str = ""

    def match(self, tfg: TFG, edge_id: str) -> Optional[Site]:
        graph = tfg.graph
        edge = graph.edges.get(edge_id)
        if edge is None:
            return None
        producer = graph.nodes.get(edge.producer)
        if producer is None or producer.is_param:
            return None
        if not self.matches_producer(tfg, edge_id):
            return None
        consumers = tfg.data_consumers(edge_id)
        if not consumers:
            return None
        spec = tfg.f(edge_id)
        for consumer_id in consumers:
            node = graph.nodes[consumer_id]
            if not self._node_ok(tfg, node, spec, len(consumers)):
                continue
            site = Site(anchor_edge=edge_id, consumer=consumer_id, nodes=[consumer_id],
                        input_edges=[edge_id],
                        output_edges=[e for e in graph.out_edges(consumer_id)])
            if self.predicate:
                fn = SITE_PREDICATES.get(self.predicate)
                if fn is not None and not fn(tfg, site):
                    continue
            return site
        return None

    def _node_ok(self, tfg: TFG, node: Node, spec: Optional[TensorSpec],
                 n_consumers: int) -> bool:
        op = canonical_op(node.op)
        if self.op is not None and op != self.op:
            return False
        if self.op_in and op not in self.op_in:
            return False
        if self.scope_regex and not re.search(self.scope_regex, node.scope or ""):
            return False
        for k, v in self.require_attrs.items():
            if node.attrs.get(k) != v:
                return False
        for k in self.forbid_attrs:
            if k in node.attrs:
                return False
        if self.rank is not None:
            if spec is None or spec.rank not in self.rank:
                return False
        if self.dtype_family is not None:
            if spec is None:
                return False
            if self.dtype_family == "float" and not spec.is_floating:
                return False
            if self.dtype_family == "integral" and not spec.is_integral:
                return False
            if self.dtype_family == "bool" and spec.dtype != "bool":
                return False
        if self.min_consumers is not None and n_consumers < self.min_consumers:
            return False
        if self.max_consumers is not None and n_consumers > self.max_consumers:
            return False
        if self.require_view is not None:
            if spec is None or bool(spec.is_view) != self.require_view:
                return False
        if self.require_alias is not None:
            if spec is None or bool(spec.alias_of) != self.require_alias:
                return False
        if self.require_param_input is not None:
            has_param = False
            for e in node.inputs:
                if e and e in tfg.graph.edges:
                    if tfg.graph.nodes[tfg.graph.edges[e].producer].is_param:
                        has_param = True
                        break
            if has_param != self.require_param_input:
                return False
        return True

    def matches_producer(self, tfg: TFG, edge_id: str) -> bool:
        if self.producer_op is None and not self.producer_op_in:
            return True
        edge = tfg.graph.edges.get(edge_id)
        if edge is None:
            return False
        op = canonical_op(tfg.graph.nodes[edge.producer].op)
        if self.producer_op is not None and op != self.producer_op:
            return False
        if self.producer_op_in and op not in self.producer_op_in:
            return False
        return True


#: Extension point for site predicates that need the whole TFG.
SITE_PREDICATES: Dict[str, Callable[[TFG, Site], bool]] = {}


def register_site_predicate(name: str):
    def deco(fn):
        SITE_PREDICATES[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
# Replacement templates
# ---------------------------------------------------------------------------

@dataclass
class NodeTemplate:
    """A node to be created by a replacement.

    ``inputs`` entries are either an existing graph edge id or a reference of the
    form ``"@key"`` / ``"@key#index"`` naming the output of another template.
    Attr values may contain the placeholders ``"{anchor}"`` (the anchor edge id)
    and ``"{consumer}"`` (the matched consumer node id).
    """

    key: str
    op: str
    inputs: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    scope: str = ""
    num_outputs: int = 1
    name: str = ""

    def resolved_attrs(self, site: Site) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.attrs.items():
            if isinstance(v, str):
                out[k] = v.replace("{anchor}", site.anchor_edge).replace(
                    "{consumer}", site.consumer)
            else:
                out[k] = v
        return out

    def resolved_scope(self, site: Site) -> str:
        return self.scope.replace("{scope}", site.consumer)


@dataclass
class Replacement:
    """How a matched subgraph is rewritten."""

    templates: List[NodeTemplate] = field(default_factory=list)
    #: old edge id -> ref, for edges whose external consumers must be reconnected.
    outputs: Dict[str, str] = field(default_factory=dict)
    #: Explicit ``(consumer node id, input position, ref)`` rewires.
    rewires: List[Tuple[str, int, str]] = field(default_factory=list)
    #: In-place attribute updates on nodes that survive the replacement.  This is
    #: how "Update" operators change an operator's configuration without
    #: disturbing the surrounding topology.
    attr_updates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: Nodes removed from the graph; defaults to ``site.nodes``.
    remove: Optional[List[str]] = None
    #: Nodes that must be preserved even when they belong to ``site.nodes``.
    keep: List[str] = field(default_factory=list)
    note: str = ""

    def remove_set(self, site: Site) -> List[str]:
        base = list(site.nodes if self.remove is None else self.remove)
        return [n for n in base if n not in set(self.keep)]


@dataclass
class ConstraintSpec:
    """A declarative constraint attached to an operator.

    ``kind`` is ``"input"``, ``"internal"`` or ``"output"``; ``predicate`` names
    an entry of :mod:`flowmut.tfg.predicates`; ``when`` limits the check to sites
    matching a pattern fragment.
    """

    kind: str
    predicate: str
    args: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    #: Restrict the constraint to replacement nodes whose op is in this set.
    applies_to_ops: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

class MutationOperator(ABC):
    """One mutation operator of the FlowMuT pool."""

    #: Stable identifier, e.g. ``"tensor.update.dtype_cast"``.
    name: str = "unnamed"
    target_object: str = "Tensor"
    primitive: str = "Update"
    #: ``"CG"`` for operators adapted from prior work, ``"TFG"`` for operators
    #: derived from the TFG, ``"CG+TFG"`` for the deduplicated overlap.
    source: str = "CG"
    #: Optional reference into the literature the operator was adapted from.
    reference: str = ""
    #: Free parameters that produced this instance (they are part of the
    #: operator identity, matching the paper's "same target, edit, parameters and
    #: applicability conditions" deduplication rule).  Always a plain dict on an
    #: instance; subclasses that do not call ``__init__`` still see ``{}``.
    params: Dict[str, Any] = {}
    description: str = ""

    def __init__(self, **params: Any):
        self.params = dict(params)

    # -- matching ----------------------------------------------------------
    @property
    @abstractmethod
    def pattern(self) -> Pattern:
        """Where the operator may be applied."""

    @abstractmethod
    def build_replacement(self, tfg: TFG, site: Site) -> Optional[Replacement]:
        """How the matched subgraph is changed (``None`` = inapplicable)."""

    # -- constraints -------------------------------------------------------
    def constraints(self, tfg: TFG, site: Site) -> List[ConstraintSpec]:
        """Operator-local constraints in addition to ``R(v)`` checking."""
        return []

    # -- identity ----------------------------------------------------------
    @property
    def category(self) -> Tuple[str, str]:
        return (self.target_object, self.primitive)

    def key(self) -> Tuple[Any, ...]:
        """Deduplication key: target, edit, parameters and conditions."""
        p = self.pattern
        return (
            self.name,
            tuple(sorted((k, repr(v)) for k, v in self.params.items())),
            p.op, tuple(sorted(p.op_in)), p.producer_op, p.scope_regex,
            tuple(sorted((k, repr(v)) for k, v in p.require_attrs.items())),
            tuple(sorted(p.forbid_attrs)), p.rank, p.dtype_family,
            p.min_consumers, p.max_consumers, p.predicate,
        )

    def to_dict(self) -> Dict[str, Any]:
        p = self.pattern
        return {
            "name": self.name,
            "object": self.target_object,
            "primitive": self.primitive,
            "source": self.source,
            "reference": self.reference,
            "params": {k: (v if isinstance(v, (str, int, float, bool, type(None)))
                           else repr(v)) for k, v in self.params.items()},
            "pattern": {"op": p.op, "op_in": list(p.op_in),
                        "producer_op": p.producer_op,
                        "scope_regex": p.scope_regex,
                        "require_attrs": {k: repr(v) for k, v in p.require_attrs.items()},
                        "forbid_attrs": sorted(p.forbid_attrs),
                        "rank": p.rank, "dtype_family": p.dtype_family},
            "description": self.description or p.description,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        ps = ",".join(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        return f"<{self.name}{'(' + ps + ')' if ps else ''}>"


class FunctionalOperator(MutationOperator):
    """Convenience base for operators defined by plain callables.

    Keeps the pool file declarative: an operator only supplies a pattern and a
    function that builds its replacement.
    """

    def __init__(self, name: str, target_object: str, primitive: str, source: str,
                 pattern: Pattern, build: Callable[[TFG, Site], Optional[Replacement]],
                 constraints: Optional[Callable[[TFG, Site], List[ConstraintSpec]]] = None,
                 reference: str = "", description: str = "", **params: Any):
        super().__init__(**params)
        self.name = name
        self.target_object = target_object
        self.primitive = primitive
        self.source = source
        self.reference = reference
        self.description = description or pattern.description
        self._pattern = pattern
        self._build = build
        self._constraints = constraints

    @property
    def pattern(self) -> Pattern:
        return self._pattern

    def build_replacement(self, tfg: TFG, site: Site) -> Optional[Replacement]:
        return self._build(tfg, site)

    def constraints(self, tfg: TFG, site: Site) -> List[ConstraintSpec]:
        return self._constraints(tfg, site) if self._constraints else []


@dataclass
class Candidate:
    """A ``(site, operator)`` pair, before constraint filtering."""

    site: Site
    operator: MutationOperator
    iteration: int = 0

    @property
    def anchor(self) -> str:
        return self.site.anchor_edge

    @property
    def key(self) -> str:
        return f"{self.site.key}|{self.operator.name}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor": self.site.anchor_edge,
            "consumer": self.site.consumer,
            "nodes": list(self.site.nodes),
            "operator": self.operator.name,
            "object": self.operator.target_object,
            "primitive": self.operator.primitive,
            "params": {k: repr(v) for k, v in self.operator.params.items()},
            "iteration": self.iteration,
        }


# ---------------------------------------------------------------------------
# Applying a replacement
# ---------------------------------------------------------------------------

class ReplacementError(Exception):
    """Raised when a replacement cannot be materialised on the graph."""


_REF_RE = re.compile(r"^@(?P<key>[A-Za-z0-9_]+)(?:#(?P<index>\d+))?$")


def apply_replacement(graph: Graph, site: Site, replacement: Replacement,
                      operator_name: str = "") -> Graph:
    """Materialise ``replacement`` on a copy of ``graph``.

    The returned graph is structurally valid; it is not yet checked against the
    operator requirements (that is :mod:`flowmut.operators.constraints`' job).
    """
    g = graph.copy()
    remove = set(replacement.remove_set(site))
    produced: Dict[str, List[str]] = {}

    # 1. Capture the external consumers of every replaced output *before* the
    #    nodes disappear, so that boundary reconnection can be applied.
    external: Dict[str, List[Tuple[str, int]]] = {}
    for old_edge in list(replacement.outputs):
        if old_edge not in g.edges:
            continue
        external[old_edge] = [(c, p) for (c, p) in g.edges[old_edge].consumers
                              if c not in remove]

    # 2. Create the templates in declaration order.
    for tmpl in replacement.templates:
        scope = tmpl.resolved_scope(site) or (
            g.nodes[site.consumer].scope if site.consumer in g.nodes else "")
        if tmpl.op == "param":
            attrs = tmpl.resolved_attrs(site)
            shape = attrs.pop("shape", (1,))
            init = attrs.pop("init", "normal")
            edge = g.add_param(tmpl.name or tmpl.key, TensorSpec(shape=tuple(shape),
                                                                 requires_grad=True),
                               init=init, extra=attrs, scope=scope or "params")
            produced[tmpl.key] = [edge.id]
            continue
        inputs: List[str] = []
        for ref in tmpl.inputs:
            inputs.append(_resolve_ref(g, produced, ref, site))
        node = g.add_node(tmpl.op, inputs,
                          attrs=tmpl.resolved_attrs(site),
                          scope=scope,
                          name=tmpl.name or tmpl.op,
                          num_outputs=tmpl.num_outputs,
                          origin=f"mutation:{operator_name}")
        produced[tmpl.key] = g.out_edges(node.id)

    # 3. Reconnect external consumers of replaced outputs.
    for old_edge, consumers in external.items():
        ref = replacement.outputs[old_edge]
        target = _resolve_ref(g, produced, ref, site)
        for (consumer, position) in consumers:
            if consumer not in g.nodes:
                continue
            g.disconnect_input(consumer, position)
            g.connect_input(consumer, position, target)

    # 4. Apply explicit rewires.
    for (consumer, position, ref) in replacement.rewires:
        if consumer not in g.nodes:
            continue
        target = _resolve_ref(g, produced, ref, site)
        g.disconnect_input(consumer, position)
        g.connect_input(consumer, position, target)

    # 5. Remove the replaced nodes.
    for nid in remove:
        if nid in g.nodes:
            g.remove_node(nid)

    # 6. Apply in-place attribute updates on surviving nodes.
    for nid, updates in replacement.attr_updates.items():
        if nid in g.nodes:
            g.nodes[nid].attrs.update(updates)
            g.nodes[nid].origin = f"mutation:{operator_name}"

    # 7. Drop edges that no longer have a producer or a consumer.
    for eid in list(g.edges):
        edge = g.edges[eid]
        if edge.producer not in g.nodes:
            g.edges.pop(eid, None)
    for eid in list(g.edges):
        g.edges[eid].consumers = [(c, p) for (c, p) in g.edges[eid].consumers
                                  if c in g.nodes]
    g.inputs = [e for e in g.inputs if e in g.edges]
    g.outputs = [e for e in g.outputs if e in g.edges]
    attach_requirements(g)
    return g


def _resolve_ref(g: Graph, produced: Dict[str, List[str]], ref: str, site: Site) -> str:
    """Resolve a template reference to a concrete edge id."""
    if ref == "{anchor}":
        return site.anchor_edge
    m = _REF_RE.match(ref)
    if m:
        key = m.group("key")
        index = int(m.group("index") or 0)
        edges = produced.get(key)
        if not edges:
            raise ReplacementError(f"reference {ref!r} names an unknown template output")
        if index >= len(edges):
            raise ReplacementError(f"reference {ref!r} exceeds the template arity")
        return edges[index]
    if ref in g.edges:
        return ref
    raise ReplacementError(f"reference {ref!r} is neither an edge nor a template output")


# ---------------------------------------------------------------------------
# Operator cohort
# ---------------------------------------------------------------------------

@dataclass
class OperatorPool:
    """The validated pool ``M`` of mutation operators."""

    operators: List[MutationOperator] = field(default_factory=list)
    #: Sorted by ``(object, primitive)`` for reproducible ordering.
    index: Dict[str, MutationOperator] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.operators = sorted(
            self.operators,
            key=lambda o: (OBJECTS.index(o.target_object) if o.target_object in OBJECTS else 9,
                           PRIMITIVES.index(o.primitive) if o.primitive in PRIMITIVES else 9,
                           o.name),
        )
        self.index = {o.name: o for o in self.operators}

    def __len__(self) -> int:
        return len(self.operators)

    def __iter__(self):
        return iter(self.operators)

    def get(self, name: str) -> MutationOperator:
        return self.index[name]

    def by_category(self) -> Dict[Tuple[str, str], List[MutationOperator]]:
        out: Dict[Tuple[str, str], List[MutationOperator]] = {}
        for op in self.operators:
            out.setdefault(op.category, []).append(op)
        return out

    def summary(self) -> Dict[str, Any]:
        table: Dict[str, Dict[str, int]] = {}
        for op in self.operators:
            cell = table.setdefault(f"{op.target_object}/{op.primitive}", {})
            cell[op.source] = cell.get(op.source, 0) + 1
            cell["total"] = cell.get("total", 0) + 1
        return {
            "size": len(self.operators),
            "by_category": table,
            "by_source": _count_by(self.operators, lambda o: o.source),
            "by_object": _count_by(self.operators, lambda o: o.target_object),
        }


def _count_by(items: Iterable[Any], fn) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        k = fn(it)
        out[k] = out.get(k, 0) + 1
    return out
