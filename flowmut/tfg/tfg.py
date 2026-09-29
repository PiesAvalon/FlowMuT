"""The Tensor Flow Graph: ``TFG = (G, F, R)``.

* ``G = (V, E)`` is the operator topology, carried by the IR :class:`~flowmut.ir.Graph`.
* ``F(e)`` is the *edge annotation*: the runtime tensor states observed on edge
  ``e`` across the profiling inputs, with value summaries, shape, rank, dtype,
  device, layout, aliasing and gradient state.
* ``R(v)`` is the *node annotation*: the operator input requirements attached to
  node ``v``.

``G`` identifies mutation sites, ``F`` decides which operators can be
instantiated, and ``R`` checks compatibility with the surrounding model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from flowmut.ir.graph import Graph, Node
from flowmut.ir.specs import OpRequirement, TensorSpec, ValueProfile
from flowmut.tfg.predicates import ReqContext, evaluate as evaluate_predicate


# ---------------------------------------------------------------------------
# Constraint violations
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    """A rejected candidate, tagged with the constraint category it broke."""

    kind: str          # "input" | "internal" | "output" | "pattern" | "structural"
    predicate: str
    node: str
    message: str
    edge: str = ""

    def __str__(self) -> str:
        where = f"node {self.node}" + (f" via edge {self.edge}" if self.edge else "")
        return f"[{self.kind}:{self.predicate}] {where}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "predicate": self.predicate, "node": self.node,
                "edge": self.edge, "message": self.message}


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def _merge_values(specs: Sequence[TensorSpec]) -> Optional[ValueProfile]:
    vps = [s.value for s in specs if s.value is not None]
    if not vps:
        return None
    n = len(vps)
    return ValueProfile(
        min=min(v.min for v in vps),
        max=max(v.max for v in vps),
        mean=sum(v.mean for v in vps) / n,
        std=max(v.std for v in vps),
        abs_mean=sum(v.abs_mean for v in vps) / n,
        l2=max(v.l2 for v in vps),
        nan_count=sum(v.nan_count for v in vps),
        inf_count=sum(v.inf_count for v in vps),
        zero_fraction=sum(v.zero_fraction for v in vps) / n,
        magnitude_bin=max(v.magnitude_bin for v in vps),
    )


def _merge_dtypes(specs: Sequence[TensorSpec]) -> str:
    """Widen the dtype when profiling inputs disagree on it.

    ``F(e)`` "keeps the states needed to check whether a mutation operator can be
    applied across those inputs", so a mixed-dtype edge must be annotated with
    the dtype that can represent all observed states.
    """
    dtypes = {s.dtype for s in specs}
    if len(dtypes) == 1:
        return specs[0].dtype
    order = ("complex64", "float64", "float32", "float16", "bfloat16",
             "int64", "int32", "int16", "int8", "uint8", "bool")
    for dtype in order:
        if dtype in dtypes:
            # A float among integers promotes to float, matching the frameworks'
            # own type-promotion rules.
            if dtype in ("int64", "int32", "int16", "int8", "uint8") and \
                    dtypes & {"float64", "float32", "float16", "bfloat16"}:
                continue
            return dtype
    return specs[0].dtype


def aggregate_specs(specs: Sequence[TensorSpec]) -> Optional[TensorSpec]:
    """Collapse per-input tensor states into the representative ``F(e)``.

    Shapes that disagree across profiling inputs make the edge *polymorphic*;
    the aggregate keeps the widest rank and marks the shape as the most common
    observed one, while :meth:`TFG.shapes_of` remains available for exact checks.
    """
    specs = [s for s in specs if s is not None]
    if not specs:
        return None
    if len(specs) == 1:
        return specs[0]
    shapes = {tuple(s.shape) for s in specs}
    base = max(specs, key=lambda s: (s.rank, s.numel))
    return TensorSpec(
        shape=tuple(base.shape),
        dtype=_merge_dtypes(specs),
        layout=base.layout if len({s.layout for s in specs}) > 1 else specs[0].layout,
        device=base.device if len({s.device for s in specs}) > 1 else specs[0].device,
        requires_grad=any(s.requires_grad for s in specs),
        alias_of=next((s.alias_of for s in specs if s.alias_of), None),
        is_view=any(s.is_view for s in specs),
        is_leaf=all(s.is_leaf for s in specs),
        strides=base.strides if len(shapes) == 1 else None,
        value=_merge_values(specs),
        numel=base.numel,
    )


# ---------------------------------------------------------------------------
# TFG
# ---------------------------------------------------------------------------

@dataclass
class TFG:
    """A profiled Tensor Flow Graph."""

    graph: Graph
    #: ``F``: edge id -> tensor state per profiling input.
    edge_specs: Dict[str, List[TensorSpec]] = field(default_factory=dict)
    #: Number of profiling inputs used to build the annotations.
    num_profiles: int = 0
    #: Per-profile execution cost in seconds (profiling inputs).
    profile_times: List[float] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- topology ----------------------------------------------------------
    @property
    def nodes(self) -> Dict[str, Node]:
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges

    def consumers(self, edge_id: str) -> List[str]:
        """``U(e)``."""
        return self.graph.consumers_of(edge_id)

    def data_consumers(self, edge_id: str) -> List[str]:
        return self.graph.data_consumers_of(edge_id)

    def requirements(self, node_id: str) -> List[OpRequirement]:
        return self.graph.nodes[node_id].requirements

    # -- annotations -------------------------------------------------------
    def specs_of(self, edge_id: str) -> List[TensorSpec]:
        return list(self.edge_specs.get(edge_id, []))

    def f(self, edge_id: str) -> Optional[TensorSpec]:
        """The representative edge annotation ``F(e)``."""
        specs = self.specs_of(edge_id)
        if not specs:
            edge = self.graph.edges.get(edge_id)
            if edge is not None and edge.spec is not None:
                return edge.spec
            return None
        return aggregate_specs(specs)

    def shapes_of(self, edge_id: str) -> List[Tuple[int, ...]]:
        return [tuple(s.shape) for s in self.specs_of(edge_id)]

    def is_polymorphic(self, edge_id: str) -> bool:
        return len({tuple(s.shape) for s in self.specs_of(edge_id)}) > 1

    def spec_map(self) -> Dict[str, Optional[TensorSpec]]:
        return {e: self.f(e) for e in self.graph.edges}

    # -- requirement checking ---------------------------------------------
    def context_for(self, node: Node, specs: Optional[Sequence[Optional[TensorSpec]]] = None
                    ) -> ReqContext:
        if specs is None:
            specs = [self.f(e) if e else None for e in node.inputs]
        return ReqContext(node=node, attrs=dict(node.attrs), specs=list(specs),
                          spec_of=self.spec_map(), graph=self.graph)

    def check_node(self, node: Node, kinds: Iterable[str] = ("input", "internal"),
                   specs: Optional[Sequence[Optional[TensorSpec]]] = None,
                   skip: Iterable[str] = ()) -> List[Violation]:
        """Evaluate ``R(v)`` for ``node`` across the requested constraint kinds."""
        kind_set = set(kinds)
        skip_set = set(skip)
        violations: List[Violation] = []
        for req in node.requirements:
            if req.kind not in kind_set or req.predicate in skip_set:
                continue
            ctx = ReqContext(node=node, attrs={**dict(node.attrs), **req.args},
                             specs=list(specs if specs is not None
                                        else [self.f(e) if e else None for e in node.inputs]),
                             spec_of=self.spec_map(), graph=self.graph)
            message = evaluate_predicate(req.predicate, ctx)
            if message:
                anchor = node.inputs[req.args["position"]] if isinstance(
                    req.args.get("position"), int) and req.args["position"] < len(node.inputs) else ""
                violations.append(Violation(kind=req.kind, predicate=req.predicate,
                                            node=node.id, message=message, edge=anchor))
        return violations

    # -- reachability ------------------------------------------------------
    def downstream_of_edges(self, edge_ids: Iterable[str]) -> Set[str]:
        """Operators reachable from ``edge_ids`` (the site's output boundary)."""
        producers = {self.graph.edges[e].producer for e in edge_ids if e in self.graph.edges}
        return self.graph.reachable_from(producers)

    def upstream_of_edges(self, edge_ids: Iterable[str]) -> Set[str]:
        producers = {self.graph.edges[e].producer for e in edge_ids if e in self.graph.edges}
        return self.graph.backward_reachable_from(producers)

    # -- reporting ---------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        s = self.graph.summary()
        s.update({
            "profiled_edges": len(self.edge_specs),
            "num_profiles": self.num_profiles,
            "polymorphic_edges": sum(1 for e in self.graph.edges if self.is_polymorphic(e)),
            "requirements": sum(len(n.requirements) for n in self.graph.nodes.values()),
        })
        return s

    def describe(self, max_nodes: int = 30) -> str:
        lines = [f"TFG({self.graph.name}) nodes={len(self.graph.nodes)} "
                 f"edges={len(self.graph.edges)} profiles={self.num_profiles}"]
        for nid in self.graph.topological_order()[:max_nodes]:
            n = self.graph.nodes[nid]
            outs = []
            for e in self.graph.out_edges(nid):
                spec = self.f(e)
                shape = tuple(spec.shape) if spec else "?"
                dtype = spec.dtype if spec else "?"
                outs.append(f"{shape}:{dtype}")
            lines.append(f"  {nid:>5} {n.op:<20} {' x '.join(outs)}  [{n.scope}]")
        if len(self.graph.nodes) > max_nodes:
            lines.append(f"  ... {len(self.graph.nodes) - max_nodes} more")
        return "\n".join(lines)

    # -- signatures --------------------------------------------------------
    def data_flow_signature(self, producer_edge: str, consumer_node: str) -> Tuple[Any, ...]:
        """The DFSD signature of one producer-consumer data flow.

        Records the producer and consumer operator types, the kernel/operator
        path, the binned value profile and the tensor properties (shape, rank,
        dtype, layout, device, aliasing, gradient state).
        """
        edge = self.graph.edges.get(producer_edge)
        if edge is None:
            return ()
        spec = self.f(producer_edge)
        producer_node = self.graph.nodes.get(edge.producer)
        consumer = self.graph.nodes.get(consumer_node)
        pos = next((p for (c, p) in edge.consumers if c == consumer_node), -1)
        return (
            producer_node.op if producer_node else "?",
            consumer.op if consumer else "?",
            f"{producer_node.scope}->{consumer.scope}" if producer_node and consumer else "",
            () if spec is None else spec.signature_tuple(),
            pos,
        )

    def all_signatures(self) -> Set[Tuple[Any, ...]]:
        sigs: Set[Tuple[Any, ...]] = set()
        for eid, edge in self.graph.edges.items():
            if self.graph.nodes[edge.producer].is_param:
                continue
            for (consumer, _pos) in edge.consumers:
                if self.graph.nodes[consumer].is_param:
                    continue
                sigs.add(self.data_flow_signature(eid, consumer))
        return sigs


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def build_tfg(graph: Graph, edge_specs: Dict[str, List[TensorSpec]],
              profile_times: Optional[Sequence[float]] = None,
              meta: Optional[Dict[str, Any]] = None) -> TFG:
    """``BuildTFG(E)``: attach the profiled tensor states to the graph."""
    from flowmut.ir.ops import attach_requirements
    attach_requirements(graph)
    specs: Dict[str, List[TensorSpec]] = {}
    for eid in graph.edges:
        states = edge_specs.get(eid)
        if states:
            specs[eid] = list(states)
        else:
            edge = graph.edges[eid]
            if edge.spec is not None:
                specs[eid] = [edge.spec]
    n_profiles = max((len(v) for v in specs.values()), default=0)
    return TFG(graph=graph, edge_specs=specs, num_profiles=n_profiles,
               profile_times=list(profile_times or []), meta=dict(meta or {}))
