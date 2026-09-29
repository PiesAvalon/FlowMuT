"""The framework-neutral model graph (IR) that FlowMuT profiles and mutates.

A :class:`Graph` is a DAG of :class:`Node` objects connected by :class:`Edge`
objects.  It is the single source of truth for three things:

* the operator topology ``G`` of the TFG (``G = (V, E)`` with ``V`` the nodes
  and ``E`` the edges),
* the bounded subgraphs that mutation operators match and replace,
* the program that framework adapters materialise into a runnable module.

Parameters are ordinary nodes with ``op == "param"``; this keeps weight tensors
inside the same data-flow representation so that tensor-level mutation
operators can target them like any other edge.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from flowmut.ir.specs import (
    OpRequirement,
    TensorSpec,
    shape_numel,
)

# ---------------------------------------------------------------------------
# Nodes and edges
# ---------------------------------------------------------------------------

#: Node ops that introduce graph inputs (no predecessor edges).
INPUT_OPS: Set[str] = {"input"}

#: Node ops that are parameters rather than computations.
PARAM_OPS: Set[str] = {"param"}

#: Node ops that terminate the graph.
OUTPUT_OPS: Set[str] = {"output"}


@dataclass
class Node:
    """One operator instance in the IR."""

    id: str
    op: str
    inputs: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    #: Dotted module scope, e.g. ``"stages.1.blocks.0.conv1"``.  Mutations and
    #: the candidate site-context feature use it to describe *where* a change
    #: happens without referring to tensor values.
    scope: str = ""
    name: str = ""
    #: Number of tensors produced.  ``> 1`` for ``split``/``topk``/``chunk``.
    num_outputs: int = 1
    #: Static, declared output shape when it can be inferred at build time.
    out_shape: Optional[Tuple[int, ...]] = None
    #: Requirements ``R(v)`` attached by the op registry.
    requirements: List[OpRequirement] = field(default_factory=list)
    #: Free-form provenance (which seed model / mutation created the node).
    origin: str = "seed"

    def clone(self, new_id: Optional[str] = None) -> "Node":
        n = copy.deepcopy(self)
        if new_id is not None:
            n.id = new_id
        return n

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op,
            "inputs": list(self.inputs),
            "attrs": _jsonable(self.attrs),
            "scope": self.scope,
            "name": self.name,
            "num_outputs": self.num_outputs,
            "out_shape": None if self.out_shape is None else list(self.out_shape),
            "origin": self.origin,
        }

    @property
    def display(self) -> str:
        return self.name or self.op

    @property
    def is_param(self) -> bool:
        return self.op in PARAM_OPS

    @property
    def is_input(self) -> bool:
        return self.op in INPUT_OPS


@dataclass
class Edge:
    """A tensor flowing from one producer output port to one or more consumers."""

    id: str
    producer: str
    index: int = 0
    #: ``(consumer_node_id, input_position)`` pairs, ordered.
    consumers: List[Tuple[str, int]] = field(default_factory=list)
    #: Declared spec (may be enriched by profiling).
    spec: Optional[TensorSpec] = None
    name: str = ""

    def clone(self, new_id: Optional[str] = None) -> "Edge":
        e = copy.deepcopy(self)
        if new_id is not None:
            e.id = new_id
        return e

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "producer": self.producer,
            "index": self.index,
            "consumers": [list(c) for c in self.consumers],
            "spec": None if self.spec is None else self.spec.to_dict(),
            "name": self.name,
        }

    @property
    def consumer_nodes(self) -> List[str]:
        return [c for c, _ in self.consumers]

    @property
    def is_boundary(self) -> bool:
        """An edge whose producer and consumers are all parametric data flow."""
        return True


def _jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class Graph:
    """A mutable, copy-on-write DAG of tensor operators."""

    def __init__(self, name: str = "model", meta: Optional[Dict[str, Any]] = None):
        self.name = name
        self.nodes: "OrderedDict[str, Node]" = OrderedDict()
        self.edges: "OrderedDict[str, Edge]" = OrderedDict()
        #: Graph input edge ids (in call order).
        self.inputs: List[str] = []
        #: Graph output edge ids.
        self.outputs: List[str] = []
        self.meta: Dict[str, Any] = dict(meta or {})
        self._counter = 0

    # -- identifiers -------------------------------------------------------
    def new_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}{self._counter}"

    # -- construction ------------------------------------------------------
    def add_node(self, op: str, inputs: Sequence[str], **kwargs: Any) -> Node:
        node = Node(id=self.new_id("n"), op=op, inputs=list(inputs), **kwargs)
        self.nodes[node.id] = node
        # Produce output edges and wire consumers.
        for i in range(max(1, node.num_outputs)):
            eid = self.new_id("e")
            edge = Edge(id=eid, producer=node.id, index=i, name=f"{node.display}:{i}")
            self.edges[eid] = edge
        for pos, edge_id in enumerate(node.inputs):
            if pos < 0:
                continue
            if edge_id not in self.edges:
                raise KeyError(f"node {node.id} ({op}) references unknown edge {edge_id}")
            self.edges[edge_id].consumers.append((node.id, pos))
        return node

    def _out_edge_ids(self, node: Node) -> List[str]:
        return [e for e in self.edges if self.edges[e].producer == node.id]

    def add_input(self, name: str, spec: TensorSpec, scope: str = "inputs") -> Edge:
        node = self.add_node("input", [], name=name, scope=scope, out_shape=tuple(spec.shape))
        edge = self.edges[self._out_edge_ids(node)[0]]
        edge.spec = spec
        edge.name = name
        self.inputs.append(edge.id)
        return edge

    def add_param(self, name: str, spec: TensorSpec, init: str = "kaiming_uniform",
                  extra: Optional[Dict[str, Any]] = None, scope: str = "params") -> Edge:
        attrs: Dict[str, Any] = {"init": init, "name": name}
        if extra:
            attrs.update(extra)
        node = self.add_node("param", [], name=name, scope=scope, attrs=attrs,
                             out_shape=tuple(spec.shape))
        edge = self.edges[self._out_edge_ids(node)[0]]
        edge.spec = spec
        edge.name = name
        return edge

    def set_outputs(self, edges: Sequence[str], names: Optional[Sequence[str]] = None) -> None:
        self.outputs = list(edges)
        for i, e in enumerate(self.outputs):
            if e not in self.edges:
                raise KeyError(e)
            if names is not None and i < len(names):
                self.edges[e].name = names[i]

    # -- queries -----------------------------------------------------------
    def out_edges(self, node_id: str) -> List[str]:
        return [e.id for e in self.edges.values() if e.producer == node_id]

    def in_edges(self, node_id: str) -> List[str]:
        return list(self.nodes[node_id].inputs)

    def consumers_of(self, edge_id: str) -> List[str]:
        """``U(e)`` -- the direct consumers of edge ``e``."""
        return [c for c, _ in self.edges[edge_id].consumers]

    def data_consumers_of(self, edge_id: str) -> List[str]:
        """Consumers of ``e`` ignoring parameter edges (used by mutations)."""
        return [c for c in self.consumers_of(edge_id) if not self.nodes[c].is_param]

    def producer_of(self, edge_id: str) -> Node:
        return self.nodes[self.edges[edge_id].producer]

    def node_by_scope(self, scope: str) -> Optional[Node]:
        for n in self.nodes.values():
            if n.scope == scope:
                return n
        return None

    def topological_order(self) -> List[str]:
        indeg: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for nid, node in self.nodes.items():
            for e in node.inputs:
                if e in self.edges:
                    indeg[nid] += 1
        ready = [nid for nid, d in indeg.items() if d == 0]
        order: List[str] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for e in self.out_edges(nid):
                for c in self.consumers_of(e):
                    indeg[c] -= 1
                    if indeg[c] == 0:
                        ready.append(c)
        if len(order) != len(self.nodes):
            raise ValueError("graph contains a cycle")
        return order

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes.values())

    def data_edges(self) -> List[str]:
        """Edges that carry activations (i.e. exclude parameter edges)."""
        return [e.id for e in self.edges.values() if not self.nodes[e.producer].is_param]

    def param_edges(self) -> List[str]:
        return [e.id for e in self.edges.values() if self.nodes[e.producer].is_param]

    # -- reachability (used by the downstream-impact feature P(c)) ---------
    def reachable_from(self, node_ids: Iterable[str]) -> Set[str]:
        seen: Set[str] = set()
        stack = list(node_ids)
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            for e in self.out_edges(nid):
                for c in self.consumers_of(e):
                    if c not in seen:
                        stack.append(c)
        return seen

    def subgraph_between(self, start_nodes: Iterable[str], end_nodes: Iterable[str]) -> Set[str]:
        fwd = self.reachable_from(start_nodes)
        bwd = self.backward_reachable_from(end_nodes)
        return fwd & bwd

    def backward_reachable_from(self, node_ids: Iterable[str]) -> Set[str]:
        seen: Set[str] = set()
        stack = list(node_ids)
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            for e in self.in_edges(nid):
                if e in self.edges:
                    stack.append(self.edges[e].producer)
        return seen

    # -- copying and rewriting --------------------------------------------
    def copy(self, name: Optional[str] = None) -> "Graph":
        g = Graph(name or self.name, copy.deepcopy(self.meta))
        g.nodes = OrderedDict((k, v.clone()) for k, v in self.nodes.items())
        g.edges = OrderedDict((k, v.clone()) for k, v in self.edges.items())
        g.inputs = list(self.inputs)
        g.outputs = list(self.outputs)
        g._counter = self._counter
        return g

    def remove_node(self, node_id: str) -> None:
        """Delete a node and its outgoing edges, leaving consumers dangling."""
        node = self.nodes.pop(node_id)
        for e in self.out_edges(node_id):
            self.edges.pop(e, None)
        for e in node.inputs:
            if e in self.edges:
                self.edges[e].consumers = [
                    (c, p) for (c, p) in self.edges[e].consumers if c != node_id
                ]

    def disconnect_input(self, node_id: str, position: int) -> None:
        node = self.nodes[node_id]
        if position >= len(node.inputs):
            return
        eid = node.inputs[position]
        if eid in self.edges:
            self.edges[eid].consumers = [
                (c, p) for (c, p) in self.edges[eid].consumers if not (c == node_id and p == position)
            ]
        node.inputs[position] = ""

    def connect_input(self, node_id: str, position: int, edge_id: str) -> None:
        node = self.nodes[node_id]
        while len(node.inputs) <= position:
            node.inputs.append("")
        node.inputs[position] = edge_id
        self.edges[edge_id].consumers.append((node_id, position))

    def bypass(self, node_id: str) -> str:
        """Remove ``node_id`` and reconnect its consumer(s) to its first input.

        Used by deletion operators.  Returns the replacement edge id.
        """
        node = self.nodes[node_id]
        src = node.inputs[0]
        if node.num_outputs == 1:
            out_edge = self.out_edges(node_id)[0]
            for (consumer, pos) in list(self.edges[out_edge].consumers):
                self.disconnect_input(consumer, pos)
                self.connect_input(consumer, pos, src)
        self.remove_node(node_id)
        return src

    def replace_subgraph(
        self,
        remove_nodes: Sequence[str],
        new_nodes: Sequence[Node],
        new_edges: Sequence[Edge],
        boundary_map: Dict[str, str],
    ) -> None:
        """Replace ``remove_nodes`` with ``new_nodes``/``new_edges``.

        ``boundary_map`` maps an old *external* edge id to the new edge id that
        should take its place for every consumer outside the removed set.  This
        is the low-level primitive behind "constrained subgraph replacement".
        """
        remove_set = set(remove_nodes)
        external: List[Tuple[str, int, str]] = []
        for nid in remove_set:
            for pos, eid in enumerate(self.nodes[nid].inputs):
                if eid and eid in self.edges and self.edges[eid].producer not in remove_set:
                    for (c, p) in self.edges[eid].consumers:
                        if c not in remove_set:
                            external.append((eid, p, c))
        for nid in remove_set:
            outs = self.out_edges(nid)
            for e in outs:
                if e in boundary_map:
                    for (c, p) in list(self.edges[e].consumers):
                        if c not in remove_set:
                            external.append((e, p, c))
            self.remove_node(nid)
        for node in new_nodes:
            node.inputs = [boundary_map.get(e, e) if e else e for e in node.inputs]
            self.nodes[node.id] = node
            for i in range(max(1, node.num_outputs)):
                eid = node.id + (f":o{i}" if node.num_outputs > 1 else ":o")
                self.edges[eid] = Edge(id=eid, producer=node.id, index=i,
                                       name=f"{node.display}:{i}")
        for edge in new_edges:
            self.edges[edge.id] = edge
        for node in new_nodes:
            for pos, eid in enumerate(node.inputs):
                if eid and eid in self.edges:
                    self.edges[eid].consumers.append((node.id, pos))
        for (old_edge, pos, consumer) in external:
            target = boundary_map.get(old_edge)
            if target is None or consumer in remove_set:
                continue
            self.connect_input(consumer, pos, target)

    # -- validation --------------------------------------------------------
    def check(self) -> List[str]:
        """Structural validation; returns a list of human-readable problems."""
        problems: List[str] = []
        for nid, node in self.nodes.items():
            for pos, e in enumerate(node.inputs):
                if not e:
                    problems.append(f"node {nid} ({node.op}) has an unconnected input {pos}")
                elif e not in self.edges:
                    problems.append(f"node {nid} ({node.op}) input {pos} -> missing edge {e}")
                else:
                    if (nid, pos) not in self.edges[e].consumers:
                        problems.append(f"edge {e} missing consumer record ({nid},{pos})")
        for eid, edge in self.edges.items():
            if edge.producer not in self.nodes:
                problems.append(f"edge {eid} has missing producer {edge.producer}")
            for (c, p) in edge.consumers:
                if c not in self.nodes:
                    problems.append(f"edge {eid} has missing consumer {c}")
                elif p >= len(self.nodes[c].inputs) or self.nodes[c].inputs[p] != eid:
                    problems.append(f"edge {eid} consumer record ({c},{p}) disagrees with node inputs")
        for e in self.outputs:
            if e not in self.edges:
                problems.append(f"graph output {e} is not an edge")
        for e in self.inputs:
            if e not in self.edges:
                problems.append(f"graph input {e} is not an edge")
        try:
            self.topological_order()
        except ValueError as exc:
            problems.append(str(exc))
        return problems

    # -- reporting ---------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        ops: Dict[str, int] = {}
        for n in self.nodes.values():
            ops[n.op] = ops.get(n.op, 0) + 1
        return {
            "name": self.name,
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "params": len(self.param_edges()),
            "data_edges": len(self.data_edges()),
            "op_histogram": dict(sorted(ops.items(), key=lambda kv: -kv[1])),
            "inputs": len(self.inputs),
            "outputs": len(self.outputs),
        }

    def describe(self, max_nodes: int = 40) -> str:
        lines = [f"Graph({self.name}) nodes={len(self.nodes)} edges={len(self.edges)}"]
        order = self.topological_order()
        for nid in order[:max_nodes]:
            n = self.nodes[nid]
            outs = ",".join(f"{self.edges[e].name}{self.edges[e].spec.shape if self.edges[e].spec else ''}"
                            for e in self.out_edges(nid))
            lines.append(f"  {nid:>5} {n.scope or '-':<40} {n.op:<18} -> {outs}")
        if len(order) > max_nodes:
            lines.append(f"  ... {len(order) - max_nodes} more nodes")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder DSL used by the seed models and by mutation replacements
# ---------------------------------------------------------------------------

class GraphBuilder:
    """Ergonomic façade over :class:`Graph` used to write seed models.

    Every method returns the id of the edge carrying its result, so model
    definitions read like ordinary framework code while emitting explicit,
    mutable IR nodes.
    """

    def __init__(self, name: str, meta: Optional[Dict[str, Any]] = None):
        self.graph = Graph(name, meta)
        self._scope_stack: List[str] = []
        self._param_index: Dict[str, int] = {}

    # -- scopes ------------------------------------------------------------
    def scope(self, name: str) -> "GraphBuilder":
        self._scope_stack.append(name)
        return self

    def _scope(self, local: str = "") -> str:
        parts = [p for p in self._scope_stack if p]
        if local:
            parts.append(local)
        return ".".join(parts)

    class _ScopeCtx:
        def __init__(self, builder: "GraphBuilder", name: str):
            self.builder = builder
            self.name = name

        def __enter__(self):
            self.builder.scope(self.name)
            return self.builder

        def __exit__(self, *exc):
            self.builder._scope_stack.pop()
            return False

    def block(self, name: str) -> "GraphBuilder._ScopeCtx":
        """Context manager that pushes a scope, e.g. ``with b.block("stage0"):``."""
        return GraphBuilder._ScopeCtx(self, name)

    # -- primitives --------------------------------------------------------
    def _emit(self, op: str, inputs: Sequence[str], **attrs: Any) -> str:
        local = attrs.pop("_scope", "")
        shape = attrs.pop("_shape", None)
        num_outputs = attrs.pop("_num_outputs", 1)
        name = attrs.pop("_name", None) or op
        node = self.graph.add_node(op, list(inputs), attrs=attrs, scope=self._scope(local),
                                   name=name, num_outputs=num_outputs, out_shape=shape)
        from flowmut.ir.ops import requirements_for
        node.requirements = requirements_for(node)
        outs = self.graph.out_edges(node.id)
        self._propagate_spec(node, outs)
        if len(outs) == 1:
            return outs[0]
        return outs  # type: ignore[return-value]

    def _propagate_spec(self, node: Node, out_edges: Sequence[str]) -> None:
        """Give every produced edge a declared :class:`TensorSpec`.

        Seed models are written in chained style, so a layer must be able to ask
        its input's shape (channel counts, feature dimensions).  Shapes are
        inferred statically here and later replaced by the profiled ``F(e)``.
        """
        from flowmut.ir.ops import infer_shape
        in_specs = [self.graph.edges[e].spec for e in node.inputs
                    if e in self.graph.edges and self.graph.edges[e].spec is not None]
        shape = node.out_shape
        if shape is None:
            shape = infer_shape(node.op, in_specs, node.attrs)
        if shape is not None:
            node.out_shape = tuple(shape)
        dtype = node.attrs.get("dtype")
        if node.op in ("argmax", "argmin"):
            dtype = "int64"
        elif dtype is None:
            dtype = in_specs[0].dtype if in_specs else "float32"
        base = in_specs[0] if in_specs else None
        spec = TensorSpec(
            shape=tuple(shape or ()),
            dtype=str(dtype),
            layout=base.layout if base else "contiguous",
            device=base.device if base else "cpu",
            requires_grad=bool(base.requires_grad) if base else False,
            alias_of=base.alias_of if base and node.op in ("clone",) else None,
            is_view=node.op in ("view", "permute", "transpose", "squeeze",
                                "unsqueeze", "expand", "broadcast_to", "getitem"),
            is_leaf=node.op in PARAM_OPS,
            strides=None,
            value=None,
            numel=shape_numel(shape) if shape else 0,
        )
        for i, eid in enumerate(out_edges):
            if i == 0:
                self.graph.edges[eid].spec = spec
            else:
                self.graph.edges[eid].spec = spec.with_(shape=spec.shape)

    def input(self, name: str, shape: Sequence[int], dtype: str = "float32",
              device: str = "cpu", layout: str = "contiguous",
              requires_grad: bool = False) -> str:
        spec = TensorSpec(shape=tuple(shape), dtype=dtype, device=device, layout=layout,
                          requires_grad=requires_grad, numel=shape_numel(shape))
        return self.graph.add_input(name, spec).id

    def param(self, shape: Sequence[int], init: str = "kaiming_uniform",
              dtype: str = "float32", name: Optional[str] = None,
              **extra: Any) -> str:
        base = name or f"p{len(self._param_index)}"
        self._param_index[base] = self._param_index.get(base, 0) + 1
        full = base if self._param_index[base] == 1 else f"{base}_{self._param_index[base]}"
        spec = TensorSpec(shape=tuple(shape), dtype=dtype, requires_grad=True,
                          numel=shape_numel(shape))
        return self.graph.add_param(full, spec, init=init, extra=extra).id

    # ---- convenience wrappers for the common ops -------------------------
    def __getattr__(self, item: str):
        """Fallback: ``b.any_op_name(...)`` emits a node for ``any_op_name``.

        This keeps the builder open for the full op registry without a wrapper
        method per op; the wrappers above exist only for the hot paths.
        """
        if item.startswith("__") or item.startswith("_"):
            raise AttributeError(item)
        from flowmut.ir.ops import OP_REGISTRY, canonical_op
        op = canonical_op(item)
        if op not in OP_REGISTRY:
            raise AttributeError(f"unknown IR op {item!r}")

        def emit(*edges, **attrs):
            flat: List[str] = []
            for e in edges:
                if isinstance(e, (list, tuple)):
                    flat.extend(e)
                else:
                    flat.append(e)
            return self._emit(op, flat, **attrs)

        return emit

    # -- composites --------------------------------------------------------
    def residual(self, x: str, fn: Callable[[str], str], scale: float = 1.0,
                 scope: str = "residual") -> str:
        y = fn(x)
        if scale != 1.0:
            y = self._emit("mul", [y], other=scale, _scope=scope)
        return self._emit("add", [x, y], _scope=scope)

    def conv_bn_act(self, x: str, out_channels: int, kernel_size: int = 3, stride: int = 1,
                    padding: Optional[int] = None, groups: int = 1, act: str = "relu",
                    norm: str = "batchnorm", scope: str = "conv_bn_act", bias: bool = False) -> str:
        if padding is None:
            padding = kernel_size // 2
        return self.conv_block(x, out_channels, kernel_size, stride, padding, groups,
                               act, norm, scope, bias)

    def conv_block(self, x: str, out_channels: int, kernel_size: int = 3, stride: int = 1,
                   padding: Optional[int] = None, groups: int = 1, act: str = "relu",
                   norm: str = "batchnorm", scope: str = "conv", bias: bool = False,
                   dilation: int = 1) -> str:
        if padding is None:
            padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
        in_channels = self._channels_of(x)
        w = self.param((out_channels, max(1, in_channels // groups), kernel_size, kernel_size),
                       init="kaiming_uniform", name=f"{scope}.weight")
        ins = [x, w]
        if bias:
            b = self.param((out_channels,), init="zeros", name=f"{scope}.bias")
            ins.append(b)
        y = self._emit("conv2d", ins, stride=stride, padding=padding, groups=groups,
                       dilation=dilation, bias=bias, _scope=scope)
        if norm:
            y = self._emit(norm, [y], num_features=out_channels, _scope=f"{scope}.norm")
        if act:
            y = self._emit(act, [y], _scope=f"{scope}.act")
        return y

    def linear(self, x: str, out_features: int, bias: bool = True, act: str = "",
               scope: str = "fc") -> str:
        in_features = self._last_dim(x)
        w = self.param((out_features, in_features), init="kaiming_uniform",
                       name=f"{scope}.weight")
        ins = [x, w]
        if bias:
            b = self.param((out_features,), init="zeros", name=f"{scope}.bias")
            ins.append(b)
        y = self._emit("linear", ins, bias=bias, _scope=scope)
        if act:
            y = self._emit(act, [y], _scope=f"{scope}.act")
        return y

    def _spec_of(self, edge_id: str) -> Optional[TensorSpec]:
        return self.graph.edges[edge_id].spec

    def _channels_of(self, edge_id: str) -> int:
        spec = self._spec_of(edge_id)
        if spec is None or spec.rank < 2:
            return 1
        return int(spec.shape[1])

    def _last_dim(self, edge_id: str) -> int:
        spec = self._spec_of(edge_id)
        if spec is None or not spec.shape:
            return 1
        return int(spec.shape[-1])

    # -- finalisation ------------------------------------------------------
    def output(self, edges, names=None) -> None:
        if isinstance(edges, str):
            edges = [edges]
        if names is None and hasattr(self, "_output_names"):
            names = self._output_names
        self.graph.set_outputs(list(edges), names)

    def build(self) -> Graph:
        return self.graph
