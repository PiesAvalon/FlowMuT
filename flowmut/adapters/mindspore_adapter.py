"""MindSpore adapter (PyNative and Graph modes).

The adapter turns a framework-neutral :class:`~flowmut.ir.Graph` into a runnable
MindSpore program, profiles the tensor on every edge and reports the framework's
control flow as a normalised trace.

Two modes are provided:

``MindSporePyNativeAdapter``
    the graph is interpreted node by node with plain Python
    (:func:`flowmut.adapters.ms_kernels.apply_pynative`).  Because the loop is
    ordinary Python, *every* produced tensor can be tapped, which is what makes
    the ``F(e)`` annotation complete.
``MindSporeGraphAdapter``
    a real Python module is generated from the graph and imported, so that
    MindSpore's parser compiles one ``nn.Cell.construct`` per graph.  The trace
    records the graph-capture / optimisation / reuse decisions, and the per-node
    dispatcher surface of the compiled graph.

``F(e)`` and graph mode
-----------------------
A compiled MindSpore graph deliberately does not expose intermediate tensors, so
:meth:`MindSporeGraphAdapter.execute` collects the edge annotations from a
**PyNative pass** executed with ``ms.set_context(mode=PYNATIVE_MODE)``.  That
pass reuses the *same parameter objects* as the graph-mode cell (the ``p_<node>``
``ms.Parameter`` attributes created by the generated ``__init__``), so the
annotation describes exactly the tensors the compiled program consumes and
produces.  The output values reported by ``execute`` always come from the
graph-mode execution; only the annotations come from the PyNative twin.

The module is import-safe on a PyTorch-only interpreter: ``import mindspore`` is
guarded, and :meth:`MindSporeAdapter.is_available` reports the failure instead of
raising.
"""

from __future__ import annotations

import hashlib
import importlib.util
import keyword
import os
import re
import sys
import tempfile
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

try:  # pragma: no cover - depends on the interpreter
    import mindspore as ms
    from mindspore import Tensor as MSTensor
    from mindspore import nn as ms_nn

    _MS_IMPORT_ERROR = ""
    MS_VERSION = str(ms.__version__)
except Exception as exc:  # pragma: no cover - the PyTorch-only interpreter
    ms = None  # type: ignore[assignment]
    MSTensor = None  # type: ignore[assignment]
    ms_nn = None  # type: ignore[assignment]
    _MS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    MS_VERSION = ""

from flowmut.adapters.base import (
    ExecutionResult,
    FrameworkAdapter,
    Sample,
    TimeoutExceeded,
    TimeoutGuard,
    TraceEvent,
    graph_fingerprint,
    node_signature,
    value_profile,
)
from flowmut.ir.graph import Graph, Node
from flowmut.ir.ops import canonical_op
from flowmut.ir.specs import TensorSpec
from flowmut.trace.normalize import TraceRecorder


# ---------------------------------------------------------------------------
# lazily imported kernel table (ms_kernels imports mindspore at module level)
# ---------------------------------------------------------------------------

def _kernels():
    from flowmut.adapters import ms_kernels
    return ms_kernels


def _kernel_missing():
    try:
        return _kernels().KernelMissing
    except Exception:  # pragma: no cover - defensive
        class _Missing(NotImplementedError):
            pass
        return _Missing


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

class MSEnv:
    """Process-wide ``ms.set_context`` helper (device ``cpu``).

    MindSpore's execution mode is process-global, so the adapter switches it
    explicitly before every execution (graph mode needs ``GRAPH_MODE``, the
    ``F(e)`` annotation pass needs ``PYNATIVE_MODE``).
    """

    _configured = False
    _device = ""
    _mode = ""

    @classmethod
    def ensure(cls, device: str = "cpu", mode: Optional[str] = None) -> bool:
        if ms is None:
            return False
        if not cls._configured:
            setter = getattr(ms, "set_device", None)
            try:
                if callable(setter):
                    setter("CPU")
                else:  # pragma: no cover - older releases
                    ms.set_context(device_target="CPU")
            except Exception:
                try:
                    ms.set_context(device_target="CPU")
                except Exception:
                    pass
            cls._configured = True
            cls._device = "cpu"
        if mode is not None and cls._mode != mode:
            ms.set_context(mode=ms.GRAPH_MODE if mode == "graph" else ms.PYNATIVE_MODE)
            cls._mode = mode
        return True

    @classmethod
    @contextmanager
    def use_mode(cls, mode: str) -> Iterator[None]:
        previous = cls._mode
        cls.ensure(mode=mode)
        try:
            yield
        finally:
            if previous and previous != mode:
                try:
                    cls.ensure(mode=previous)
                except Exception:  # pragma: no cover - defensive
                    pass


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

_NUMPY_DTYPES: Dict[str, Any] = {
    "float64": np.float64,
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": np.float32,
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "int8": np.int8,
    "uint8": np.uint8,
    "bool": np.bool_,
    "complex64": np.complex64,
}


def _numpy_dtype(name: str):
    return _NUMPY_DTYPES.get(str(name), np.float32)


def _ms_dtype(name: str):
    return _kernels().ms_dtype(name)


def _requires_grad(tensor: Any) -> bool:
    try:
        return bool(getattr(tensor, "requires_grad", False))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# parameter bank
# ---------------------------------------------------------------------------

class ParamBank:
    """Deterministically initialised tensors, reused across mutants.

    Parameters are keyed by :func:`~flowmut.adapters.base.node_signature`, which
    is invariant under graph rewriting, so a mutation that only changes, say, an
    activation keeps the surrounding convolution weights.  The tensors are
    created from numpy with the *same* fan-in/fan-out formulas as the PyTorch
    bank, hence both frameworks see parameters of the same scale.
    """

    def __init__(self, seed: int = 20240929):
        self.seed = int(seed)
        self.tensors: Dict[str, Any] = {}

    def key_for(self, node: Node) -> str:
        return node_signature(node)

    def tensor_for(self, node: Node, graph: Graph) -> Any:
        key = self.key_for(node)
        cached = self.tensors.get(key)
        if cached is not None:
            return cached
        shape = tuple(node.out_shape or _declared_shape(node, graph) or ())
        dtype = str(node.attrs.get("param_dtype", "float32"))
        array = np.zeros(shape, dtype=_numpy_dtype(dtype))
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        seed = (int(digest, 16) ^ self.seed) % (2 ** 31)
        generator = np.random.Generator(np.random.PCG64(seed))
        _kernels().init_parameter(array, str(node.attrs.get("init", "kaiming_uniform")),
                                  generator)
        tensor = ms.Tensor.from_numpy(np.ascontiguousarray(array))
        self.tensors[key] = tensor
        return tensor

    @property
    def numel(self) -> int:
        total = 0
        for tensor in self.tensors.values():
            try:
                total += int(tensor.size)
            except Exception:
                try:
                    total += int(np.prod(tuple(tensor.shape)))
                except Exception:
                    continue
        return total


def _declared_shape(node: Node, graph: Graph) -> Optional[Tuple[int, ...]]:
    for eid in graph.out_edges(node.id):
        spec = graph.edges[eid].spec
        if spec is not None and spec.shape:
            return tuple(int(d) for d in spec.shape)
    return None


# ---------------------------------------------------------------------------
# node attributes handed to the kernel table
# ---------------------------------------------------------------------------

def _attrs_for_node(node: Node, graph: Graph) -> Dict[str, Any]:
    """``node.attrs`` plus the shape context some MindSpore kernels need."""
    attrs: Dict[str, Any] = dict(node.attrs)
    attrs.setdefault("num_outputs", int(node.num_outputs or 1))
    attrs["_arity"] = len(node.inputs)
    shapes: List[List[int]] = []
    for eid in node.inputs:
        edge = graph.edges.get(eid)
        spec = edge.spec if edge is not None else None
        shapes.append([int(d) for d in spec.shape] if spec is not None else [])
    attrs["_input_shapes"] = shapes
    attrs["_out_shape"] = [int(d) for d in (node.out_shape or ())]
    return attrs


# ---------------------------------------------------------------------------
# PyNative interpreter
# ---------------------------------------------------------------------------

class PyNativeInterpreter:
    """Executes an IR graph node by node in Python (MindSpore PyNative mode)."""

    def __init__(self, graph: Graph, params: Dict[str, Any],
                 fingerprint: str = "") -> None:
        self.graph_name = graph.name
        self.fingerprint = fingerprint
        self._nodes: Dict[str, Node] = dict(graph.nodes)
        self._order: List[str] = graph.topological_order()
        self._input_edges: List[str] = list(graph.inputs)
        self._output_edges: List[str] = list(graph.outputs)
        self._out_edges: Dict[str, List[str]] = {
            nid: graph.out_edges(nid) for nid in self._nodes
        }
        self._attrs: Dict[str, Dict[str, Any]] = {
            nid: _attrs_for_node(node, graph) for nid, node in self._nodes.items()
        }
        self.params: Dict[str, Any] = dict(params)
        #: Edges produced by ``param`` nodes; consuming them is a differentiable
        #: (autograd-tracked) operation even when the bank tensor itself is a
        #: plain ``ms.Tensor``.
        self.param_edges: Set[str] = set()
        for nid, node in self._nodes.items():
            if node.op == "param":
                self.param_edges.update(self._out_edges[nid])
        self.tap_enabled: bool = False
        self.taps: Dict[str, Any] = {}
        self.alias_log: List[str] = []
        self.node_ops: List[Tuple[str, str]] = []

    @property
    def order(self) -> List[str]:
        return list(self._order)

    def ops(self) -> List[str]:
        return [canonical_op(self._nodes[nid].op) for nid in self._order
                if self._nodes[nid].op not in ("input", "param", "output")]

    def run(self, inputs: Sequence[Any], recorder: Optional[TraceRecorder] = None
            ) -> Tuple[Any, ...]:
        vals: Dict[str, Any] = {}
        if len(inputs) != len(self._input_edges):
            raise ValueError(f"expected {len(self._input_edges)} inputs, "
                             f"got {len(inputs)}")
        for edge_id, tensor in zip(self._input_edges, inputs):
            vals[edge_id] = tensor
        taps = self.taps
        if self.tap_enabled:
            taps.clear()
            for edge_id, tensor in zip(self._input_edges, inputs):
                taps[edge_id] = tensor

        alias_log: List[str] = []
        node_ops: List[Tuple[str, str]] = []
        for nid in self._order:
            node = self._nodes[nid]
            out_edges = self._out_edges[nid]
            if node.op == "input":
                continue
            if node.op == "param":
                tensor = self.params.get(nid)
                if tensor is None:
                    raise RuntimeError(f"parameter node {nid} has no tensor")
                for e in out_edges:
                    vals[e] = tensor
                    if self.tap_enabled:
                        taps[e] = tensor
                continue
            if node.op == "output":
                source = vals[node.inputs[0]]
                for e in out_edges:
                    vals[e] = source
                    if self.tap_enabled:
                        taps[e] = source
                continue
            tensors: List[Any] = []
            for position, e in enumerate(node.inputs):
                if e not in vals:
                    raise RuntimeError(
                        f"node {nid} ({node.op}) input {position} edge {e} "
                        f"was never produced")
                tensors.append(vals[e])
            result = _kernels().apply_pynative(node.op, self._attrs[nid], tensors)
            produced = list(result) if isinstance(result, (tuple, list)) else [result]
            if len(produced) != len(out_edges):
                raise RuntimeError(
                    f"node {nid} ({node.op}) produced {len(produced)} tensors "
                    f"but the graph declares {len(out_edges)} outputs")
            aliased = any(p is t for p in produced for t in tensors)
            for e, tensor in zip(out_edges, produced):
                vals[e] = tensor
                if self.tap_enabled:
                    taps[e] = tensor
            op = canonical_op(node.op)
            node_ops.append((nid, op))
            if aliased:
                alias_log.append(nid)
            if recorder is not None:
                recorder.record(f"ops.{op}", "dispatcher")
                differentiable = any(_requires_grad(t) for t in tensors) or \
                    any(e in self.param_edges for e in node.inputs)
                if differentiable:
                    recorder.record("autograd.forward_node", "autograd")
                recorder.record("memory.alias_shared_storage" if aliased
                                else "memory.fresh_storage", "memory")

        self.node_ops = node_ops
        self.alias_log = alias_log
        if not self._output_edges:
            if not vals:
                return ()
            return (list(vals.values())[-1],)
        return tuple(vals[e] for e in self._output_edges)


# ---------------------------------------------------------------------------
# graph-mode source generation
# ---------------------------------------------------------------------------

class _SourceGenError(RuntimeError):
    pass


class _NamePool:
    """Deterministic, collision-free Python identifier allocation."""

    def __init__(self) -> None:
        self.used: Set[str] = set()

    def take(self, prefix: str, raw: Any) -> str:
        base = re.sub(r"\W", "_", f"{prefix}{raw}")
        if not base or base[0].isdigit():
            base = "_" + base
        if keyword.iskeyword(base):
            base += "_"
        name = base
        counter = 1
        while name in self.used:
            counter += 1
            name = f"{base}_{counter}"
        self.used.add(name)
        return name


@dataclass
class _Layout:
    order: List[str]
    edge_var: Dict[str, str]
    edge_expr: Dict[str, str]
    param_attr: Dict[str, str]
    func_attr: Dict[str, str]
    input_args: List[str]
    returns: List[str]


def _build_layout(graph: Graph) -> _Layout:
    pool = _NamePool()
    order = graph.topological_order()
    edge_var: Dict[str, str] = {}
    edge_expr: Dict[str, str] = {}
    param_attr: Dict[str, str] = {}
    func_attr: Dict[str, str] = {}
    for nid in order:
        node = graph.nodes[nid]
        outs = graph.out_edges(nid)
        for e in outs:
            edge_var.setdefault(e, pool.take("e", e))
        if node.op == "param":
            param_attr[nid] = pool.take("p_", nid)
        elif node.op not in ("input", "output"):
            func_attr[nid] = pool.take("f_", nid)
        for e in outs:
            if node.op == "param":
                edge_expr[e] = f"self.{param_attr[nid]}"
            else:
                edge_expr[e] = edge_var[e]

    input_args: List[str] = []
    for eid in graph.inputs:
        if eid not in edge_var:
            raise _SourceGenError(f"graph input edge {eid!r} has no producing node")
        input_args.append(edge_var[eid])

    returns: List[str] = []
    for eid in graph.outputs:
        if eid not in edge_expr:
            raise _SourceGenError(f"graph output edge {eid!r} is never produced")
        returns.append(edge_expr[eid])
    if not returns:
        last: Optional[str] = None
        for nid in order:
            if graph.nodes[nid].op in ("input", "param", "output"):
                continue
            outs = graph.out_edges(nid)
            if outs:
                last = edge_expr[outs[-1]]
        if last is not None:
            returns = [last]
    return _Layout(order=order, edge_var=edge_var, edge_expr=edge_expr,
                   param_attr=param_attr, func_attr=func_attr,
                   input_args=input_args, returns=returns)


def param_attribute_names(graph: Graph) -> Dict[str, str]:
    """``node id -> p_<node> attribute name`` (identical to the generated file)."""
    return dict(_build_layout(graph).param_attr)


def function_attribute_names(graph: Graph) -> Dict[str, str]:
    """``node id -> f_<node> attribute name`` (identical to the generated file)."""
    return dict(_build_layout(graph).func_attr)


def _literal(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, str)):
        return repr(value)
    if isinstance(value, float):
        if value != value:
            return "float('nan')"
        if value == float("inf"):
            return "float('inf')"
        if value == float("-inf"):
            return "float('-inf')"
        return repr(value)
    if isinstance(value, np.bool_):
        return repr(bool(value))
    if isinstance(value, np.integer):
        return repr(int(value))
    if isinstance(value, np.floating):
        return _literal(float(value))
    if isinstance(value, np.ndarray):
        return _literal(value.tolist())
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_literal(k)}: {_literal(v)}"
                               for k, v in value.items()) + "}"
    raise _SourceGenError(f"attribute value {value!r} is not a Python literal")


def _sanitise_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in attrs.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise _SourceGenError(f"attribute key {key!r} is not an identifier")
        clean[key] = value
    return clean


def generate_cell_source(graph: Graph, class_name: str,
                         bank_keys: Dict[str, str]) -> str:
    """Generate the source of an ``nn.Cell`` that executes ``graph``.

    ``bank_keys`` maps a parameter node id to the :class:`ParamBank` key whose
    tensor the generated ``__init__`` wraps in an ``ms.Parameter``.  The graph is
    executed with one plain statement per node, so MindSpore's parser sees an
    ordinary per-node program.
    """
    layout = _build_layout(graph)
    lines: List[str] = []
    lines.append('"""Generated by FlowMuT -- do not edit."""')
    lines.append("import mindspore as ms")
    lines.append("from mindspore import nn")
    lines.append("from flowmut.adapters.ms_kernels import make_fn")
    lines.append("")
    lines.append("")
    lines.append(f"class {class_name}(nn.Cell):")
    lines.append("    def __init__(self, bank):")
    lines.append("        super().__init__()")
    created = 0
    for nid in layout.order:
        node = graph.nodes[nid]
        if node.op == "param":
            key = bank_keys.get(nid)
            if key is None:
                raise _SourceGenError(f"parameter node {nid!r} has no bank key")
            attr = layout.param_attr[nid]
            lines.append(f"        self.{attr} = ms.Parameter(bank[{key!r}], "
                         f"name={attr!r})")
            created += 1
        elif node.op not in ("input", "output"):
            attr = layout.func_attr[nid]
            attrs = _sanitise_attrs(_attrs_for_node(node, graph))
            op = canonical_op(node.op)
            lines.append(f"        self.{attr} = make_fn({op!r}, {_literal(attrs)})")
            created += 1
    if created == 0:
        lines.append("        pass")
    signature = ", ".join(["self"] + layout.input_args)
    lines.append("")
    lines.append(f"    def construct({signature}):")

    body: List[str] = []
    for nid in layout.order:
        node = graph.nodes[nid]
        outs = graph.out_edges(nid)
        if node.op in ("input", "param"):
            continue
        if node.op == "output":
            if not node.inputs:
                raise _SourceGenError(f"output node {nid!r} has no input")
            source = layout.edge_expr[node.inputs[0]]
            for e in outs:
                body.append(f"        {layout.edge_var[e]} = {source}")
            continue
        attr = layout.func_attr[nid]
        args = ", ".join(layout.edge_expr[e] for e in node.inputs)
        call = f"self.{attr}({args})"
        targets = [layout.edge_var[e] for e in outs]
        if len(targets) == 1:
            body.append(f"        {targets[0]} = {call}")
        else:
            body.append(f"        {', '.join(targets)} = {call}")

    if layout.returns:
        if len(layout.returns) == 1:
            body.append(f"        return {layout.returns[0]}")
        else:
            body.append(f"        return ({', '.join(layout.returns)})")
    if not body:
        body.append("        pass")
    lines.extend(body)
    lines.append("")
    return "\n".join(lines)


def build_graph_module(graph: Graph, bank: Any, cache_dir: str
                       ) -> Tuple[Any, str]:
    """Write, import and instantiate the graph-mode ``nn.Cell``.

    ``bank`` is either a :class:`ParamBank` or a ``signature -> tensor`` mapping.
    Returns ``(cell_instance, module_path)``.
    """
    tensors = bank.tensors if isinstance(bank, ParamBank) else bank
    fingerprint = graph_fingerprint(graph)
    class_name = f"FlowMuTCell_{fingerprint}"
    bank_keys = {
        nid: node_signature(node)
        for nid, node in graph.nodes.items() if node.op == "param"
    }
    source = generate_cell_source(graph, class_name, bank_keys)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"flowmut_ms_{fingerprint}.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source)
    module_name = f"flowmut_ms_{fingerprint}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise _SourceGenError(f"cannot import generated module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    cell_class = getattr(module, class_name, None)
    if cell_class is None:
        raise _SourceGenError(f"generated module {path} has no class {class_name}")
    cell = cell_class(tensors)
    return cell, path


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------

@dataclass
class MSBundle:
    """The materialised form of one graph in one MindSpore mode."""

    mode: str
    fingerprint: str
    bank: ParamBank
    pynative: PyNativeInterpreter
    cell: Any = None
    module_path: str = ""
    fallback: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------

class MindSporeAdapter(FrameworkAdapter):
    """Shared behaviour of the two MindSpore modes."""

    framework = "mindspore"
    mode = "pynative"

    def __init__(self, mode: Optional[str] = None, device: str = "cpu",
                 seed: int = 20240929, timeout_s: Optional[float] = None,
                 capture_values: bool = True, trace_limit: int = 2048,
                 cache_dir: Optional[str] = None, **kwargs: Any):
        super().__init__(mode=mode or type(self).mode or "pynative", device=device,
                         seed=seed, timeout_s=timeout_s,
                         capture_values=capture_values)
        self.trace_limit = int(trace_limit)
        self.bank = ParamBank(seed=self.seed)
        self._compile_cache: Dict[str, MSBundle] = {}
        self._executed: Set[str] = set()
        self.cache_dir = str(
            cache_dir or os.environ.get("FLOWMUT_MS_CACHE")
            or tempfile.mkdtemp(prefix="flowmut_ms_"))
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception:  # pragma: no cover - defensive
            self.cache_dir = tempfile.mkdtemp(prefix="flowmut_ms_")

    # -- capabilities ------------------------------------------------------
    @classmethod
    def is_available(cls) -> Tuple[bool, str]:
        if ms is None:
            return False, f"mindspore import failed: {_MS_IMPORT_ERROR}"
        try:
            _kernels()
        except Exception as exc:
            return False, f"ms_kernels import failed: {type(exc).__name__}: {exc}"
        return True, f"mindspore {MS_VERSION}"

    def describe(self) -> Dict[str, Any]:
        data = super().describe()
        data.update({"device": "cpu", "mindspore": MS_VERSION,
                     "cache_dir": self.cache_dir})
        return data

    # -- tensors -----------------------------------------------------------
    def to_tensor(self, array: np.ndarray, dtype: str, device: Optional[str] = None,
                  requires_grad: bool = False) -> Any:
        if ms is None:
            raise RuntimeError(f"mindspore is not available: {_MS_IMPORT_ERROR}")
        data = np.ascontiguousarray(array)
        target = _numpy_dtype(dtype)
        if obj_dtype_mismatch(data.dtype, target):
            data = data.astype(target)
        return ms.Tensor(data)

    def to_numpy(self, tensor: Any) -> Optional[np.ndarray]:
        if tensor is None:
            return None
        try:
            if isinstance(tensor, (list, tuple)):
                arrays = [self.to_numpy(t) for t in tensor]
                if any(a is None for a in arrays):
                    return None
                return np.asarray(arrays)
            if MSTensor is not None and isinstance(tensor, MSTensor):
                return tensor.asnumpy()
            return np.asarray(tensor)
        except Exception:
            return None

    def tensor_spec(self, tensor: Any) -> TensorSpec:
        if MSTensor is None or not isinstance(tensor, MSTensor):
            return TensorSpec()
        strides = None
        try:
            getter = getattr(tensor, "strides", None)
            if callable(getter):
                values = getter()
                if values is not None:
                    strides = tuple(int(s) for s in values)
        except Exception:
            strides = None
        value = None
        if self.capture_values:
            try:
                value = value_profile(self.to_numpy(tensor))
            except Exception:
                value = None
        try:
            numel = int(tensor.size)
        except Exception:
            numel = 0
        return TensorSpec(
            shape=tuple(int(d) for d in tensor.shape),
            dtype=_kernels().dtype_name(tensor.dtype),
            layout="contiguous",
            device="cpu",
            requires_grad=_requires_grad(tensor),
            alias_of=None,
            is_view=False,
            is_leaf=True,
            strides=strides,
            value=value,
            numel=numel,
        )

    # -- materialisation ---------------------------------------------------
    def _param_tensors(self, graph: Graph) -> Dict[str, Any]:
        return {nid: self.bank.tensor_for(node, graph)
                for nid, node in graph.nodes.items() if node.op == "param"}

    def materialize(self, graph: Graph, reuse: Any = None) -> MSBundle:
        params = self._param_tensors(graph)
        fingerprint = graph_fingerprint(graph)
        interpreter = PyNativeInterpreter(graph, params, fingerprint)
        return MSBundle(mode=self.mode, fingerprint=fingerprint, bank=self.bank,
                        pynative=interpreter, params=params)

    def _inputs_for(self, graph: Graph, sample: Sample) -> List[Any]:
        tensors: List[Any] = []
        names = {graph.edges[e].name: e for e in graph.inputs}
        for edge_id in graph.inputs:
            edge = graph.edges[edge_id]
            array = None
            if edge_id in sample:
                array = sample[edge_id]
            elif edge.name in sample:
                array = sample[edge.name]
            else:
                for key, value in sample.items():
                    if names.get(key) == edge_id:
                        array = value
                        break
            spec = edge.spec
            dtype = spec.dtype if spec else "float32"
            if array is None:
                shape = tuple(spec.shape) if spec else (1,)
                array = np.zeros(shape, dtype=np.float32 if dtype.startswith("float")
                                 else np.int64)
            tensors.append(self.to_tensor(array, dtype))
        return tensors

    # -- execution ---------------------------------------------------------
    def execute(self, graph: Graph, sample: Sample, module: Any = None,
                capture_specs: bool = False) -> ExecutionResult:
        bundle = module if isinstance(module, MSBundle) else self.materialize(graph)
        recorder = TraceRecorder(max_events=self.trace_limit)
        started = time.perf_counter()
        try:
            with TimeoutGuard(self.timeout_s):
                inputs = self._inputs_for(graph, sample)
                outputs = self._run_once(bundle, inputs, graph, recorder, capture_specs)
            duration = time.perf_counter() - started
            specs = self._collect_specs(bundle, graph) if capture_specs else {}
            arrays = [self.to_numpy(o) for o in _as_output_list(outputs)] \
                if self.capture_values else []
            return ExecutionResult(status="ok", framework=self.framework,
                                   mode=self.mode, duration_s=duration,
                                   events=recorder.events, edge_specs=specs,
                                   outputs=arrays, extras=self._extras(bundle))
        except TimeoutExceeded:
            return ExecutionResult(status="timeout", framework=self.framework,
                                   mode=self.mode,
                                   duration_s=time.perf_counter() - started,
                                   error_type="TimeoutExceeded",
                                   error_message=f"execution exceeded {self.timeout_s}s",
                                   events=recorder.events, extras=self._extras(bundle))
        except Exception as exc:
            return ExecutionResult(
                status="error", framework=self.framework, mode=self.mode,
                duration_s=time.perf_counter() - started,
                error_type=type(exc).__name__, error_message=str(exc)[:2000],
                error_traceback=traceback.format_exc()[-6000:],
                events=recorder.events, extras=self._extras(bundle))

    # -- hooks -------------------------------------------------------------
    def _run_once(self, bundle: MSBundle, inputs: List[Any], graph: Graph,
                  recorder: TraceRecorder, capture_specs: bool) -> Sequence[Any]:
        raise NotImplementedError

    def _collect_specs(self, bundle: MSBundle, graph: Graph
                       ) -> Dict[str, List[TensorSpec]]:
        taps = bundle.pynative.taps
        counts: Counter = Counter()
        for tensor in taps.values():
            if MSTensor is not None and isinstance(tensor, MSTensor):
                counts[id(tensor)] += 1
        specs: Dict[str, List[TensorSpec]] = {}
        storage: Dict[int, str] = {}
        for eid, tensor in taps.items():
            spec = self.tensor_spec(tensor)
            if MSTensor is not None and isinstance(tensor, MSTensor) \
                    and counts[id(tensor)] > 1:
                key = id(tensor)
                storage.setdefault(key, f"buf{len(storage)}")
                spec = spec.with_(alias_of=storage[key])
            specs[eid] = [spec]
        return specs

    def _extras(self, bundle: MSBundle) -> Dict[str, Any]:
        extras: Dict[str, Any] = {
            "fingerprint": bundle.fingerprint,
            "params": self.bank.numel,
            "mindspore": MS_VERSION,
            "mode": self.mode,
            "device": "cpu",
            "graph_fallback": bundle.fallback,
            "cache_dir": self.cache_dir,
            "aliasing_detection": "object_identity",
            "profile_pass": "pynative",
        }
        if bundle.module_path:
            extras["graph_module"] = bundle.module_path
        if bundle.cell is not None:
            extras["graph_cell"] = type(bundle.cell).__name__
        return extras

    def close(self) -> None:
        super().close()
        self._compile_cache.clear()
        self._executed.clear()


class MindSporePyNativeAdapter(MindSporeAdapter):
    """MindSpore in PyNative (eager) mode."""

    framework = "mindspore"
    mode = "pynative"

    def _run_once(self, bundle: MSBundle, inputs: List[Any], graph: Graph,
                  recorder: TraceRecorder, capture_specs: bool) -> Sequence[Any]:
        MSEnv.ensure(device="cpu", mode="pynative")
        for node in graph.nodes.values():
            if node.op in ("cast", "contiguous", "clone", "detach", "stop_gradient"):
                recorder.record(f"adaptation.{canonical_op(node.op)}", "adaptation")
        recorder.record("control.pynative_node_loop", "control")
        interpreter = bundle.pynative
        interpreter.tap_enabled = bool(capture_specs)
        outputs = interpreter.run(inputs, recorder=recorder)
        recorder.record("control.pynative_eager_dispatch", "control")
        return outputs


class MindSporeGraphAdapter(MindSporeAdapter):
    """MindSpore in Graph mode (generated ``nn.Cell`` + ``ms.Parameter`` bank)."""

    framework = "mindspore"
    mode = "graph"

    def materialize(self, graph: Graph, reuse: Any = None) -> MSBundle:
        params = self._param_tensors(graph)
        fingerprint = graph_fingerprint(graph)
        cached = self._compile_cache.get(fingerprint)
        if cached is not None and cached.bank is self.bank:
            return cached
        interpreter = PyNativeInterpreter(graph, params, fingerprint)
        bundle = MSBundle(mode=self.mode, fingerprint=fingerprint, bank=self.bank,
                          pynative=interpreter, params=params)
        try:
            MSEnv.ensure(device="cpu", mode="graph")
            cell, path = build_graph_module(graph, self.bank.tensors, self.cache_dir)
            bundle.cell = cell
            bundle.module_path = path
            # the annotation pass must see exactly the tensors the compiled
            # program consumes, so the ``p_<node>`` Parameters are shared
            names = param_attribute_names(graph)
            shared: Dict[str, Any] = {}
            for nid, attr in names.items():
                parameter = getattr(cell, attr, None)
                if parameter is not None:
                    shared[nid] = parameter
            bundle.params = shared or params
            bundle.pynative.params = bundle.params
        except Exception as exc:
            bundle.cell = None
            bundle.fallback = f"{type(exc).__name__}: {exc}"[:500]
        self._compile_cache[fingerprint] = bundle
        return bundle

    def _run_once(self, bundle: MSBundle, inputs: List[Any], graph: Graph,
                  recorder: TraceRecorder, capture_specs: bool) -> Sequence[Any]:
        if bundle.cell is None:
            recorder.record("control.graph_mode_fallback", "control")
            MSEnv.ensure(device="cpu", mode="pynative")
            interpreter = bundle.pynative
            interpreter.tap_enabled = bool(capture_specs)
            outputs = interpreter.run(inputs, recorder=recorder)
            recorder.record("control.graph_mode_execution", "control")
            self._record_graph_dispatch(graph, recorder)
            return outputs

        first = bundle.fingerprint not in self._executed
        MSEnv.ensure(device="cpu", mode="graph")
        if first:
            recorder.record("graph_capture.ms_graph_compile_enter", "graph_capture")
        else:
            recorder.record("graph_capture.ms_graph_reuse", "graph_capture")
        try:
            outputs = bundle.cell(*inputs)
        finally:
            if first:
                recorder.record("optimization.ms_graph_optimize", "optimization")
                recorder.record("graph_capture.ms_graph_compile_exit", "graph_capture")
                self._executed.add(bundle.fingerprint)
        recorder.record("control.graph_mode_execution", "control")
        self._record_graph_dispatch(graph, recorder)

        if capture_specs:
            # A compiled MindSpore graph does not expose intermediate tensors, so
            # the annotation is collected from a PyNative pass over the same
            # parameters (see the module docstring).
            interpreter = bundle.pynative
            interpreter.tap_enabled = True
            try:
                with MSEnv.use_mode("pynative"):
                    interpreter.run(inputs)
            except Exception:
                interpreter.tap_enabled = False
            finally:
                MSEnv.ensure(device="cpu", mode="graph")
        return outputs

    def _record_graph_dispatch(self, graph: Graph, recorder: TraceRecorder) -> None:
        """The graph-mode dispatch surface: one event per node in the graph."""
        try:
            order = graph.topological_order()
        except Exception:
            order = list(graph.nodes)
        for nid in order:
            node = graph.nodes[nid]
            if node.op in ("input", "param", "output"):
                continue
            recorder.record(f"dispatcher.ops.{canonical_op(node.op)}", "dispatcher")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def obj_dtype_mismatch(source: Any, target: Any) -> bool:
    """True when a numpy array must be cast before becoming a MindSpore tensor."""
    try:
        return np.dtype(source) != np.dtype(target)
    except Exception:
        return False


def _as_output_list(outputs: Any) -> List[Any]:
    if isinstance(outputs, (tuple, list)):
        return list(outputs)
    return [outputs]
