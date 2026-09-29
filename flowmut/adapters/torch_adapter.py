"""PyTorch adapter (eager and ``torch.compile`` modes).

The adapter turns a framework-neutral :class:`~flowmut.ir.Graph` into an
``nn.Module`` whose ``forward`` executes the graph node by node in topological
order, so that

* every edge's runtime tensor can be captured as the TFG annotation ``F(e)``, and
* the framework's own control flow (dispatcher entries, autograd actions, graph
  capture and optimisation decisions, tensor adaptations, memory/alias handling)
  can be observed as the execution trace.

Two modes are provided:

``PyTorchEagerAdapter``
    the graph runs op by op under a ``TorchDispatchMode`` that records the
    dispatcher's choices;
``PyTorchCompiledAdapter``
    the whole module is wrapped in ``torch.compile``; the trace records the
    graph-capture and optimisation decisions taken by Dynamo/Inductor, and the
    ``F(e)`` annotations are collected from an eager twin that **shares the same
    parameter tensors**, because a compiled graph deliberately does not expose
    intermediate tensors.
"""

from __future__ import annotations

import hashlib
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:  # the dispatch-mode API has moved between releases
    from torch.utils._python_dispatch import TorchDispatchMode
except Exception:  # pragma: no cover - very old/new torch
    TorchDispatchMode = None  # type: ignore[assignment]

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
from flowmut.adapters.torch_kernels import (
    KernelMissing,
    apply as apply_kernel,
    dtype_name,
    init_parameter,
)
from flowmut.ir.graph import Graph, Node
from flowmut.ir.specs import TensorSpec
from flowmut.trace.normalize import TraceRecorder

TORCH_VERSION = torch.__version__


# ---------------------------------------------------------------------------
# Parameter bank
# ---------------------------------------------------------------------------

class ParamBank:
    """Deterministically initialised parameter tensors, reused across mutants.

    Parameters are keyed by :func:`~flowmut.adapters.base.node_signature`, which
    is invariant under graph rewriting.  A mutation that only changes, say, an
    activation therefore keeps the surrounding convolution weights, so the
    mutant differs from its parent exactly in the mutated region.
    """

    def __init__(self, seed: int = 20240929):
        self.seed = int(seed)
        self.tensors: Dict[str, torch.Tensor] = {}

    def key_for(self, node: Node) -> str:
        return node_signature(node)

    def tensor_for(self, node: Node, graph: Graph) -> torch.Tensor:
        key = self.key_for(node)
        cached = self.tensors.get(key)
        if cached is not None:
            return cached
        shape = tuple(node.out_shape or _declared_shape(node, graph) or ())
        dtype = str(node.attrs.get("param_dtype", "float32"))
        tensor = torch.empty(shape, dtype=_torch_dtype(dtype))
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        generator = torch.Generator()
        generator.manual_seed((int(digest, 16) ^ self.seed) % (2 ** 31))
        init_parameter(tensor, str(node.attrs.get("init", "kaiming_uniform")), generator)
        tensor.requires_grad_(True)
        self.tensors[key] = tensor
        return tensor

    @property
    def numel(self) -> int:
        return sum(int(t.numel()) for t in self.tensors.values())


def _declared_shape(node: Node, graph: Graph) -> Optional[Tuple[int, ...]]:
    for eid in graph.out_edges(node.id):
        spec = graph.edges[eid].spec
        if spec is not None:
            return tuple(spec.shape)
    return None


def _torch_dtype(name: str) -> torch.dtype:
    from flowmut.adapters.torch_kernels import _TORCH_DTYPES
    return _TORCH_DTYPES.get(str(name), torch.float32)


# ---------------------------------------------------------------------------
# IR module
# ---------------------------------------------------------------------------

class IRModule(nn.Module):
    """An ``nn.Module`` that executes an IR graph node by node."""

    def __init__(self, graph: Graph, bank: ParamBank):
        super().__init__()
        self.graph_name = graph.name
        self.fingerprint = graph_fingerprint(graph)
        self._nodes: Dict[str, Node] = dict(graph.nodes)
        self._order: List[str] = graph.topological_order()
        self._input_edges: List[str] = list(graph.inputs)
        self._output_edges: List[str] = list(graph.outputs)
        self._out_edges: Dict[str, List[str]] = {
            nid: graph.out_edges(nid) for nid in self._nodes
        }
        self._edge_index: Dict[str, int] = {e: i for i, e in enumerate(self._input_edges)}
        self.tap_enabled: bool = False
        self.taps: Dict[str, torch.Tensor] = {}
        self.alias_log: List[str] = []
        self.node_events: List[str] = []
        self._params: Dict[str, torch.Tensor] = {}
        for nid in self._order:
            node = self._nodes[nid]
            if node.op == "param":
                tensor = bank.tensor_for(node, graph)
                self._params[nid] = tensor
                self.register_parameter(f"p_{nid}", nn.Parameter(tensor, requires_grad=True))

    # -- execution ---------------------------------------------------------
    def forward(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        vals: Dict[str, torch.Tensor] = {}
        if len(inputs) != len(self._input_edges):
            raise ValueError(f"expected {len(self._input_edges)} inputs, got {len(inputs)}")
        for edge_id, tensor in zip(self._input_edges, inputs):
            vals[edge_id] = tensor
        taps = self.taps if self.tap_enabled else None
        if taps is not None:
            taps.clear()
            for edge_id, tensor in zip(self._input_edges, inputs):
                taps[edge_id] = tensor
        for nid in self._order:
            node = self._nodes[nid]
            out_edges = self._out_edges[nid]
            if node.op == "input":
                continue
            if node.op == "param":
                tensor = self._params[nid]
                for e in out_edges:
                    vals[e] = tensor
                if taps is not None:
                    for e in out_edges:
                        taps[e] = tensor
                continue
            tensors = []
            for position, e in enumerate(node.inputs):
                if e not in vals:
                    raise RuntimeError(
                        f"node {nid} ({node.op}) input {position} edge {e} was never produced")
                tensors.append(vals[e])
            result = apply_kernel(node.op, tensors, node.attrs)
            if isinstance(result, (tuple, list)):
                produced = list(result)
            else:
                produced = [result]
            if len(produced) != len(out_edges):
                raise RuntimeError(
                    f"node {nid} ({node.op}) produced {len(produced)} tensors "
                    f"but the graph declares {len(out_edges)} outputs")
            for e, tensor in zip(out_edges, produced):
                vals[e] = tensor
            if taps is not None:
                for e, tensor in zip(out_edges, produced):
                    taps[e] = tensor
        if not self._output_edges:
            # A graph without declared outputs (possible after aggressive
            # deletions) still has to return something for the executor.
            return (list(vals.values())[-1],)
        return tuple(vals[e] for e in self._output_edges)


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------

class _DispatchRecorder:
    """Context manager that records every ``TorchDispatchMode`` entry.

    The dispatcher is where a framework chooses the backend kernel for an
    operator, so its entries are exactly the "dispatcher choices" the paper maps
    onto trace labels.  Tensor values, producer identities and kernel names are
    not recorded.
    """

    def __init__(self, limit: int = 2048):
        self.limit = int(limit)
        self.labels: List[str] = []
        self._mode = None

    def __enter__(self) -> "_DispatchRecorder":
        if TorchDispatchMode is None:
            return self
        labels, limit = self.labels, self.limit

        class _Mode(TorchDispatchMode):  # type: ignore[misc, valid-type]
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if len(labels) < limit:
                    labels.append(str(func))
                return func(*args, **(kwargs or {}))

        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, *exc) -> bool:
        if self._mode is not None:
            try:
                self._mode.__exit__(*exc)
            except Exception:
                pass
        return False


def _compile_metrics() -> Dict[str, float]:
    """Snapshot the graph-capture / optimisation counters, if available."""
    out: Dict[str, float] = {}
    try:
        from torch._dynamo.utils import counters  # type: ignore
        for key, value in counters.items():
            if isinstance(value, dict):
                for sub, n in value.items():
                    try:
                        out[f"dynamo.{key}.{sub}"] = float(n)
                    except (TypeError, ValueError):
                        continue
            else:
                try:
                    out[f"dynamo.{key}"] = float(value)
                except (TypeError, ValueError):
                    continue
    except Exception:
        pass
    try:
        import torch._inductor.metrics as metrics  # type: ignore
        for name in ("generated_kernel_count", "generated_cpp_vec_kernel_count",
                     "cpp_outer_loop_fused_inner_counts", "codegen_mix_order_reduction",
                     "cpp_to_dtype_count"):
            value = getattr(metrics, name, None)
            if isinstance(value, (int, float)):
                out[f"inductor.{name}"] = float(value)
    except Exception:
        pass
    return out


def _metric_events(before: Dict[str, float], after: Dict[str, float]) -> List[str]:
    events: List[str] = []
    for key, value in after.items():
        delta = value - before.get(key, 0.0)
        if delta <= 0:
            continue
        if key.startswith("dynamo.") and ("graph_break" in key or "fail" in key):
            events.append(f"graph_capture.graph_break.{key}")
        elif key.startswith("inductor."):
            events.append(f"optimization.{key}")
        elif "compile" in key or "frame" in key or "graph" in key:
            events.append(f"graph_capture.{key}")
        else:
            events.append(f"control.{key}")
    return events


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@dataclass
class TorchBundle:
    """The materialised form of one graph."""

    eager: IRModule
    compiled: Any = None
    fingerprint: str = ""
    compile_error: str = ""


class PyTorchAdapter(FrameworkAdapter):
    """Shared behaviour of the two PyTorch modes."""

    framework = "pytorch"

    def __init__(self, mode: str = "eager", device: str = "cpu", seed: int = 20240929,
                 timeout_s: Optional[float] = None, capture_values: bool = True,
                 trace_limit: int = 2048, compile_cache_size: int = 8, **kwargs: Any):
        super().__init__(mode=mode, device=device, seed=seed, timeout_s=timeout_s,
                         capture_values=capture_values)
        self.trace_limit = int(trace_limit)
        self.bank = ParamBank(seed=self.seed)
        self._compile_cache: Dict[str, Any] = {}
        self.compile_cache_size = int(compile_cache_size)
        self._device = self._resolve_device(device)

    # -- capabilities ------------------------------------------------------
    @classmethod
    def is_available(cls) -> Tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except Exception as exc:
            return False, f"torch import failed: {exc}"
        return True, f"torch {TORCH_VERSION}"

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["device"] = self._device
        d["torch"] = TORCH_VERSION
        return d

    # -- tensors -----------------------------------------------------------
    def to_tensor(self, array: np.ndarray, dtype: str, device: Optional[str] = None,
                  requires_grad: bool = False) -> torch.Tensor:
        tensor = torch.as_tensor(np.ascontiguousarray(array), dtype=_torch_dtype(dtype))
        tensor = tensor.to(device or self._device)
        if requires_grad and tensor.is_floating_point():
            tensor = tensor.clone().requires_grad_(True)
        return tensor

    def to_numpy(self, tensor: Any) -> Optional[np.ndarray]:
        if tensor is None:
            return None
        try:
            t = tensor
            if isinstance(t, torch.Tensor) and t.requires_grad:
                t = t.detach()
            return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        except Exception:
            return None

    def tensor_spec(self, tensor: Any) -> TensorSpec:
        if not isinstance(tensor, torch.Tensor):
            return TensorSpec()
        layout = "contiguous"
        try:
            if tensor.dim() == 4 and tensor.is_contiguous(memory_format=torch.channels_last):
                layout = "channels_last"
            elif not tensor.is_contiguous():
                layout = "strided"
        except Exception:
            layout = "contiguous"
        alias = None
        try:
            alias = str(tensor.untyped_storage().data_ptr())
        except Exception:
            alias = None
        is_view = False
        try:
            is_view = bool(tensor._is_view()) if hasattr(tensor, "_is_view") else \
                (tensor.base is not None)
        except Exception:
            is_view = tensor.base is not None
        value = None
        if self.capture_values:
            try:
                value = value_profile(self.to_numpy(tensor))
            except Exception:
                value = None
        return TensorSpec(
            shape=tuple(int(d) for d in tensor.shape),
            dtype=dtype_name(tensor.dtype),
            layout=layout,
            device=str(tensor.device.type),
            requires_grad=bool(tensor.requires_grad),
            alias_of=None,
            is_view=is_view,
            is_leaf=bool(tensor.is_leaf),
            strides=tuple(int(s) for s in tensor.stride()),
            value=value,
            numel=int(tensor.numel()),
        )

    # -- materialisation ---------------------------------------------------
    def materialize(self, graph: Graph, reuse: Any = None) -> TorchBundle:
        if isinstance(reuse, TorchBundle):
            pass
        eager = IRModule(graph, self.bank)
        return TorchBundle(eager=eager, fingerprint=eager.fingerprint)

    def _inputs_for(self, graph: Graph, sample: Sample) -> List[torch.Tensor]:
        tensors: List[torch.Tensor] = []
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
            if array is None:
                spec = edge.spec
                shape = tuple(spec.shape) if spec else (1,)
                dtype = spec.dtype if spec else "float32"
                array = np.zeros(shape, dtype=np.float32 if dtype.startswith("float")
                                 else np.int64)
            spec = edge.spec
            dtype = spec.dtype if spec else "float32"
            tensors.append(self.to_tensor(array, dtype))
        return tensors

    # -- execution ---------------------------------------------------------
    def execute(self, graph: Graph, sample: Sample, module: Any = None,
                capture_specs: bool = False) -> ExecutionResult:
        bundle = module if isinstance(module, TorchBundle) else self.materialize(graph)
        inputs = self._inputs_for(graph, sample)
        recorder = TraceRecorder(max_events=self.trace_limit)
        started = time.perf_counter()
        try:
            with TimeoutGuard(self.timeout_s):
                outputs = self._run_once(bundle, inputs, graph, recorder, capture_specs)
            duration = time.perf_counter() - started
            specs = self._collect_specs(bundle, graph) if capture_specs else {}
            arrays = [self.to_numpy(o) for o in outputs] if self.capture_values else []
            return ExecutionResult(status="ok", framework=self.framework, mode=self.mode,
                                   duration_s=duration, events=recorder.events,
                                   edge_specs=specs, outputs=arrays,
                                   extras=self._extras(bundle))
        except TimeoutExceeded:
            return ExecutionResult(status="timeout", framework=self.framework, mode=self.mode,
                                   duration_s=time.perf_counter() - started,
                                   error_type="TimeoutExceeded",
                                   error_message=f"execution exceeded {self.timeout_s}s",
                                   events=recorder.events)
        except Exception as exc:
            return ExecutionResult(
                status="error", framework=self.framework, mode=self.mode,
                duration_s=time.perf_counter() - started,
                error_type=type(exc).__name__, error_message=str(exc)[:2000],
                error_traceback=traceback.format_exc()[-6000:],
                events=recorder.events, extras=self._extras(bundle))

    # -- hooks -------------------------------------------------------------
    def _run_once(self, bundle: TorchBundle, inputs: List[torch.Tensor], graph: Graph,
                  recorder: TraceRecorder, capture_specs: bool) -> Sequence[torch.Tensor]:
        raise NotImplementedError

    def _collect_specs(self, bundle: TorchBundle, graph: Graph
                       ) -> Dict[str, List[TensorSpec]]:
        taps = bundle.eager.taps
        specs: Dict[str, List[TensorSpec]] = {}
        storage: Dict[str, str] = {}
        for eid, tensor in taps.items():
            spec = self.tensor_spec(tensor)
            key = spec.alias_of or ""
            if key:
                storage.setdefault(key, f"buf{len(storage)}")
                spec = spec.with_(alias_of=storage[key])
            specs[eid] = [spec]
        return specs

    def _extras(self, bundle: TorchBundle) -> Dict[str, Any]:
        return {"fingerprint": bundle.fingerprint, "params": self.bank.numel}

    # -- gradient influence (the I(c) feature) ----------------------------
    def gradient_influence_map(self, graph: Graph, sample: Sample, module: Any = None
                               ) -> Dict[str, float]:
        """``(1/|a|) * |d ell / d a|`` for **every** edge, in one backward pass.

        The paper defines the gradient-influence feature per candidate.  Because
        ``ell`` is a scalar reduction of the model output, a single
        ``torch.autograd.grad`` call over all tapped tensors yields the value for
        every anchor at once, which keeps the feature affordable per round.
        """
        bundle = module if isinstance(module, TorchBundle) else self.materialize(graph)
        module_ir = bundle.eager
        inputs = self._inputs_for(graph, sample)
        result: Dict[str, float] = {}
        module_ir.tap_enabled = True
        try:
            with torch.enable_grad():
                outputs = module_ir(*inputs)
                taps = [(eid, t) for eid, t in module_ir.taps.items()
                        if isinstance(t, torch.Tensor) and t.requires_grad
                        and t.is_floating_point()]
                if not taps:
                    return result
                scalar = None
                for out in outputs:
                    if isinstance(out, torch.Tensor) and out.is_floating_point():
                        contribution = out.float().sum()
                        scalar = contribution if scalar is None else scalar + contribution
                if scalar is None:
                    return result
                grads = torch.autograd.grad(scalar, [t for _, t in taps],
                                            retain_graph=False, allow_unused=True)
                for (eid, tensor), grad in zip(taps, grads):
                    if grad is None:
                        continue
                    numel = max(1, int(tensor.numel()))
                    result[eid] = float(grad.detach().abs().sum().item()) / numel
        except Exception:
            return result
        finally:
            module_ir.tap_enabled = False
        return result


class PyTorchEagerAdapter(PyTorchAdapter):
    """PyTorch in eager mode."""

    framework = "pytorch"
    mode = "eager"

    def _run_once(self, bundle: TorchBundle, inputs, graph, recorder, capture_specs):
        module = bundle.eager
        module.tap_enabled = bool(capture_specs)
        module.alias_log = []
        if torch.is_grad_enabled():
            recorder.record("autograd.grad_enabled", "autograd")
        else:
            recorder.record("autograd.no_grad", "autograd")
        for node in graph.nodes.values():
            if node.op in ("cast", "contiguous", "clone", "detach", "stop_gradient",
                           "to"):
                recorder.record(f"adaptation.{node.op}", "adaptation")
        with _DispatchRecorder(limit=self.trace_limit) as dispatch:
            outputs = module(*inputs)
        for label in dispatch.labels:
            recorder.record(label)
        recorder.record("control.eager_node_loop", "control")
        try:
            if any(t.grad_fn is not None for t in outputs if isinstance(t, torch.Tensor)):
                recorder.record("autograd.tracked_output", "autograd")
        except Exception:
            pass
        return outputs


class PyTorchCompiledAdapter(PyTorchAdapter):
    """PyTorch with ``torch.compile`` (mode ``compiled``)."""

    framework = "pytorch"
    mode = "compiled"

    def __init__(self, *args: Any, backend: str = "inductor",
                 fullgraph: bool = False, dynamic: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.backend = str(backend)
        self.fullgraph = bool(fullgraph)
        self.dynamic = bool(dynamic)
        # Every mutant is a fresh module, so Dynamo's default recompile budget is
        # exhausted after a handful of rounds and quietly falls back to eager.
        # Raise it so the compiled mode keeps exercising the compiler.
        try:
            import torch._dynamo as _dynamo  # type: ignore
            _dynamo.config.recompile_limit = max(
                int(getattr(_dynamo.config, "recompile_limit", 8)), 256)
        except Exception:
            pass

    @classmethod
    def is_available(cls) -> Tuple[bool, str]:
        ok, reason = PyTorchAdapter.is_available()
        if not ok:
            return ok, reason
        if not hasattr(torch, "compile"):
            return False, "torch.compile is not present in this torch build"
        return True, f"torch {TORCH_VERSION} + torch.compile ({'cuda' if torch.cuda.is_available() else 'cpu'})"

    def materialize(self, graph: Graph, reuse: Any = None) -> TorchBundle:
        eager = IRModule(graph, self.bank)
        fingerprint = eager.fingerprint
        compiled = self._compile_cache.get(fingerprint)
        compile_error = ""
        if compiled is None:
            try:
                compiled = torch.compile(eager, backend=self.backend,
                                         fullgraph=self.fullgraph, dynamic=self.dynamic)
            except Exception as exc:  # compilation is best-effort per graph
                compiled = None
                compile_error = f"{type(exc).__name__}: {exc}"
            if compiled is not None:
                self._compile_cache[fingerprint] = compiled
                while len(self._compile_cache) > self.compile_cache_size:
                    self._compile_cache.pop(next(iter(self._compile_cache)))
        return TorchBundle(eager=eager, compiled=compiled, fingerprint=fingerprint,
                           compile_error=compile_error)

    def _run_once(self, bundle: TorchBundle, inputs, graph, recorder, capture_specs):
        # ``F(e)`` cannot be read out of a compiled graph, so the annotation pass
        # runs on the eager twin.  Both modules share the same parameter tensors,
        # hence the annotation describes exactly the tensors the compiled program
        # consumes and produces.
        if capture_specs:
            bundle.eager.tap_enabled = True
        before = _compile_metrics()
        target = bundle.compiled if bundle.compiled is not None else bundle.eager
        recorder.record("graph_capture.torch_compile_enter", "graph_capture")
        try:
            outputs = target(*inputs)
        finally:
            after = _compile_metrics()
            for label in _metric_events(before, after):
                recorder.record(label)
            recorder.record("graph_capture.torch_compile_exit", "graph_capture")
        if capture_specs and bundle.compiled is not None:
            # Populate the taps without affecting the compiled result.
            try:
                bundle.eager.tap_enabled = True
                with torch.no_grad():
                    bundle.eager(*inputs)
            except Exception:
                pass
        return outputs

    def _extras(self, bundle: TorchBundle) -> Dict[str, Any]:
        extras = super()._extras(bundle)
        extras.update({"backend": self.backend, "fullgraph": self.fullgraph,
                       "dynamic": self.dynamic,
                       "compiled": bundle.compiled is not None,
                       "compile_error": bundle.compile_error,
                       "compile_cache": len(self._compile_cache)})
        return extras
