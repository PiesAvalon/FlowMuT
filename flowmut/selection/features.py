"""Candidate context features and coverage bookkeeping.

This module implements the five context features of the paper's *Candidate
Selection* subsection, Section "Candidate Selection":

* three **coverage features** ``L(c)``, ``T(c)``, ``M(c)`` over the site
  context, the anchor tensor state and the mutation operator,
* two **impact features** ``P(c)`` (downstream impact) and ``I(c)`` (gradient
  influence).

For a feature set ``B`` the coverage value is

.. math::

    d(B) = \\frac{1}{|B|} \\sum_{b \\in B} \\frac{1}{\\sqrt{1 + n(b)}},

where ``n(b)`` is the number of previously executed candidates whose feature
set contained ``b``.  :class:`CoverageTracker` stores ``n`` and scores feature
sets; :class:`FeatureExtractor` turns a
:class:`~flowmut.operators.base.Candidate` into the five-dimensional context
vector the bandit policies consume.

Design notes
------------
* The module is deliberately framework agnostic: it never imports ``torch`` or
  ``mindspore`` and only touches the TFG's tensor *specifications*.
* The gradient influence ``I(c)`` needs autograd, which lives in the adapter
  layer.  Instead of importing it, the extractor accepts an optional
  ``gradient_probe`` callable.  When it is absent (or returns ``None``, raises,
  or produces a non-finite value) a deterministic, still informative fallback is
  used: ``1 / (1 + log1p(numel))`` -- i.e. larger tensors are treated as
  individually less influential per element, mirroring the paper's division by
  ``|a(c, x)|``.
* ``I(c)`` is min-max normalised to ``[0, 1]`` **within the current candidate
  set** inside :meth:`FeatureExtractor.context`, exactly as the paper prescribes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from flowmut.ir.ops import canonical_op

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the import light
    from flowmut.adapters.base import Sample
    from flowmut.operators.base import Candidate
    from flowmut.tfg.tfg import TFG

# ---------------------------------------------------------------------------
# Feature names and ablation configurations
# ---------------------------------------------------------------------------

#: The five context features, in the order used by every vector in this package.
FEATURE_NAMES: Tuple[str, ...] = ("site", "tensor", "operator", "downstream", "gradient")

#: Column of each feature inside a full 5-dimensional context vector.
FEATURE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}

#: Ablation configurations used by RQ4 (feature removal + coverage-only /
#: impact-only).  Each value lists the features kept, in ``FEATURE_NAMES`` order.
FEATURE_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "full": ("site", "tensor", "operator", "downstream", "gradient"),
    "without_site": ("tensor", "operator", "downstream", "gradient"),
    "without_tensor": ("site", "operator", "downstream", "gradient"),
    "without_operator": ("site", "tensor", "downstream", "gradient"),
    "without_downstream": ("site", "tensor", "operator", "gradient"),
    "without_gradient": ("site", "tensor", "operator", "downstream"),
    "coverage_only": ("site", "tensor", "operator"),
    "impact_only": ("downstream", "gradient"),
}

_ALL_COLUMNS: Tuple[int, ...] = tuple(range(len(FEATURE_NAMES)))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _dedupe(items: Sequence[str]) -> List[str]:
    """Order-preserving deduplication (a feature set is a *set*)."""
    seen = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _coverage_value(elements: Sequence[str], counts: Dict[str, int]) -> float:
    """``d(B)``: mean of ``1/sqrt(1 + n(b))`` over the elements of ``B``."""
    unique = _dedupe(list(elements))
    if not unique:
        return 0.0
    total = 0.0
    for element in unique:
        total += 1.0 / math.sqrt(1.0 + float(counts.get(element, 0)))
    return float(total / len(unique))


def _finite_or(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


# ---------------------------------------------------------------------------
# Coverage tracker
# ---------------------------------------------------------------------------

class CoverageTracker:
    """Counts ``n(b)`` per coverage element and scores feature sets.

    The tracker is intentionally tiny and serialisable: :meth:`state` returns a
    plain ``dict`` so a campaign can checkpoint and resume its coverage history.
    """

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}
        #: Optional hook installed by :class:`FeatureExtractor` so that
        #: ``observe(candidate)`` can derive the element sets without the caller
        #: having to pass them explicitly.
        self._element_fn: Optional[Callable[["Candidate"], Dict[str, List[str]]]] = None

    # -- basic counting ----------------------------------------------------
    @property
    def counts(self) -> Dict[str, int]:
        """A snapshot of ``n(b)`` for every observed element ``b``."""
        return dict(self._counts)

    def increment(self, elements: Sequence[str]) -> None:
        """Record that one candidate containing ``elements`` has been executed."""
        for element in _dedupe(list(elements)):
            self._counts[element] = self._counts.get(element, 0) + 1

    def value(self, elements: Sequence[str]) -> float:
        """The coverage value ``d(B)`` of a feature set."""
        return _coverage_value(elements, self._counts)

    # -- candidate-level bookkeeping --------------------------------------
    def element_sets(self, candidate: "Candidate") -> Dict[str, List[str]]:
        """Resolve the candidate's element sets.

        Uses the hook installed by :class:`FeatureExtractor` when present;
        otherwise falls back to a candidate-only approximation that needs no
        TFG (site identity and operator identity, but no tensor state).
        """
        if self._element_fn is not None:
            try:
                sets = self._element_fn(candidate)
                if sets:
                    return sets
            except Exception:  # pragma: no cover - defensive
                pass
        return _candidate_only_element_sets(candidate)

    def observe(self, candidate: "Candidate",
                element_sets: Optional[Dict[str, List[str]]] = None) -> None:
        """Update ``n(b)`` for an *executed* candidate.

        ``element_sets`` should be the dict returned by
        :meth:`FeatureExtractor.element_sets`; only the three coverage feature
        sets (``site``, ``tensor``, ``operator``) are counted -- the impact
        features are not coverage elements.
        """
        sets = element_sets if element_sets is not None else self.element_sets(candidate)
        for key in ("site", "tensor", "operator"):
            elements = sets.get(key) if isinstance(sets, dict) else None
            if elements:
                self.increment(elements)

    def reset(self) -> None:
        """Forget the whole coverage history."""
        self._counts.clear()

    def state(self) -> Dict[str, int]:
        """Checkpoint of the coverage counters."""
        return dict(self._counts)

    def load_state(self, state: Optional[Dict[str, int]]) -> None:
        """Restore a checkpoint produced by :meth:`state`."""
        self._counts = {str(k): int(v) for k, v in dict(state or {}).items()}

    def __len__(self) -> int:
        return len(self._counts)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CoverageTracker(elements={len(self._counts)})"


def _candidate_only_element_sets(candidate: "Candidate") -> Dict[str, List[str]]:
    """Fallback element sets that need no TFG (used by a bare tracker)."""
    site = getattr(candidate, "site", None)
    operator = getattr(candidate, "operator", None)
    site_elements: List[str] = []
    tensor_elements: List[str] = []
    if site is not None:
        site_elements.append(f"anchor:{getattr(site, 'consumer', '')}")
        for node_id in getattr(site, "nodes", []) or []:
            site_elements.append(f"op:{node_id}")
        for edge_id in list(getattr(site, "input_edges", []) or []) + \
                list(getattr(site, "output_edges", []) or []):
            site_elements.append(f"edge:{edge_id}")
        anchor = getattr(site, "anchor_edge", "")
        if anchor:
            tensor_elements.append(f"tensor:edge:{anchor}")
    return {
        "site": _dedupe(site_elements),
        "tensor": _dedupe(tensor_elements),
        "operator": _operator_elements(operator),
    }


def _operator_elements(operator: Any) -> List[str]:
    """The *mutation operator* feature set: name, ``(object, primitive)`` and params."""
    if operator is None:
        return []
    elements = [f"op:{getattr(operator, 'name', '?')}",
                f"cell:{getattr(operator, 'target_object', '?')}/{getattr(operator, 'primitive', '?')}"]
    params = getattr(operator, "params", None)
    if not isinstance(params, Mapping):
        # ``MutationOperator`` declares ``params`` as a dataclass ``field`` but
        # is not itself a dataclass, so an operator without explicit parameters
        # exposes the unbound ``Field`` object; treat anything non-mapping as
        # "no parameters".
        params = {}
    try:
        keys = sorted(params, key=lambda k: str(k))
    except Exception:  # pragma: no cover - exotic param keys
        keys = list(params)
    for key in keys:
        try:
            elements.append(f"param:{key}={params[key]!r}")
        except Exception:  # pragma: no cover - defensive
            elements.append(f"param:{key}")
    return _dedupe(elements)


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Turns mutation candidates into the five-dimensional context vector.

    Parameters
    ----------
    tfg:
        The profiled Tensor Flow Graph the candidates were built from.
    samples:
        The profiling samples.  Kept for provenance/reporting: the codebase
        profiles the graph with a single input set, so the reachability
        fractions agree with the paper's average over ``X_p``.
    tracker:
        Coverage history to use; a fresh :class:`CoverageTracker` otherwise.
    gradient_probe:
        Optional ``edge_id -> mean normalised absolute gradient`` callable.  It
        must never raise -- if it does, or returns ``None``/non-finite, the
        deterministic fallback ``1 / (1 + log1p(numel))`` is used instead.
    weights:
        Optional per-feature scaling applied to the context columns (default:
        all ones).  Accepts a 5-element sequence in ``FEATURE_NAMES`` order;
        other lengths are padded/truncated with ``1.0`` so it can never fail.
    """

    def __init__(self, tfg: "TFG", samples: Sequence["Sample"] = (),
                 tracker: Optional[CoverageTracker] = None,
                 gradient_probe: Optional[Callable[[str], Optional[float]]] = None,
                 weights: Optional[Sequence[float]] = None) -> None:
        self._tfg = tfg
        self.samples: List[Any] = list(samples or [])
        self._tracker = tracker if tracker is not None else CoverageTracker()
        self._probe = gradient_probe if callable(gradient_probe) else None
        self._weights = self._make_weights(weights)
        #: Columns of ``FEATURE_NAMES`` exposed by this extractor.
        self._columns: Tuple[int, ...] = _ALL_COLUMNS
        #: Cache of ``P(c)`` keyed by the candidate's output boundary.
        self._downstream_cache: Dict[Tuple[str, ...], float] = {}
        # Let a bare ``tracker.observe(candidate)`` derive the element sets.
        if getattr(self._tracker, "_element_fn", None) is None:
            self._tracker._element_fn = self.element_sets

    # -- configuration -----------------------------------------------------
    @staticmethod
    def _make_weights(weights: Optional[Sequence[float]]) -> np.ndarray:
        out = np.ones(len(FEATURE_NAMES), dtype=float)
        if weights is None:
            return out
        try:
            values = list(weights)
        except TypeError:
            return out
        for i in range(len(FEATURE_NAMES)):
            if i < len(values):
                w = _finite_or(values[i], 1.0)
                out[i] = w
        return out

    @property
    def tfg(self) -> "TFG":
        return self._tfg

    @property
    def tracker(self) -> CoverageTracker:
        return self._tracker

    @property
    def gradient_probe(self) -> Optional[Callable[[str], Optional[float]]]:
        return self._probe

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return tuple(FEATURE_NAMES[i] for i in self._columns)

    @property
    def output_dim(self) -> int:
        """Width of the vectors returned by :meth:`context`."""
        return len(self._columns)

    # -- coverage element sets --------------------------------------------
    def element_sets(self, candidate: "Candidate") -> Dict[str, List[str]]:
        """The three coverage feature sets of ``candidate``.

        Returns a dict with the keys ``"site"``, ``"tensor"`` and ``"operator"``
        (the order of :data:`FEATURE_NAMES`).
        """
        return {
            "site": self._site_elements(candidate),
            "tensor": self._tensor_elements(candidate),
            "operator": self._operator_elements(candidate),
        }

    # -- feature values ----------------------------------------------------
    def raw_vector(self, candidate: "Candidate") -> np.ndarray:
        """The five feature values, unnormalised, as a ``(5,)`` array.

        The coverage features are already in ``(0, 1]``; ``P(c)`` is already a
        fraction; the gradient value is whatever the probe returned (or the
        fallback) and is normalised later, per candidate set, by
        :meth:`context`.
        """
        elements = self.element_sets(candidate)
        values = np.empty(len(FEATURE_NAMES), dtype=float)
        values[FEATURE_INDEX["site"]] = self._tracker.value(elements["site"])
        values[FEATURE_INDEX["tensor"]] = self._tracker.value(elements["tensor"])
        values[FEATURE_INDEX["operator"]] = self._tracker.value(elements["operator"])
        values[FEATURE_INDEX["downstream"]] = self._downstream_impact(candidate)
        gradient, _source = self._gradient_influence(candidate)
        values[FEATURE_INDEX["gradient"]] = gradient
        return values * self._weights

    def context(self, candidates: Sequence["Candidate"]) -> np.ndarray:
        """The ``(n, 5)`` context matrix for ``candidates``.

        The gradient column is min-max normalised over this batch, which is the
        paper's "normalised to ``[0, 1]`` within the current candidate set".
        When every candidate has the same gradient value the column is set to a
        neutral ``0.5``.
        """
        n = len(candidates)
        if n == 0:
            return np.zeros((0, self.output_dim), dtype=float)
        matrix = np.vstack([self.raw_vector(c) for c in candidates])
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        gradient_col = FEATURE_INDEX["gradient"]
        if gradient_col in self._columns:
            column = matrix[:, gradient_col]
            lo = float(np.min(column))
            hi = float(np.max(column))
            if hi - lo > 1e-12:
                matrix[:, gradient_col] = (column - lo) / (hi - lo)
            else:
                matrix[:, gradient_col] = 0.5
        return matrix[:, list(self._columns)]

    def context_one(self, candidate: "Candidate") -> np.ndarray:
        """The context vector of a single candidate, shape ``(5,)``."""
        return self.context([candidate])[0]

    # -- bookkeeping -------------------------------------------------------
    def observe(self, candidate: "Candidate") -> None:
        """Record an executed candidate in the coverage history."""
        self._tracker.observe(candidate, self.element_sets(candidate))

    def reset(self) -> None:
        """Forget coverage history and cached reachability."""
        self._tracker.reset()
        self._downstream_cache.clear()

    def describe(self, candidate: "Candidate") -> Dict[str, Any]:
        """Human-readable feature report for logs and test-case summaries."""
        elements = self.element_sets(candidate)
        values = self.raw_vector(candidate)
        gradient, source = self._gradient_influence(candidate)
        num_nodes = self._graph_node_count()
        downstream_nodes = self._downstream_count(candidate)
        report: Dict[str, Any] = {
            name: float(values[FEATURE_INDEX[name]]) for name in FEATURE_NAMES
        }
        report.update({
            "anchor": getattr(getattr(candidate, "site", None), "anchor_edge", ""),
            "consumer": getattr(getattr(candidate, "site", None), "consumer", ""),
            "operator": getattr(getattr(candidate, "operator", None), "name", ""),
            "anchor_numel": self._anchor_numel(candidate),
            "downstream_nodes": downstream_nodes,
            "graph_nodes": num_nodes,
            "gradient_raw": float(gradient),
            "gradient_source": source,
            "coverage_elements": {k: list(v) for k, v in elements.items()},
        })
        return report

    # -- feature internals -------------------------------------------------
    def _site_elements(self, candidate: "Candidate") -> List[str]:
        """The *site context* feature set ``L(c)``.

        The anchor edge, the operators in the site and the tensor edges crossing
        the site, each rendered as a stable string.  Tensor values are
        deliberately excluded.
        """
        graph = self._tfg.graph
        site = getattr(candidate, "site", None)
        elements: List[str] = []
        anchor = getattr(site, "anchor_edge", "")
        producer_scope, _consumer_scope = self._edge_scopes(anchor)
        consumer = graph.nodes.get(getattr(site, "consumer", "")) if site is not None else None
        consumer_op = canonical_op(consumer.op) if consumer is not None else "?"
        elements.append(f"anchor:{producer_scope}->{consumer_op}")
        for node_id in getattr(site, "nodes", []) or []:
            node = graph.nodes.get(node_id)
            if node is not None:
                elements.append(f"op:{canonical_op(node.op)}")
        edges = list(getattr(site, "input_edges", []) or []) + \
            list(getattr(site, "output_edges", []) or [])
        for edge_id in edges:
            if edge_id and edge_id in graph.edges:
                p_scope, c_scope = self._edge_scopes(edge_id)
                elements.append(f"edge:{p_scope}->{c_scope}")
        return _dedupe(elements)

    def _tensor_elements(self, candidate: "Candidate") -> List[str]:
        """The *tensor state* feature set ``T(c)``.

        Uses :meth:`flowmut.ir.specs.TensorSpec.key` on the anchor edge: rank,
        dtype, device, layout, gradient state, aliasing, view flag and the value
        magnitude bin.  Tensor values are never included.
        """
        anchor = getattr(getattr(candidate, "site", None), "anchor_edge", "")
        spec = self._tfg.f(anchor) if anchor else None
        if spec is None:
            return ["tensor:none"]
        return [f"tensor:{spec.key()}"]

    def _operator_elements(self, candidate: "Candidate") -> List[str]:
        """The *mutation operator* feature set ``M(c)``."""
        return _operator_elements(getattr(candidate, "operator", None))

    def _edge_scopes(self, edge_id: str) -> Tuple[str, str]:
        """``(producer scope, consumer scope)`` of an edge, empty when unknown."""
        graph = self._tfg.graph
        edge = graph.edges.get(edge_id) if edge_id else None
        if edge is None:
            return ("", "")
        producer = graph.nodes.get(edge.producer)
        producer_scope = producer.scope if producer is not None else ""
        consumer_scope = ""
        for consumer_id, _pos in edge.consumers:
            node = graph.nodes.get(consumer_id)
            if node is None or node.is_param:
                continue
            consumer_scope = node.scope
            break
        return (producer_scope, consumer_scope)

    def _downstream_impact(self, candidate: "Candidate") -> float:
        """``P(c)``: fraction of the executed graph reachable from the site.

        ``|D(c)| / |V|`` over non-parameter nodes, cached per output boundary
        (candidates sharing a boundary share the reachability set, as the paper
        prescribes).
        """
        total = self._graph_node_count()
        if total <= 0:
            return 0.0
        boundary = self._boundary(candidate)
        if boundary not in self._downstream_cache:
            self._downstream_cache[boundary] = (
                float(self._downstream_count(candidate)) / float(total)
            )
        return self._downstream_cache[boundary]

    def _boundary(self, candidate: "Candidate") -> Tuple[str, ...]:
        site = getattr(candidate, "site", None)
        edges = list(getattr(site, "output_edges", []) or [])
        if not edges:
            anchor = getattr(site, "anchor_edge", "")
            edges = [anchor] if anchor else []
        return tuple(sorted(set(e for e in edges if e)))

    def _downstream_count(self, candidate: "Candidate") -> int:
        """``|D(c)|``: reachable non-parameter nodes from the site's boundary."""
        boundary = self._boundary(candidate)
        if not boundary:
            return 0
        try:
            reachable = self._tfg.downstream_of_edges(boundary)
        except Exception:  # pragma: no cover - defensive
            return 0
        graph = self._tfg.graph
        return sum(1 for node_id in reachable
                   if node_id in graph.nodes and not graph.nodes[node_id].is_param)

    def _graph_node_count(self) -> int:
        """``|V(c)|``: all non-parameter nodes of the graph (the executed set)."""
        return sum(1 for node in self._tfg.graph.nodes.values() if not node.is_param)

    def _anchor_numel(self, candidate: "Candidate") -> int:
        anchor = getattr(getattr(candidate, "site", None), "anchor_edge", "")
        spec = self._tfg.f(anchor) if anchor else None
        if spec is None:
            return 0
        numel = int(getattr(spec, "numel", 0) or 0)
        if numel > 0:
            return numel
        shape = tuple(getattr(spec, "shape", ()) or ())
        if not shape:
            return 1
        out = 1
        for dim in shape:
            out *= int(dim)
        return max(1, out)

    def _gradient_influence(self, candidate: "Candidate") -> Tuple[float, str]:
        """``I(c)`` before batch normalisation, and where it came from.

        The probe is asked first; any failure (exception, ``None`` or a
        non-finite value) silently falls back to ``1 / (1 + log1p(numel))``.
        """
        anchor = getattr(getattr(candidate, "site", None), "anchor_edge", "")
        if self._probe is not None and anchor:
            try:
                raw = self._probe(anchor)
            except Exception:
                raw = None
            if raw is not None:
                value = _finite_or(raw, math.nan)
                if math.isfinite(value):
                    return (abs(float(value)), "probe")
        return (self._fallback_gradient(candidate), "fallback")

    def _fallback_gradient(self, candidate: "Candidate") -> float:
        """Deterministic influence proxy used when no gradient is available.

        ``1 / (1 + log1p(numel))``: an unobserved tensor state is assumed to be
        moderately influential, decreasing with the number of elements (the
        paper's per-element normalisation) and never collapsing to ``0``.
        """
        numel = max(1, self._anchor_numel(candidate))
        return float(1.0 / (1.0 + math.log1p(float(numel))))


# ---------------------------------------------------------------------------
# RQ4 ablation extractor
# ---------------------------------------------------------------------------

class VariantFeatureExtractor(FeatureExtractor):
    """A :class:`FeatureExtractor` restricted to one :data:`FEATURE_VARIANTS` entry.

    :meth:`context` returns only the selected columns, in ``FEATURE_VARIANTS``
    order (which is always ``FEATURE_NAMES`` order).  Selection policies infer
    their feature dimension from the width of the returned matrix, so every RQ4
    ablation works with the same policy code.  An unknown variant name is not an
    error: it silently falls back to the full feature set, keeping the requested
    name available through :attr:`variant`.
    """

    def __init__(self, tfg: "TFG", variant: str = "full", **kwargs: Any) -> None:
        super().__init__(tfg, **kwargs)
        requested = str(variant)
        self._variant = requested
        names = FEATURE_VARIANTS.get(requested)
        if names is None:
            self._unknown_variant = True
            names = FEATURE_NAMES
        else:
            self._unknown_variant = False
        self._variant_names: Tuple[str, ...] = tuple(names)
        self._columns: Tuple[int, ...] = tuple(FEATURE_INDEX[n] for n in self._variant_names)
        # Variants with a different boundary need their own downstream cache key
        # only if the boundary set changes; the cache is boundary-keyed, so it
        # is safely shared.

    @property
    def variant(self) -> str:
        """The variant name this extractor was constructed with."""
        return self._variant

    @property
    def variant_names(self) -> Tuple[str, ...]:
        """The feature names actually kept (``FEATURE_NAMES`` order)."""
        return self._variant_names

    @property
    def unknown_variant(self) -> bool:
        """True when the requested variant was unknown and ``full`` was used."""
        return self._unknown_variant

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"VariantFeatureExtractor(variant={self._variant!r}, dim={self.output_dim})"


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_INDEX",
    "FEATURE_VARIANTS",
    "CoverageTracker",
    "FeatureExtractor",
    "VariantFeatureExtractor",
]
