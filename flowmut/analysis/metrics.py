"""Diversity and coverage metrics for FlowMuT (RQ2).

Four metrics are computed from a TFG -- the profiled execution of the current
seed model -- against a *test space* derived from the union of the TFGs seen so
far:

``LIC``   Layer Input Coverage: covered input datatypes, dimensionalities and
          shapes.  The value is **macro-averaged over operator types**.
``LPC``   Layer Parameter Coverage: covered parameter values.
``LSC``   Layer Sequence Coverage: covered valid producer-consumer operator
          pairs.
``DFSD``  Data Flow Signature Diversity: covered data flow signatures.

The "space" is the union of the corresponding element sets observed across all
rounds, so it is monotonic in the number of rounds.  :class:`MetricsRecorder`
accumulates those sets (and only those sets) so the accumulation stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

from flowmut.ir.ops import canonical_op

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from flowmut.tfg.tfg import TFG

__all__ = [
    "CoverageSnapshot",
    "MetricsRecorder",
    "layer_input_coverage",
    "layer_parameter_coverage",
    "layer_sequence_coverage",
    "data_flow_signature_diversity",
    "distinct_signatures",
]

#: Structural IO ops that are not treated as "operators" for the diagnostics.
_IO_OPS: Set[str] = {"input", "param", "output"}

#: The four element-set families the recorder accumulates.
_KINDS: Tuple[str, ...] = ("lic", "lpc", "lsc", "dfsd")


# ---------------------------------------------------------------------------
# Element extraction
# ---------------------------------------------------------------------------

def _node_spec(tfg: "TFG", edge_id: str):
    try:
        return tfg.f(edge_id)
    except Exception:
        return None


def _input_elements(tfg: "TFG") -> Set[Any]:
    """LIC elements: ``(consumer_op, rank, shape[1:], dtype)`` per data input."""
    elements: Set[Any] = set()
    try:
        nodes = tfg.graph.nodes
        edges = tfg.graph.edges
    except Exception:
        return elements
    for node in list(nodes.values()):
        try:
            op = canonical_op(node.op)
        except Exception:
            op = node.op
        for edge_id in list(node.inputs):
            if not edge_id:
                continue
            edge = edges.get(edge_id)
            if edge is None:
                continue
            producer = nodes.get(edge.producer)
            if producer is not None and producer.is_param:
                # parameter inputs are measured by LPC instead
                continue
            spec = _node_spec(tfg, edge_id)
            if spec is None:
                continue
            try:
                shape = tuple(spec.shape)
                element = (op, int(spec.rank), tuple(shape[1:]), spec.dtype)
            except Exception:
                continue
            elements.add(element)
    return elements


def _parameter_elements(tfg: "TFG") -> Set[Any]:
    """LPC elements: ``(scope, op, shape, dtype, magnitude_bin)`` per param."""
    elements: Set[Any] = set()
    try:
        nodes = tfg.graph.nodes
    except Exception:
        return elements
    for node_id, node in list(nodes.items()):
        if not getattr(node, "is_param", False):
            continue
        spec = None
        try:
            out_edges = tfg.graph.out_edges(node_id)
        except Exception:
            out_edges = []
        for edge_id in out_edges:
            spec = _node_spec(tfg, edge_id)
            if spec is not None:
                break
        if spec is None:
            continue
        try:
            value = spec.value
            magnitude_bin = None if value is None else int(value.magnitude_bin)
            element = (node.scope, canonical_op(node.op), tuple(spec.shape),
                       spec.dtype, magnitude_bin)
        except Exception:
            continue
        elements.add(element)
    return elements


def _sequence_elements(tfg: "TFG") -> Set[Any]:
    """LSC elements: ``(producer_op, consumer_op)`` for data flow pairs."""
    elements: Set[Any] = set()
    try:
        nodes = tfg.graph.nodes
        edges = tfg.graph.edges
    except Exception:
        return elements
    for edge in list(edges.values()):
        producer = nodes.get(edge.producer)
        if producer is None or getattr(producer, "is_param", False):
            continue
        try:
            producer_op = canonical_op(producer.op)
        except Exception:
            producer_op = producer.op
        for consumer_id, _position in list(edge.consumers):
            consumer = nodes.get(consumer_id)
            if consumer is None or getattr(consumer, "is_param", False):
                continue
            try:
                consumer_op = canonical_op(consumer.op)
            except Exception:
                consumer_op = consumer.op
            elements.add((producer_op, consumer_op))
    return elements


def distinct_signatures(tfg: "TFG") -> Set[Any]:
    """The set of data flow signatures observed in ``tfg`` (DFSD elements)."""
    try:
        return set(tfg.all_signatures())
    except Exception:
        return set()


def _signature_elements(tfg: "TFG") -> Set[Any]:
    return distinct_signatures(tfg)


def _operators_seen(tfg: "TFG") -> Set[str]:
    """Distinct non-structural operator types exercised by ``tfg``."""
    seen: Set[str] = set()
    try:
        nodes = tfg.graph.nodes
    except Exception:
        return seen
    for node in list(nodes.values()):
        try:
            op = canonical_op(node.op)
        except Exception:
            op = node.op
        if op in _IO_OPS:
            continue
        seen.add(op)
    return seen


# ---------------------------------------------------------------------------
# Metric computations
# ---------------------------------------------------------------------------

def _elements_or_empty(func, tfg: "TFG") -> Set[Any]:
    try:
        return set(func(tfg))
    except Exception:
        return set()


def _ratio(covered: int, total: int) -> float:
    if total <= 0:
        return 0.0
    value = covered / total
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def layer_input_coverage(tfg: "TFG",
                         space: Optional[Set[Any]] = None
                         ) -> Tuple[float, int, int]:
    """Macro-averaged LIC plus ``(covered, space)`` element counts.

    The coverage ratio is computed per operator type and then averaged over the
    operator types present in ``tfg``; the returned counts are the global
    covered/space element counts.
    """
    elements = _elements_or_empty(_input_elements, tfg)
    universe = set(space) if space is not None else set(elements)
    if not elements:
        return 0.0, 0, len(universe)

    by_op: Dict[Any, Set[Any]] = {}
    for element in elements:
        by_op.setdefault(element[0], set()).add(element)
    space_by_op: Dict[Any, Set[Any]] = {}
    for element in universe:
        try:
            key = element[0]
        except Exception:
            continue
        space_by_op.setdefault(key, set()).add(element)

    ratios: List[float] = []
    for op, op_elements in by_op.items():
        op_space = space_by_op.get(op)
        if not op_space:
            continue
        ratios.append(len(op_elements & op_space) / len(op_space))
    value = sum(ratios) / len(ratios) if ratios else 0.0
    covered = len(elements & universe)
    return float(min(1.0, max(0.0, value))), covered, len(universe)


def layer_parameter_coverage(tfg: "TFG",
                             space: Optional[Set[Any]] = None
                             ) -> Tuple[float, int, int]:
    """LPC value plus ``(covered, space)`` parameter-value counts."""
    elements = _elements_or_empty(_parameter_elements, tfg)
    universe = set(space) if space is not None else set(elements)
    covered = len(elements & universe)
    return _ratio(covered, len(universe)), covered, len(universe)


def layer_sequence_coverage(tfg: "TFG",
                            space: Optional[Set[Any]] = None
                            ) -> Tuple[float, int, int]:
    """LSC value plus ``(covered, space)`` valid-pair counts."""
    elements = _elements_or_empty(_sequence_elements, tfg)
    universe = set(space) if space is not None else set(elements)
    covered = len(elements & universe)
    return _ratio(covered, len(universe)), covered, len(universe)


def data_flow_signature_diversity(tfg: "TFG",
                                  space: Optional[Set[Any]] = None
                                  ) -> Tuple[float, int, int]:
    """DFSD value plus ``(distinct_signatures, signatures_in_space)`` counts."""
    signatures = _elements_or_empty(_signature_elements, tfg)
    universe = set(space) if space is not None else set(signatures)
    covered = len(signatures & universe)
    return _ratio(covered, len(universe)), len(signatures), len(universe)


# ---------------------------------------------------------------------------
# Snapshot and recorder
# ---------------------------------------------------------------------------

@dataclass
class CoverageSnapshot:
    """The four RQ2 metrics at one round, plus their raw counts."""

    round: int = 0
    lic: float = 0.0        # Layer Input Coverage
    lpc: float = 0.0        # Layer Parameter Coverage
    lsc: float = 0.0        # Layer Sequence Coverage
    dfsd: float = 0.0       # Data Flow Signature Diversity
    distinct_signatures: int = 0
    signatures_in_space: int = 0
    covered_input_values: int = 0
    input_values_in_space: int = 0
    covered_parameter_values: int = 0
    parameter_values_in_space: int = 0
    covered_valid_pairs: int = 0
    valid_pairs: int = 0
    operators_seen: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": int(self.round),
            "lic": float(self.lic),
            "lpc": float(self.lpc),
            "lsc": float(self.lsc),
            "dfsd": float(self.dfsd),
            "distinct_signatures": int(self.distinct_signatures),
            "signatures_in_space": int(self.signatures_in_space),
            "covered_input_values": int(self.covered_input_values),
            "input_values_in_space": int(self.input_values_in_space),
            "covered_parameter_values": int(self.covered_parameter_values),
            "parameter_values_in_space": int(self.parameter_values_in_space),
            "covered_valid_pairs": int(self.covered_valid_pairs),
            "valid_pairs": int(self.valid_pairs),
            "operators_seen": int(self.operators_seen),
        }

    # Convenience accessors used by the paper tables.
    @property
    def signature_count(self) -> int:
        return int(self.distinct_signatures)

    @property
    def metrics(self) -> Dict[str, float]:
        return {"lic": float(self.lic), "lpc": float(self.lpc),
                "lsc": float(self.lsc), "dfsd": float(self.dfsd)}


class MetricsRecorder:
    """Accumulate the observed test space across rounds and snapshot metrics."""

    def __init__(self, checkpoint_rounds: Sequence[int] = (50, 100, 150, 200, 250)
                 ) -> None:
        try:
            self.checkpoint_rounds: Tuple[int, ...] = tuple(
                int(r) for r in checkpoint_rounds)
        except Exception:
            self.checkpoint_rounds = (50, 100, 150, 200, 250)
        self._space: Dict[str, Set[Any]] = {kind: set() for kind in _KINDS}
        self._last: Optional["TFG"] = None
        self._history: List[CoverageSnapshot] = []
        self._by_round: Dict[int, CoverageSnapshot] = {}

    # -- accumulation ------------------------------------------------------
    def observe(self, tfg: "TFG") -> None:
        """Fold one round's TFG into the accumulated test space.

        Only the element sets are retained; the TFG itself is not kept (beyond
        the most recent reference used by :meth:`snapshot`).
        """
        if tfg is None:
            return
        self._last = tfg
        try:
            self._space["lic"].update(_input_elements(tfg))
            self._space["lpc"].update(_parameter_elements(tfg))
            self._space["lsc"].update(_sequence_elements(tfg))
            self._space["dfsd"].update(_signature_elements(tfg))
        except Exception:
            pass

    # -- metric computation ------------------------------------------------
    def snapshot(self, round_index: int) -> CoverageSnapshot:
        """Compute the four metrics at ``round_index`` against the space seen."""
        try:
            round_value = int(round_index)
        except Exception:
            round_value = 0
        tfg = self._last
        if tfg is None:
            snapshot = CoverageSnapshot(round=round_value)
        else:
            lic, lic_covered, lic_space = layer_input_coverage(
                tfg, self._space["lic"])
            lpc, lpc_covered, lpc_space = layer_parameter_coverage(
                tfg, self._space["lpc"])
            lsc, lsc_covered, lsc_space = layer_sequence_coverage(
                tfg, self._space["lsc"])
            dfsd, distinct, dfsd_space = data_flow_signature_diversity(
                tfg, self._space["dfsd"])
            snapshot = CoverageSnapshot(
                round=round_value,
                lic=lic, lpc=lpc, lsc=lsc, dfsd=dfsd,
                distinct_signatures=distinct,
                signatures_in_space=dfsd_space,
                covered_input_values=lic_covered,
                input_values_in_space=lic_space,
                covered_parameter_values=lpc_covered,
                parameter_values_in_space=lpc_space,
                covered_valid_pairs=lsc_covered,
                valid_pairs=lsc_space,
                operators_seen=len(_operators_seen(tfg)),
            )
        self._history.append(snapshot)
        self._by_round[round_value] = snapshot
        return snapshot

    # -- reporting ---------------------------------------------------------
    def history(self) -> List[CoverageSnapshot]:
        """All snapshots taken so far, in order."""
        return list(self._history)

    def at_checkpoints(self) -> Dict[int, Dict[str, Any]]:
        """The snapshot nearest to (and not after) each checkpoint round."""
        out: Dict[int, Dict[str, Any]] = {}
        for checkpoint in self.checkpoint_rounds:
            snapshot = self._by_round.get(int(checkpoint))
            if snapshot is None:
                candidates = [s for r, s in self._by_round.items()
                              if r <= int(checkpoint)]
                snapshot = max(candidates, key=lambda s: s.round) \
                    if candidates else None
            if snapshot is not None:
                out[int(checkpoint)] = snapshot.to_dict()
        return out

    def space_sizes(self) -> Dict[str, int]:
        """Current accumulated test-space sizes, per element family."""
        return {kind: len(self._space[kind]) for kind in _KINDS}

    def summary(self) -> Dict[str, Any]:
        """Aggregate view of the recorder's history and accumulated space."""
        final = self._history[-1] if self._history else None
        rounds = len(self._history)
        mean: Dict[str, float] = {}
        for name in ("lic", "lpc", "lsc", "dfsd"):
            mean[name] = (sum(getattr(s, name) for s in self._history) / rounds
                          if rounds else 0.0)
        mean["distinct_signatures"] = (
            sum(s.distinct_signatures for s in self._history) / rounds
            if rounds else 0.0)
        return {
            "rounds": rounds,
            "checkpoint_rounds": list(self.checkpoint_rounds),
            "checkpoints": self.at_checkpoints(),
            "final": final.to_dict() if final is not None else None,
            "mean": mean,
            "test_space": {
                "input_values": len(self._space["lic"]),
                "parameter_values": len(self._space["lpc"]),
                "valid_pairs": len(self._space["lsc"]),
                "signatures": len(self._space["dfsd"]),
            },
        }
