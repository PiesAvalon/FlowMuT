"""The FlowMuT main loop (Algorithm 1).

    FlowMuT(f, X_p, M, N)
      E <- ProfileExecution(f, X_p)                       # 1. profile execution
      G <- BuildTFG(E); H, B <- empty                     # 2. build the profiled TFG
      for t = 1 to N do
        C <- GenerateCandidates(G, M, X_p)
        if C = empty then break
        c <- SelectWithLinUCB(C, G, H)                    # 3. generate + select
        f' <- Apply(f, c); z <- ExecuteAndObserve(f', X_p) # 4. detect + iterate
        H <- UpdateLinUCB(H, c, z)
        if z reports a bug then B <- B + {z}
        else if SuccessfulExecution(z) then
          f <- f'; G <- BuildTFG(ProfileExecution(f, X_p))
      return B

The engine keeps one framework adapter for the campaign and, when the mode is a
compiled/graph mode, a second adapter in the semantics-equivalent eager/PyNative
mode so that the oracle can compare the two executions.
"""

from __future__ import annotations

import random
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from flowmut.adapters.base import ExecutionResult, FrameworkAdapter, Sample
from flowmut.adapters.registry import make_adapter
from flowmut.config import FlowMuTConfig
from flowmut.ir.graph import Graph
from flowmut.ir.specs import parse_mode
from flowmut.operators.base import Candidate, OperatorPool
from flowmut.operators.constraints import (
    CONSTRAINT_KINDS,
    ConstraintChecker,
    ConstraintReport,
    filter_candidates,
)
from flowmut.operators.pool import build_pool, generate_candidates
from flowmut.oracle.oracle import BugOracle, BugReport
from flowmut.seed_models.registry import SeedModel, default_samples, seed as get_seed
from flowmut.selection.features import (
    FEATURE_NAMES,
    CoverageTracker,
    FeatureExtractor,
    VariantFeatureExtractor,
)
from flowmut.selection.policies import make_policy
from flowmut.tfg.tfg import TFG, build_tfg
from flowmut.trace.reward import TraceMemory


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """Everything observable about one mutation round."""

    iteration: int = 0
    candidates_total: int = 0
    #: Candidates actually handed to the constraint filter (after sampling).
    candidates_filtered: int = 0
    candidates_retained: int = 0
    rejected_by_kind: Dict[str, int] = field(default_factory=dict)
    operator: str = ""
    target_object: str = ""
    primitive: str = ""
    source: str = ""
    anchor: str = ""
    consumer: str = ""
    scope: str = ""
    status: str = ""
    legal: bool = True
    is_bug: bool = False
    symptom: str = "none"
    reward: float = 0.0
    duration_s: float = 0.0
    graph_nodes: int = 0
    graph_edges: int = 0
    features: Dict[str, float] = field(default_factory=dict)
    seed_updated: bool = False
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CampaignResult:
    """The outcome of one FlowMuT campaign (one seed model, one mode)."""

    seed_model: str = ""
    task: str = ""
    framework: str = ""
    mode: str = ""
    rounds_requested: int = 0
    rounds_completed: int = 0
    iterations: List[IterationRecord] = field(default_factory=list)
    bugs: List[BugReport] = field(default_factory=list)
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    checkpoints: Dict[str, Any] = field(default_factory=dict)
    trace_memory: Dict[str, Any] = field(default_factory=dict)
    pool: Dict[str, Any] = field(default_factory=dict)
    candidate_statistics: Dict[str, Any] = field(default_factory=dict)
    legality: Dict[str, Any] = field(default_factory=dict)
    final_graph: Dict[str, Any] = field(default_factory=dict)
    #: Profiling inputs that the seed model could not execute (usually because a
    #: compact model fixes the batch dimension in a shape adapter).  Profiling
    #: continues with the inputs that do work.
    sample_failures: List[str] = field(default_factory=list)
    samples_used: List[int] = field(default_factory=list)
    final_graph_data: Optional[Dict[str, Any]] = None
    started_at: float = 0.0
    finished_at: float = 0.0
    duration_s: float = 0.0
    error: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def bugs_found(self) -> int:
        return len({b.key() for b in self.bugs})

    def to_dict(self, with_graph: bool = False) -> Dict[str, Any]:
        data = asdict(self)
        data["bugs"] = [b.to_dict() for b in self.bugs]
        data["bugs_found"] = self.bugs_found
        if not with_graph:
            data.pop("final_graph_data", None)
        return data

    def summary(self) -> str:
        lines = [
            f"FlowMuT campaign: {self.seed_model} [{self.framework}:{self.mode}]",
            f"  rounds: {self.rounds_completed}/{self.rounds_requested}"
            f"  duration: {self.duration_s:.1f}s",
            f"  bugs: {self.bugs_found}  (legal mutants failing: "
            f"{sum(1 for i in self.iterations if i.is_bug)})",
            f"  trace memory: {self.trace_memory.get('distinct_events', 0)} events, "
            f"{self.trace_memory.get('distinct_transitions', 0)} transitions, "
            f"mean reward {self.trace_memory.get('mean_reward', 0.0):.4f}",
            f"  final graph: {self.final_graph.get('nodes', 0)} nodes, "
            f"{self.final_graph.get('params', 0)} params",
        ]
        if self.checkpoints:
            last = max(self.checkpoints, key=lambda k: int(k))
            c = self.checkpoints[last]
            lines.append(f"  round {last}: LIC={c.get('lic', 0):.4f} LPC={c.get('lpc', 0):.4f} "
                         f"LSC={c.get('lsc', 0):.4f} DFSD={c.get('dfsd', 0):.4f} "
                         f"signatures={c.get('distinct_signatures', 0)}")
        if self.error:
            lines.append(f"  ERROR: {self.error}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class FlowMuTEngine:
    """One campaign: seed model x framework x mode."""

    def __init__(self, config: FlowMuTConfig, seed_model: str,
                 pool: Optional[OperatorPool] = None,
                 adapter: Optional[FrameworkAdapter] = None,
                 reference_adapter: Optional[FrameworkAdapter] = None):
        self.config = config
        self.seed_model_key = seed_model
        self.seed_model: Optional[SeedModel] = None
        self.pool = pool
        self.adapter = adapter
        self.reference_adapter = reference_adapter
        self.checker = ConstraintChecker.for_variant(config.constraint_variant)
        self.oracle = BugOracle(
            tolerance=config.tolerance,
            relative_tolerance=config.relative_tolerance,
            check_divergence=config.effective_reference_mode is not None,
            reproduce_attempts=config.reproduce_attempts,
        )
        self.trace_memory = TraceMemory()
        self.metrics = None  # set in setup()
        self.rng = random.Random(config.seed)

        # campaign state
        self.graph: Optional[Graph] = None
        self.tfg: Optional[TFG] = None
        self.samples: List[Sample] = []
        self.bundle: Any = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.policy = None
        self.iterations: List[IterationRecord] = []
        self.bugs: List[BugReport] = []
        self._rejection_counts: Dict[str, int] = {k: 0 for k in CONSTRAINT_KINDS}
        self._rejection_counts["structural"] = 0
        self._rejection_counts["pattern"] = 0
        self._operator_use: Dict[str, int] = {}
        self._pool_report: Dict[str, Any] = {}
        self._gradient_cache: Dict[str, float] = {}
        self._gradient_round = -1
        self._checkpoints: List[int] = []
        self._checkpoint_set: set = set()

    # -- setup -------------------------------------------------------------
    def setup(self, validate: bool = False) -> None:
        cfg = self.config
        self.seed_model = get_seed(self.seed_model_key)
        if self.pool is None:
            self.pool, report = build_pool(include_cg=cfg.include_cg,
                                           include_tfg=cfg.include_tfg,
                                           deduplicate=cfg.deduplicate)
            self._pool_report = report.to_dict()
            self._pool_report["summary"] = self.pool.summary()
        if self.adapter is None:
            self.adapter = make_adapter(cfg.framework, cfg.mode_name, device=cfg.device,
                                        seed=cfg.seed, timeout_s=cfg.timeout_s)
        ref = cfg.effective_reference_mode
        if ref and self.reference_adapter is None:
            fw, name = parse_mode(ref)
            try:
                self.reference_adapter = make_adapter(fw, name, device=cfg.device,
                                                      seed=cfg.seed, timeout_s=cfg.timeout_s)
            except Exception:
                self.reference_adapter = None

        self.samples = default_samples(self.seed_model, count=cfg.profile_samples,
                                       batch_sizes=cfg.sample_batch_sizes or None)
        self.graph = self.seed_model.fresh_graph()
        self._refresh_tfg(initial=True)
        self.policy = make_policy(cfg.policy, alpha=cfg.policy_alpha, seed=cfg.seed)

        from flowmut.analysis.metrics import MetricsRecorder
        # Checkpoints are the reported rounds; short runs snapshot their last round.
        self._checkpoints = cfg.checkpoint_set() or [cfg.rounds]
        self._checkpoint_set = set(self._checkpoints)
        self.metrics = MetricsRecorder(checkpoint_rounds=self._checkpoints)
        self.metrics.observe(self.tfg)

    def _refresh_tfg(self, initial: bool = False) -> None:
        """``G <- BuildTFG(ProfileExecution(f, X_p))``.

        A profiling input that the model cannot execute (some compact seed models
        fix the batch dimension in a shape adapter) is skipped rather than
        aborting the campaign; only a seed model that fails on *every* profiling
        input is a hard error.
        """
        cfg = self.config
        self.bundle = self.adapter.materialize(self.graph, reuse=self.bundle)
        edge_specs: Dict[str, List[Any]] = {}
        times: List[float] = []
        used: List[int] = []
        failures: List[str] = []
        for index, sample in enumerate(self.samples):
            result = self.adapter.execute(self.graph, sample, module=self.bundle,
                                          capture_specs=True)
            times.append(result.duration_s)
            if result.failed:
                failures.append(f"sample {index}: {result.error_type}: "
                                f"{result.error_message[:200]}")
                continue
            used.append(index)
            for eid, specs in result.edge_specs.items():
                edge_specs.setdefault(eid, []).extend(specs)
        if not used and failures and initial:
            raise RuntimeError(
                f"seed model {self.seed_model_key} failed to profile on "
                f"{self.adapter.key}: " + "; ".join(failures))
        self.profile_failures = failures
        self.profile_samples_used = used
        self.tfg = build_tfg(self.graph, edge_specs, profile_times=times,
                             meta={"seed": self.seed_model_key,
                                   "framework": cfg.framework, "mode": cfg.mode_name,
                                   "samples_used": used,
                                   "sample_failures": failures})
        tracker = self.feature_extractor.tracker if self.feature_extractor else CoverageTracker()
        extractor_cls = (VariantFeatureExtractor if cfg.feature_variant != "full"
                         else FeatureExtractor)
        if extractor_cls is VariantFeatureExtractor:
            self.feature_extractor = VariantFeatureExtractor(
                self.tfg, variant=cfg.feature_variant, samples=self.samples,
                tracker=tracker, gradient_probe=self._gradient_probe)
        else:
            self.feature_extractor = FeatureExtractor(
                self.tfg, samples=self.samples, tracker=tracker,
                gradient_probe=self._gradient_probe)

    # -- gradient influence ------------------------------------------------
    def _gradient_probe(self, edge_id: str) -> Optional[float]:
        if not self.config.gradient_probe or self.adapter is None:
            return None
        fn = getattr(self.adapter, "gradient_influence_map", None)
        if fn is None:
            return None
        if self._gradient_round != len(self.iterations):
            self._gradient_round = len(self.iterations)
            self._gradient_cache = {}
            sample = self.samples[0] if self.samples else None
            if sample is not None:
                try:
                    self._gradient_cache = fn(self.graph, sample, module=self.bundle) or {}
                except Exception:
                    self._gradient_cache = {}
        return self._gradient_cache.get(edge_id)

    # -- main loop ---------------------------------------------------------
    def run(self) -> CampaignResult:
        cfg = self.config
        result = CampaignResult(
            seed_model=self.seed_model_key,
            task=self.seed_model.task if self.seed_model else "",
            framework=cfg.framework, mode=cfg.mode_name,
            rounds_requested=cfg.rounds, config=cfg.to_dict(),
            started_at=time.time(),
        )
        if self.graph is None:
            try:
                self.setup()
            except Exception as exc:
                result.error = f"setup failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()[-4000:]}"
                result.finished_at = time.time()
                result.duration_s = result.finished_at - result.started_at
                return result
        result.pool = self._pool_report
        started = time.time()
        consecutive_failures = 0
        try:
            for t in range(1, cfg.rounds + 1):
                if cfg.time_budget_s and (time.time() - started) > cfg.time_budget_s:
                    result.error = result.error or "time budget exhausted"
                    break
                record = self.step(t)
                self.iterations.append(record)
                result.iterations.append(record)
                if record.status in ("error", "timeout"):
                    consecutive_failures += 1
                    if consecutive_failures >= cfg.max_consecutive_failures:
                        result.error = (f"aborted after {consecutive_failures} consecutive "
                                        f"execution failures")
                        break
                else:
                    consecutive_failures = 0
                if self.metrics is not None:
                    self.metrics.observe(self.tfg)
                    if t in self._checkpoint_set:
                        self.metrics.snapshot(t)
        except KeyboardInterrupt:
            result.error = result.error or "interrupted"
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-4000:]}"

        result.sample_failures = list(getattr(self, "profile_failures", []) or [])
        result.samples_used = list(getattr(self, "profile_samples_used", []) or [])
        result.rounds_completed = len(self.iterations)
        result.bugs = list(self.bugs)
        result.trace_memory = self.trace_memory.summary()
        result.candidate_statistics = {
            "total_generated": sum(i.candidates_total for i in self.iterations),
            "total_filtered": sum(i.candidates_filtered for i in self.iterations),
            "total_retained": sum(i.candidates_retained for i in self.iterations),
            "rejected_by_kind": dict(self._rejection_counts),
            "operator_usage": dict(sorted(self._operator_use.items(),
                                          key=lambda kv: -kv[1])[:40]),
            "distinct_operators_used": len(self._operator_use),
        }
        result.legality = self._legality_summary()
        if self.metrics is not None:
            result.metrics = [s.to_dict() for s in self.metrics.history()]
            result.checkpoints = {str(k): v for k, v in self.metrics.at_checkpoints().items()}
        if self.graph is not None:
            result.final_graph = self.graph.summary()
            if cfg.save_final_graph:
                result.final_graph_data = self._graph_payload()
        result.finished_at = time.time()
        result.duration_s = result.finished_at - result.started_at
        return result

    # -- one iteration -----------------------------------------------------
    def step(self, t: int) -> IterationRecord:
        cfg = self.config
        record = IterationRecord(iteration=t)
        graphs_nodes = len(self.graph.nodes)
        record.graph_nodes = graphs_nodes
        record.graph_edges = len(self.graph.edges)
        t0 = time.perf_counter()

        candidates = generate_candidates(self.tfg, self.pool, iteration=t,
                                        only_objects=cfg.only_objects or None,
                                        only_operators=cfg.only_operators or None)
        record.candidates_total = len(candidates)
        if cfg.max_candidates and len(candidates) > cfg.max_candidates:
            candidates = self.rng.sample(candidates, cfg.max_candidates)
        record.candidates_filtered = len(candidates)
        if not candidates:
            record.detail = "no candidates"
            record.duration_s = time.perf_counter() - t0
            return record

        retained, rejected = self._filter(candidates)
        record.candidates_retained = len(retained)
        record.rejected_by_kind = self._tally_rejections(rejected)
        if not retained:
            record.detail = "no retained candidates"
            record.duration_s = time.perf_counter() - t0
            return record

        index = self._select(retained)
        candidate = retained[index]
        contexts = getattr(self, "_last_contexts", None)
        if contexts is not None and index < len(contexts):
            record.features = {name: float(contexts[index][i])
                               for i, name in enumerate(self._feature_names())}
        record.operator = candidate.operator.name
        record.target_object = candidate.operator.target_object
        record.primitive = candidate.operator.primitive
        record.source = candidate.operator.source
        record.anchor = candidate.site.anchor_edge
        record.consumer = candidate.site.consumer
        record.scope = self.graph.nodes[candidate.site.consumer].scope
        self._operator_use[candidate.operator.name] = \
            self._operator_use.get(candidate.operator.name, 0) + 1

        report = self.checker.filter(self.tfg, candidate)
        if not report.accepted or report.mutated_graph is None:
            record.legal = False
            record.detail = f"selected candidate rejected: {report.error or report.first_violation_kind}"
            record.duration_s = time.perf_counter() - t0
            return record
        mutant = report.mutated_graph

        reference = self._reference_execution(mutant)
        execution = self._execute(mutant)
        observation = self.oracle.judge(report, execution, reference=reference,
                                        mutant_graph=mutant,
                                        seed_model=self.seed_model_key,
                                        operator=candidate.operator.name,
                                        anchor=candidate.site.anchor_edge)
        record.status = execution.status
        record.symptom = observation.symptom
        record.is_bug = observation.is_bug
        record.detail = observation.detail[:400]

        reward = 0.0
        if execution.status == "ok":
            reward = self.trace_memory.observe(execution)
        else:
            self.trace_memory.observe(execution)
        record.reward = reward
        self._update_policy(index, reward)

        if observation.is_bug:
            bug_report = self.oracle.to_report(
                observation, execution, seed_model=self.seed_model_key,
                operator=candidate.operator.name, anchor=candidate.site.anchor_edge,
                mutant_summary=mutant.summary())
            if bug_report is not None and self.oracle.reproduce(
                    lambda: self._execute(mutant), attempts=cfg.reproduce_attempts):
                self.bugs.append(bug_report)
        elif execution.status == "ok":
            self.graph = mutant
            self._refresh_tfg()
            record.seed_updated = True

        record.duration_s = time.perf_counter() - t0
        return record

    # -- helpers -----------------------------------------------------------
    def _feature_names(self) -> Tuple[str, ...]:
        from flowmut.selection.features import FEATURE_VARIANTS
        variant = self.config.feature_variant
        if variant in FEATURE_VARIANTS:
            return FEATURE_VARIANTS[variant]
        return FEATURE_NAMES

    def _filter(self, candidates: Sequence[Candidate]
                ) -> Tuple[List[Candidate], List[Tuple[Candidate, ConstraintReport]]]:
        retained: List[Candidate] = []
        rejected: List[Tuple[Candidate, ConstraintReport]] = []
        for candidate in candidates:
            try:
                report = self.checker.filter(self.tfg, candidate)
            except Exception as exc:
                report = ConstraintReport(accepted=False, error=f"{type(exc).__name__}: {exc}",
                                          first_violation_kind="structural")
            if report.accepted:
                retained.append(candidate)
            else:
                rejected.append((candidate, report))
        return retained, rejected

    def _tally_rejections(self, rejected: Sequence[Tuple[Candidate, ConstraintReport]]
                          ) -> Dict[str, int]:
        tally: Dict[str, int] = {}
        for _candidate, report in rejected:
            kind = report.first_violation_kind or "structural"
            tally[kind] = tally.get(kind, 0) + 1
            self._rejection_counts[kind] = self._rejection_counts.get(kind, 0) + 1
        return tally

    def _select(self, retained: Sequence[Candidate]) -> int:
        contexts = self.feature_extractor.context(retained)
        self._last_contexts = contexts
        index = self.policy.select(retained, contexts)
        if index < 0:
            index = 0
        return int(index)

    def _update_policy(self, index: int, reward: float) -> None:
        contexts = getattr(self, "_last_contexts", None)
        if contexts is None or index >= len(contexts):
            return
        try:
            self.policy.update(index, contexts[index], float(reward))
        except Exception:
            pass

    def _execute(self, graph: Graph) -> ExecutionResult:
        try:
            module = self.adapter.materialize(graph, reuse=self.bundle)
            sample = self.samples[0] if self.samples else {}
            return self.adapter.execute(graph, sample, module=module, capture_specs=False)
        except Exception as exc:
            return ExecutionResult(status="error", framework=self.config.framework,
                                   mode=self.config.mode_name,
                                   error_type=type(exc).__name__,
                                   error_message=str(exc)[:2000],
                                   error_traceback=traceback.format_exc()[-4000:])

    def _reference_execution(self, graph: Graph) -> Optional[ExecutionResult]:
        if self.reference_adapter is None:
            return None
        try:
            module = self.reference_adapter.materialize(graph)
            sample = self.samples[0] if self.samples else {}
            return self.reference_adapter.execute(graph, sample, module=module,
                                                   capture_specs=False)
        except Exception:
            return None

    def _legality_summary(self) -> Dict[str, Any]:
        """``LMR`` and filter selectivity of the (sampled) candidate set.

        The paper defines the primary measure as the fraction of *retained*
        candidates that satisfy the legality definition, so retention is computed
        against the candidates the static filter actually saw.
        """
        total_retained = sum(i.candidates_retained for i in self.iterations)
        total_filtered = sum(i.candidates_filtered for i in self.iterations)
        total_generated = sum(i.candidates_total for i in self.iterations)
        return {
            "generated": total_generated,
            "filtered": total_filtered,
            "retained": total_retained,
            "retention_rate": (total_retained / total_filtered) if total_filtered else 0.0,
            "coverage_of_generated": (total_filtered / total_generated)
            if total_generated else 0.0,
            "filter_selectivity": (1.0 - total_retained / total_filtered)
            if total_filtered else 0.0,
            "rejected_by_kind": dict(self._rejection_counts),
            "constraint_variant": self.config.constraint_variant,
            "legal_mutant_rate": 1.0 if total_retained else 0.0,
        }

    def _graph_payload(self) -> Dict[str, Any]:
        return {
            "name": self.graph.name,
            "seed_model": self.seed_model_key,
            "nodes": [n.to_dict() for n in self.graph.nodes.values()],
            "edges": [e.to_dict() for e in self.graph.edges.values()],
            "inputs": list(self.graph.inputs),
            "outputs": list(self.graph.outputs),
        }
