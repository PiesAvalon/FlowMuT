"""Candidate selection policies for the FlowMuT main loop.

The paper models candidate selection as a *contextual bandit*: the value of a
candidate is observable only after the mutant is executed, and each accepted
mutation changes the TFG and therefore the candidate set.  This module provides
the FlowMuT policy :class:`LinUCB` plus the baselines used by RQ4.

All policies share the :class:`SelectionPolicy` interface:

.. code-block:: python

    policy = make_policy("linucb", alpha=1.0, seed=0)
    index = policy.select(candidates, contexts)   # -1 when there is no candidate
    policy.update(index, contexts[index], reward)

Robustness contract
-------------------
* An empty ``candidates`` sequence always yields ``-1``.
* ``contexts`` may have any width; policies infer their feature dimension from
  ``contexts.shape[1]`` so the RQ4 feature-removal variants (3 or 4 columns)
  work unchanged.
* Ties are always broken towards the lowest candidate index, so runs are
  reproducible.
"""

from __future__ import annotations

import abc
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from flowmut.operators.base import Candidate

__all__ = [
    "SelectionPolicy",
    "LinUCB",
    "UCB1",
    "RandomSearch",
    "FixedPriority",
    "OneShot",
    "CoverageGreedy",
    "ImpactGreedy",
    "POLICIES",
    "make_policy",
]

#: Small ridge added to every normal equation before solving, for numerical
#: stability when a feature has not been observed yet.
_DEFAULT_RIDGE = 1e-6


def _finite_or(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class SelectionPolicy(abc.ABC):
    """Abstract candidate-selection policy."""

    name: str = "base"

    def __init__(self, feature_dim: int = 5, seed: int = 0, **kwargs: Any) -> None:
        try:
            dim = int(feature_dim)
        except (TypeError, ValueError):
            dim = 5
        #: Expected context width.  Policies may re-infer this from ``contexts``.
        self.feature_dim: int = max(1, dim) if dim > 0 else 5
        self.seed: int = int(seed) if isinstance(seed, (int, np.integer)) else 0
        self.rng: np.random.Generator = np.random.default_rng(self.seed)
        #: Number of :meth:`update` calls seen so far.
        self.updates: int = 0
        self._last_n: int = 0
        self._selected_operator: Optional[Any] = None

    # -- interface ---------------------------------------------------------
    @abc.abstractmethod
    def select(self, candidates: Sequence["Candidate"], contexts: np.ndarray) -> int:
        """Return the index of the candidate to execute (``-1`` when empty)."""
        raise NotImplementedError

    def update(self, index: int, context: np.ndarray, reward: float) -> None:
        """Incorporate the observed reward of the candidate chosen last."""
        self.updates += 1

    def state(self) -> Dict[str, Any]:
        """Serialisable summary used by the campaign bookkeeping."""
        return {
            "name": self.name,
            "feature_dim": int(self.feature_dim),
            "seed": int(self.seed),
            "updates": int(self.updates),
        }

    def reset(self) -> None:
        """Clear all learned statistics and restore the initial RNG state."""
        self.rng = np.random.default_rng(self.seed)
        self.updates = 0
        self._last_n = 0
        self._selected_operator = None

    # -- shared helpers ----------------------------------------------------
    def _ensure_dim(self, dim: int) -> None:
        """Adopt a new context width (the default policy is stateless)."""
        self.feature_dim = max(1, int(dim))

    def _matrix(self, contexts: Any, n: int) -> np.ndarray:
        """Coerce ``contexts`` into an ``(n, feature_dim)`` float matrix.

        The feature dimension is inferred from ``contexts.shape[1]`` whenever
        that is available, which is what makes the RQ4 ablations work with a
        policy constructed for the full five-feature context.
        """
        array: Optional[np.ndarray] = None
        if contexts is not None:
            try:
                array = np.asarray(contexts, dtype=float)
            except Exception:  # pragma: no cover - defensive
                array = None
        if array is None or array.ndim == 0 or array.size == 0:
            dim = int(self.feature_dim)
            return np.zeros((n, dim), dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        dim = int(array.shape[1]) if array.shape[1] > 0 else int(self.feature_dim)
        if dim <= 0:  # pragma: no cover - defensive
            dim = int(self.feature_dim)
        self._ensure_dim(dim)
        if array.shape[1] != dim:
            if array.shape[1] > dim:
                array = array[:, :dim]
            else:
                array = np.hstack([array, np.zeros((array.shape[0], dim - array.shape[1]))])
        if array.shape[0] < n:
            array = np.vstack([array, np.zeros((n - array.shape[0], dim))])
        elif array.shape[0] > n:
            array = array[:n]
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    def _remember(self, candidate: Any) -> None:
        self._selected_operator = getattr(candidate, "operator", None)

    @staticmethod
    def _argmax(scores: np.ndarray) -> int:
        """Deterministic argmax: the *first* maximal element wins."""
        return int(np.argmax(scores))


# ---------------------------------------------------------------------------
# LinUCB -- the FlowMuT policy
# ---------------------------------------------------------------------------

class LinUCB(SelectionPolicy):
    """Hybrid LinUCB over the candidate context (the FlowMuT policy).

    A *shared* linear model over the five-feature context transfers feedback
    among candidates with similar contexts as their identities change, while a
    per-cell model keyed by ``(target_object, primitive)`` lets structurally
    similar operators share statistics.  Each candidate is scored

    .. math::

        \\mu_{shared} + \\mu_{cell}
        + \\alpha\\,(\\sqrt{u_{shared}} + \\sqrt{u_{cell}}),

    with ``mu = theta . x``, ``theta = solve(A, b)`` and ``u = x . solve(A, x)``.
    ``A`` starts at the identity and is regularised by a small ridge before
    solving; ``numpy.linalg.lstsq`` is used when the solve is singular.
    """

    name = "linucb"

    def __init__(self, feature_dim: int = 5, seed: int = 0, alpha: float = 1.0,
                 ridge: float = _DEFAULT_RIDGE, **kwargs: Any) -> None:
        super().__init__(feature_dim=feature_dim, seed=seed, **kwargs)
        self.alpha: float = _finite_or(alpha, 1.0)
        self.ridge: float = _finite_or(ridge, _DEFAULT_RIDGE)
        self._A_shared: np.ndarray = np.eye(self.feature_dim)
        self._b_shared: np.ndarray = np.zeros(self.feature_dim)
        self._A_cells: Dict[Any, np.ndarray] = {}
        self._b_cells: Dict[Any, np.ndarray] = {}
        self._shared_updates: int = 0
        self._selected_cell: Optional[Any] = None

    # -- model bookkeeping -------------------------------------------------
    def _init_models(self, dim: int) -> None:
        self._A_shared = np.eye(dim)
        self._b_shared = np.zeros(dim)
        self._A_cells = {}
        self._b_cells = {}
        self._shared_updates = 0
        self._selected_cell = None

    def _ensure_dim(self, dim: int) -> None:
        dim = max(1, int(dim))
        shared = getattr(self, "_A_shared", None)
        if shared is None or shared.shape[0] != dim:
            self.feature_dim = dim
            self._init_models(dim)
        else:
            self.feature_dim = dim

    @staticmethod
    def _cell_key(candidate: Any) -> Any:
        operator = getattr(candidate, "operator", None)
        if operator is None:
            return ("?", "?")
        return (str(getattr(operator, "target_object", "?")),
                str(getattr(operator, "primitive", "?")))

    def _solve(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        regularised = matrix + self.ridge * np.eye(matrix.shape[0])
        try:
            return np.linalg.solve(regularised, vector)
        except np.linalg.LinAlgError:
            solution, *_ = np.linalg.lstsq(regularised, vector, rcond=None)
            return solution

    def _estimate(self, matrix: np.ndarray, vector: np.ndarray,
                  context: np.ndarray) -> tuple:
        """``(mu, u)`` for one (A, b) pair and context."""
        theta = self._solve(matrix, vector)
        scaled = self._solve(matrix, context)
        mu = _finite_or(np.dot(theta, context), 0.0)
        u = max(0.0, _finite_or(np.dot(context, scaled), 0.0))
        return mu, u

    # -- selection ---------------------------------------------------------
    def select(self, candidates: Sequence["Candidate"], contexts: np.ndarray) -> int:
        n = len(candidates)
        if n == 0:
            self._last_n = 0
            self._selected_operator = None
            self._selected_cell = None
            return -1
        matrix = self._matrix(contexts, n)
        scores = np.empty(n, dtype=float)
        for i, candidate in enumerate(candidates):
            context = matrix[i]
            mu_shared, u_shared = self._estimate(
                self._A_shared, self._b_shared, context)
            key = self._cell_key(candidate)
            cell_a = self._A_cells.get(key)
            cell_b = self._b_cells.get(key)
            if cell_a is None or cell_b is None:
                mu_cell, u_cell = 0.0, max(0.0, _finite_or(np.dot(context, context), 0.0))
            else:
                mu_cell, u_cell = self._estimate(cell_a, cell_b, context)
            scores[i] = mu_shared + mu_cell + self.alpha * (
                math.sqrt(u_shared) + math.sqrt(u_cell))
        index = self._argmax(scores)
        self._last_n = n
        self._selected_operator = getattr(candidates[index], "operator", None)
        self._selected_cell = self._cell_key(candidates[index])
        return index

    def update(self, index: int, context: np.ndarray, reward: float) -> None:
        self.updates += 1
        try:
            vector = np.asarray(context, dtype=float).ravel()
        except Exception:  # pragma: no cover - defensive
            return
        if vector.size == 0:
            return
        self._ensure_dim(int(vector.size))
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        value = _finite_or(reward, 0.0)
        outer = np.outer(vector, vector)
        self._A_shared = self._A_shared + outer
        self._b_shared = self._b_shared + value * vector
        self._shared_updates += 1
        key = self._selected_cell
        if key is None:
            key = ("?", "?")
        if key not in self._A_cells:
            self._A_cells[key] = np.eye(vector.size)
            self._b_cells[key] = np.zeros(vector.size)
        elif self._A_cells[key].shape[0] != vector.size:
            self._A_cells[key] = np.eye(vector.size)
            self._b_cells[key] = np.zeros(vector.size)
        self._A_cells[key] = self._A_cells[key] + outer
        self._b_cells[key] = self._b_cells[key] + value * vector

    def state(self) -> Dict[str, Any]:
        state = super().state()
        state.update({
            "alpha": float(self.alpha),
            "ridge": float(self.ridge),
            "shared_updates": int(self._shared_updates),
            "cells": sorted("/".join(key) for key in self._A_cells),
        })
        return state

    def reset(self) -> None:
        super().reset()
        self._init_models(int(self.feature_dim))


# ---------------------------------------------------------------------------
# Classic UCB1 baseline (context-free)
# ---------------------------------------------------------------------------

class UCB1(SelectionPolicy):
    """Classic UCB1 over the candidate's operator as the arm.

    The context is ignored; this is the RQ4 baseline that isolates the value of
    contextual information.
    """

    name = "ucb1"

    def __init__(self, feature_dim: int = 5, seed: int = 0,
                 exploration: float = 2.0, **kwargs: Any) -> None:
        super().__init__(feature_dim=feature_dim, seed=seed, **kwargs)
        self.exploration: float = _finite_or(exploration, 2.0)
        self._counts: Dict[str, int] = {}
        self._sums: Dict[str, float] = {}
        self._selected_arm: Optional[str] = None

    @staticmethod
    def _arm(candidate: Any) -> str:
        operator = getattr(candidate, "operator", None)
        if operator is None:
            return "?"
        return str(getattr(operator, "name", "?"))

    def select(self, candidates: Sequence["Candidate"], contexts: np.ndarray) -> int:
        n = len(candidates)
        if n == 0:
            self._last_n = 0
            self._selected_arm = None
            self._selected_operator = None
            return -1
        self._matrix(contexts, n)
        total = sum(self._counts.values())
        log_total = math.log(max(1, total))
        scores = np.empty(n, dtype=float)
        for i, candidate in enumerate(candidates):
            arm = self._arm(candidate)
            count = self._counts.get(arm, 0)
            if count <= 0:
                scores[i] = float("inf")
            else:
                mean = self._sums.get(arm, 0.0) / float(count)
                scores[i] = mean + math.sqrt(self.exploration * log_total / float(count))
        index = self._argmax(scores)
        self._last_n = n
        self._selected_arm = self._arm(candidates[index])
        self._selected_operator = getattr(candidates[index], "operator", None)
        return index

    def update(self, index: int, context: np.ndarray, reward: float) -> None:
        self.updates += 1
        if self._selected_arm is None:
            return
        self._counts[self._selected_arm] = self._counts.get(self._selected_arm, 0) + 1
        self._sums[self._selected_arm] = (
            self._sums.get(self._selected_arm, 0.0) + _finite_or(reward, 0.0))

    def state(self) -> Dict[str, Any]:
        state = super().state()
        state.update({
            "exploration": float(self.exploration),
            "arms": sorted(self._counts),
            "arm_pulls": int(sum(self._counts.values())),
        })
        return state

    def reset(self) -> None:
        super().reset()
        self._counts = {}
        self._sums = {}
        self._selected_arm = None


# ---------------------------------------------------------------------------
# Random search baseline
# ---------------------------------------------------------------------------

class RandomSearch(SelectionPolicy):
    """Uniform random candidate selection using ``numpy.random.default_rng(seed)``."""

    name = "random"

    def select(self, candidates: Sequence["Candidate"], contexts: np.ndarray) -> int:
        n = len(candidates)
        if n == 0:
            self._last_n = 0
            self._selected_operator = None
            return -1
        self._matrix(contexts, n)
        index = int(self.rng.integers(0, n))
        self._last_n = n
        self._remember(candidates[index])
        return index


# ---------------------------------------------------------------------------
# Fixed-priority baseline
# ---------------------------------------------------------------------------

class FixedPriority(SelectionPolicy):
    """Always pick the first candidate, optionally reordered by ``priority``.

    ``priority`` is a list of operator names: candidates whose operator appears
    earlier in the list are preferred; everything else keeps its original order
    behind them.  The returned index always refers to the *original* candidate
    sequence.
    """

    name = "fixed"

    def __init__(self, feature_dim: int = 5, seed: int = 0,
                 priority: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        super().__init__(feature_dim=feature_dim, seed=seed, **kwargs)
        self.priority: List[str] = [str(p) for p in (priority or [])]
        self._rank: Dict[str, int] = {name: i for i, name in enumerate(self.priority)}

    def select(self, candidates: Sequence["Candidate"], contexts: np.ndarray) -> int:
        n = len(candidates)
        if n == 0:
            self._last_n = 0
            self._selected_operator = None
            return -1
        self._matrix(contexts, n)
        best_index = 0
        best_key = None
        fallback_rank = len(self._rank)
        for i, candidate in enumerate(candidates):
            operator = getattr(candidate, "operator", None)
            name = str(getattr(operator, "name", ""))
            key = (self._rank.get(name, fallback_rank), i)
            if best_key is None or key < best_key:
                best_key = key
                best_index = i
        self._last_n = n
        self._remember(candidates[best_index])
        return best_index

    def state(self) -> Dict[str, Any]:
        state = super().state()
        state["priority"] = list(self.priority)
        return state


# ---------------------------------------------------------------------------
# Non-adaptive fixed schedule
# ---------------------------------------------------------------------------

class OneShot(SelectionPolicy):
    """Pick every candidate index at most once, cycling when exhausted.

    A deterministic, non-adaptive schedule: index ``0``, then ``1``, ...  Once
    every index of the current candidate list has been used the schedule starts
    over.  The schedule is keyed by index, matching the paper's fixed ordering
    of the candidate set.
    """

    name = "one_shot"

    def __init__(self, feature_dim: int = 5, seed: int = 0, **kwargs: Any) -> None:
        super().__init__(feature_dim=feature_dim, seed=seed, **kwargs)
        self._chosen: set = set()

    def select(self, candidates: Sequence["Candidate"], contexts: np.ndarray) -> int:
        n = len(candidates)
        if n == 0:
            self._last_n = 0
            self._selected_operator = None
            return -1
        self._matrix(contexts, n)
        for i in range(n):
            if i not in self._chosen:
                self._chosen.add(i)
                self._last_n = n
                self._remember(candidates[i])
                return i
        self._chosen = {0}
        self._last_n = n
        self._remember(candidates[0])
        return 0

    def state(self) -> Dict[str, Any]:
        state = super().state()
        state["chosen"] = sorted(int(i) for i in self._chosen)
        return state

    def reset(self) -> None:
        super().reset()
        self._chosen = set()


# ---------------------------------------------------------------------------
# Coverage-only and impact-only greedy baselines
# ---------------------------------------------------------------------------

class _ColumnGreedy(SelectionPolicy):
    """Shared implementation for the two column-subset greedy baselines."""

    name = "greedy"
    _columns: slice = slice(0, 3)

    def select(self, candidates: Sequence["Candidate"], contexts: np.ndarray) -> int:
        n = len(candidates)
        if n == 0:
            self._last_n = 0
            self._selected_operator = None
            return -1
        matrix = self._matrix(contexts, n)
        subset = matrix[:, self._columns]
        if subset.size == 0:
            scores = np.zeros(n, dtype=float)
        else:
            scores = np.asarray(subset, dtype=float).sum(axis=1)
            if scores.size != n:  # pragma: no cover - defensive
                scores = np.zeros(n, dtype=float)
        index = self._argmax(scores)
        self._last_n = n
        self._remember(candidates[index])
        return index


class CoverageGreedy(_ColumnGreedy):
    """Pick the candidate whose coverage features ``(site, tensor, operator)``
    sum the highest."""

    name = "coverage_greedy"
    _columns = slice(0, 3)


class ImpactGreedy(_ColumnGreedy):
    """Pick the candidate whose impact features ``(downstream, gradient)`` sum
    the highest."""

    name = "impact_greedy"
    _columns = slice(3, None)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

POLICIES: Dict[str, type] = {
    "linucb": LinUCB,
    "ucb1": UCB1,
    "random": RandomSearch,
    "fixed": FixedPriority,
    "one_shot": OneShot,
    "coverage_greedy": CoverageGreedy,
    "impact_greedy": ImpactGreedy,
}

#: Forgiving aliases so CLI/JSON experiment configs need not be exact.
_POLICY_ALIASES: Dict[str, str] = {
    "lin_ucb": "linucb",
    "linear_ucb": "linucb",
    "hybrid_linucb": "linucb",
    "ucb": "ucb1",
    "ucb_1": "ucb1",
    "random_search": "random",
    "uniform": "random",
    "fixed_priority": "fixed",
    "priority": "fixed",
    "oneshot": "one_shot",
    "one-shot": "one_shot",
    "coverage": "coverage_greedy",
    "greedy_coverage": "coverage_greedy",
    "impact": "impact_greedy",
    "greedy_impact": "impact_greedy",
}


def make_policy(name: str, **kwargs: Any) -> SelectionPolicy:
    """Instantiate a policy by name (case- and separator-insensitive)."""
    key = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    key = _POLICY_ALIASES.get(key, key)
    cls = POLICIES.get(key)
    if cls is None:
        raise ValueError(
            f"unknown selection policy {name!r}; expected one of {sorted(POLICIES)}")
    return cls(**kwargs)  # type: ignore[return-value]
