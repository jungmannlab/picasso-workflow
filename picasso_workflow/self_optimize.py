#!/usr/bin/env python
"""Offline analysis-parameter self-optimization (WP-11).

An **offline, advisory** auto-tune stage. Given a *metric to maximize* (or an
A1/C19 lower-is-better metric to minimize), it fans out analysis runs over a
small set of **flagged-expensive** parameters, reads each run's metric back,
and proposes the best parameter set together with the full search provenance.
Like the recommender it **never mutates a running workflow** — it proposes a
spec for review *before* a run and records *why*.

Where the WP-10 recommender's :func:`recommend_parameters` gives the
**data-driven default** for every param (from the taxon node / learned cohort
ranges), this stage runs an **objective search** only for the params the caller
*flags* as worth the compute — "data-driven defaults first, objective search
only for flagged-expensive params". The start point of every search is seeded
from those data-driven defaults.

The design keeps the SEARCH logic pure and fully testable here, while the real
per-candidate evaluation (a cluster analysis run) stays an injected seam:

* ``evaluate_fn(params) -> metrics`` is the **injected evaluator**. In
  PRODUCTION it fans out one cluster analysis run per candidate (via the WP-9
  :mod:`picasso_workflow.cluster_submit` path / an injected runner) and reads
  the metrics back from the registry; in TESTS it is a synthetic objective.
  Whether the fan-out is parallel is a property of the *evaluator* (or the
  optional ``parallel_map`` hook) — :func:`optimize` stays agnostic, it just
  needs every candidate scored.
* ``registry`` is the injected provenance sink. The chosen params + the full
  search trace are written back **best-effort / non-blocking** (a registry
  outage never fails the optimize), mirroring WP-6/WP-10. picasso-registry is
  **not** a dependency; the client is imported lazily inside the write path and
  the module imports headlessly without it.

Reproducibility: every strategy is driven by the recorded ``seed``, so two
``optimize`` calls with the same seed produce identical proposals and winner,
and the seed is persisted in the provenance.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from loguru import logger

from picasso_workflow.recommender import _LOWER_IS_BETTER

#: Optimizer version tag persisted alongside every search. Bumped by hand when
#: the search/scoring logic changes in a way that should invalidate "this
#: parameter came from optimizer X" comparisons (the package version moves on
#: every commit via setuptools-scm, so a coarser, stable behaviour tag is the
#: provenance key). Mirrors ``recommender.RECOMMENDER_VERSION``.
OPTIMIZER_VERSION = "wp11-self-optimize-1"

#: A (module, param) address into a workflow's parameters — the key of a
#: :data:`ParamSpace`. ModuleSpec does NOT carry param bounds, so the searchable
#: space is supplied explicitly by the caller.
ParamKey = Tuple[str, str]

#: One param's search spec. ``type`` ∈ ``"float" | "int" | "choice"``.
#: float/int carry ``low``/``high`` (inclusive) and an optional grid ``step``;
#: choice carries ``choices``. ``default`` is the data-driven starting value
#: (from WP-10 / node_defaults); it is always a valid in-bounds candidate.
ParamSpecDict = Dict[str, Any]

#: The searchable space: an *ordered* mapping ``(module, param) -> spec``. Only
#: the params the caller puts here are searched; everything else keeps its
#: data-driven default. Insertion order fixes the grid's axis order (and the
#: coordinate-descent sweep order), so a fixed space + fixed seed is fully
#: deterministic.
ParamSpace = "Mapping[ParamKey, ParamSpecDict]"


# ---------------------------------------------------------------------------
# Result container.
# ---------------------------------------------------------------------------
@dataclass
class OptimizeResult:
    """The chosen parameters plus the full, auditable search provenance.

    ``best_params`` maps ``(module, param) -> value`` — the proposed override
    for each searched param (data-driven defaults for the rest are untouched and
    not repeated here). ``rationale`` is a human-facing summary; ``provenance``
    is the machine record persisted to the registry (strategy, seed, objective,
    every candidate + its metric, the winner, the held-out score).
    """

    best_params: Dict[ParamKey, Any]
    best_score: Optional[float]
    objective: str
    minimize: bool
    strategy: str
    seed: int
    n_evals: int
    optimizer_version: str = OPTIMIZER_VERSION
    holdout_score: Optional[float] = None
    rationale: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def params_as_str_keys(self) -> Dict[str, Any]:
        """Return ``best_params`` with ``"module.param"`` string keys.

        JSON/YAML (and the registry) cannot key on a tuple, so the serialized
        provenance uses dotted-string keys; this is that projection.
        """
        return {f"{m}.{p}": v for (m, p), v in self.best_params.items()}


# ---------------------------------------------------------------------------
# Direction (minimize vs maximize).
# ---------------------------------------------------------------------------
def infer_minimize(objective: str) -> bool:
    """True if ``objective`` is a lower-is-better metric (A1/C19).

    Reuses the recommender's :data:`~picasso_workflow.recommender._LOWER_IS_BETTER`
    table (NeNA / localization precision / drift residual / FRC), so the two
    stages agree on metric direction. Any metric not in that set is treated as
    higher-is-better (maximize). The caller can always override via
    ``optimize(..., minimize=...)``.
    """
    return objective in _LOWER_IS_BETTER


# ---------------------------------------------------------------------------
# Param-space validation + candidate generation (the bounds property lives here).
# ---------------------------------------------------------------------------
def _validate_space(param_space: "Mapping[ParamKey, ParamSpecDict]") -> None:
    """Raise ``ValueError`` if any param spec is malformed.

    A malformed space (bad type, missing bounds/choices, inverted range) is a
    programming error in the *caller*, not a runtime registry hiccup, so it is
    surfaced loudly here before any (expensive) evaluation runs.
    """
    if not param_space:
        raise ValueError("param_space is empty: nothing to optimize")
    for key, spec in param_space.items():
        if not (isinstance(key, tuple) and len(key) == 2):
            raise ValueError(f"param key must be (module, param): {key!r}")
        t = spec.get("type")
        if t in ("float", "int"):
            if "low" not in spec or "high" not in spec:
                raise ValueError(f"{key}: {t} spec needs 'low' and 'high'")
            if spec["low"] > spec["high"]:
                raise ValueError(f"{key}: low > high ({spec['low']!r})")
            step = spec.get("step")
            if step is not None and step <= 0:
                raise ValueError(
                    f"{key}: step must be > 0 (got {step!r}) -- a non-positive "
                    "step yields an empty grid axis (a silent no-op search)"
                )
        elif t == "choice":
            if not spec.get("choices"):
                raise ValueError(
                    f"{key}: choice spec needs non-empty 'choices'"
                )
        else:
            raise ValueError(f"{key}: unknown param type {t!r}")


def _default_of(spec: ParamSpecDict) -> Any:
    """The data-driven starting value for a param, clamped into its bounds.

    Prefers an explicit ``default`` (seeded from WP-10 / node_defaults); falls
    back to the low end (numeric) or the first choice. The result is always a
    valid in-bounds candidate — the search's seed point.
    """
    if "default" in spec and spec["default"] is not None:
        return _clamp(spec, spec["default"])
    if spec["type"] == "choice":
        return spec["choices"][0]
    return _clamp(spec, spec["low"])


def _clamp(spec: ParamSpecDict, value: Any) -> Any:
    """Project ``value`` into the spec's declared bounds / choice set.

    The single enforcement point for the **bounds property**: every candidate
    this module emits passes through here, so no proposed param can ever leave
    its declared range (numeric → clamped to ``[low, high]`` and re-typed to
    int when ``type == "int"``; choice → snapped to the nearest declared choice,
    defaulting to the first if the value is not a member).
    """
    t = spec["type"]
    if t == "choice":
        return value if value in spec["choices"] else spec["choices"][0]
    lo, hi = spec["low"], spec["high"]
    v = min(max(value, lo), hi)
    if t == "int":
        v = int(round(v))
        v = min(max(v, int(lo)), int(hi))
    return v


def _in_bounds(spec: ParamSpecDict, value: Any) -> bool:
    """True iff ``value`` lies within the spec's declared bounds / choices."""
    if spec["type"] == "choice":
        return value in spec["choices"]
    if not isinstance(value, (int, float)):
        return False
    # An int param must be integer-valued (a fractional "int" would reach
    # evaluate_fn unflagged otherwise).
    if spec["type"] == "int" and not float(value).is_integer():
        return False
    return spec["low"] <= value <= spec["high"]


def assert_in_bounds(
    param_space: "Mapping[ParamKey, ParamSpecDict]",
    candidate: "Mapping[ParamKey, Any]",
) -> None:
    """Assert every value in ``candidate`` is within its declared bounds.

    The runtime guard behind the bounds *property*: :func:`optimize` calls this
    on every candidate before it is evaluated, so an out-of-bounds param raises
    here (a bug in a strategy) rather than silently reaching ``evaluate_fn``.
    """
    for key, spec in param_space.items():
        if key not in candidate:
            raise AssertionError(f"candidate missing param {key!r}")
        if not _in_bounds(spec, candidate[key]):
            raise AssertionError(
                f"candidate {key!r}={candidate[key]!r} out of bounds for "
                f"{spec!r}"
            )


def _grid_values(spec: ParamSpecDict) -> List[Any]:
    """The discrete candidate values for one param along a grid axis.

    choice → its choices; numeric with a ``step`` → ``low, low+step, …, high``
    (clamped, de-duplicated); numeric without a ``step`` → a small default
    sweep of 5 points across ``[low, high]`` (so a grid is always finite and the
    default endpoint is included). Int axes are rounded to ints.
    """
    if spec["type"] == "choice":
        return list(spec["choices"])
    lo, hi = float(spec["low"]), float(spec["high"])
    step = spec.get("step")
    if step:
        n = int(round((hi - lo) / step))
        vals = [lo + i * step for i in range(n + 1)]
        if vals and vals[-1] < hi:
            vals.append(hi)
    elif hi == lo:
        vals = [lo]
    else:
        # Default 5-point sweep, endpoints included.
        vals = [lo + (hi - lo) * i / 4 for i in range(5)]
    out: List[Any] = []
    seen = set()
    for v in vals:
        cv = _clamp(spec, v)
        if cv not in seen:
            seen.add(cv)
            out.append(cv)
    return out


def _random_value(spec: ParamSpecDict, rng: random.Random) -> Any:
    """A single seeded random in-bounds value for one param."""
    if spec["type"] == "choice":
        return rng.choice(list(spec["choices"]))
    lo, hi = spec["low"], spec["high"]
    if spec["type"] == "int":
        return rng.randint(int(lo), int(hi))
    return _clamp(spec, rng.uniform(lo, hi))


# ---------------------------------------------------------------------------
# Strategies. Each yields an ordered list of candidate dicts (all in-bounds).
# ---------------------------------------------------------------------------
def _grid_candidates(
    param_space: "Mapping[ParamKey, ParamSpecDict]", max_evals: int
) -> List[Dict[ParamKey, Any]]:
    """Cartesian product over each param's grid axis, capped at ``max_evals``.

    Deterministic: the axis order is the (ordered) ``param_space`` order and
    each axis is :func:`_grid_values`, so the enumeration is stable. If the full
    grid exceeds ``max_evals`` it is truncated to the first ``max_evals`` points
    (a documented, deterministic cap — not a random subsample).
    """
    keys = list(param_space.keys())
    axes = [_grid_values(param_space[k]) for k in keys]
    full = 1
    for ax in axes:
        full *= max(len(ax), 1)
    out: List[Dict[ParamKey, Any]] = []
    for combo in itertools.product(*axes):
        out.append(dict(zip(keys, combo)))
        if len(out) >= max_evals:
            break
    if full > len(out):
        # The cap drops the tail of the lexicographic enumeration (high values
        # of the first axis), so the optimum there is never evaluated -- warn
        # loudly rather than silently ship a corner-of-grid result.
        logger.warning(
            "self_optimize: grid truncated to %d of %d points by max_evals; "
            "raise max_evals or coarsen the grid for full coverage.",
            len(out),
            full,
        )
    return out


def _coordinate_candidates(
    param_space: "Mapping[ParamKey, ParamSpecDict]",
    seed: int,
    max_evals: int,
    rounds: int,
) -> Tuple[
    List[Dict[ParamKey, Any]],
    Callable[[Dict[Tuple, float]], None],
]:
    """Few-round **coordinate-descent with random restart** (dependency-light).

    This strategy is *adaptive*: the next batch of candidates depends on which
    earlier candidates scored well, so it cannot be fully pre-generated. Instead
    it returns ``(first_batch, advance)`` where ``advance(scores)`` is called by
    :func:`optimize` after each batch with the ``{frozen_candidate -> score}``
    seen so far, and mutates internal state to stage the next batch (exposed via
    the closure's ``pending`` list). Kept in this hand-rolled form so there is
    **no heavy new dependency** (no GP/TPE library); it is a lightweight,
    fully-seeded search:

    * start at the data-driven defaults (one eval);
    * each round, sweep one param at a time over its grid values, keeping the
      best-so-far on the other params (coordinate descent);
    * between rounds inject a seeded random restart so a bowl with the optimum
      off the default axis is still reachable.

    The whole thing is driven by ``random.Random(seed)`` so it is reproducible.
    """
    rng = random.Random(seed)
    keys = list(param_space.keys())
    state: Dict[str, Any] = {
        "best": {k: _default_of(param_space[k]) for k in keys},
        "best_score": None,  # filled by optimize's minimize/maximize pick
        "pending": [],
    }

    # The optimize loop reads this to know what to score next; we expose it.
    first = [dict(state["best"])]
    state["pending"] = []

    def _frozen(cand: Dict[ParamKey, Any]) -> Tuple:
        return tuple(cand[k] for k in keys)

    def advance(scores: Dict[Tuple, float], minimize: bool) -> None:
        # Pick the current best from everything scored so far.
        if not scores:
            state["pending"] = []
            return
        pick = min if minimize else max
        best_frozen = pick(scores, key=lambda fc: scores[fc])
        state["best"] = dict(zip(keys, best_frozen))
        # Stage the next batch: sweep each coordinate over its grid, plus a
        # seeded random restart. Skip already-scored points.
        batch: List[Dict[ParamKey, Any]] = []
        for k in keys:
            for v in _grid_values(param_space[k]):
                cand = dict(state["best"])
                cand[k] = v
                if _frozen(cand) not in scores and cand not in batch:
                    batch.append(cand)
        restart = {k: _random_value(param_space[k], rng) for k in keys}
        if _frozen(restart) not in scores and restart not in batch:
            batch.append(restart)
        remaining = max_evals - len(scores)
        state["pending"] = batch[:remaining] if remaining > 0 else []

    # Attach for optimize to drive.
    advance._state = state  # type: ignore[attr-defined]
    advance._rounds = rounds  # type: ignore[attr-defined]
    return first, advance


# ---------------------------------------------------------------------------
# The optimizer.
# ---------------------------------------------------------------------------
def optimize(
    param_space: "Mapping[ParamKey, ParamSpecDict]",
    objective: str,
    evaluate_fn: Callable[[Dict[ParamKey, Any]], Mapping[str, Any]],
    *,
    seed: int,
    strategy: str = "grid",
    max_evals: int = 64,
    rounds: int = 3,
    minimize: Optional[bool] = None,
    holdout_eval: Optional[
        Callable[[Dict[ParamKey, Any]], Mapping[str, Any]]
    ] = None,
    parallel_map: Optional[
        Callable[
            [
                Callable[[Dict[ParamKey, Any]], Mapping[str, Any]],
                Sequence[Dict[ParamKey, Any]],
            ],
            List[Mapping[str, Any]],
        ]
    ] = None,
    registry: Any = None,
    run_context: Optional[Dict[str, Any]] = None,
) -> OptimizeResult:
    """Search ``param_space`` for the params that best hit ``objective``.

    Fans every candidate out through ``evaluate_fn`` (the injected seam — a
    cluster analysis run in production, a synthetic objective in tests), reads
    back the ``objective`` metric, and returns the best params + full search
    provenance. **Advisory**: it proposes params; it never mutates a run.

    Parameters
    ----------
    param_space : Mapping[(module, param), spec]
        The *ordered* searchable space (see :data:`ParamSpace`). Only these
        flagged params are searched; a spec's ``default`` seeds the start point
        from the WP-10 data-driven default.
    objective : str
        The metric name to optimize (e.g. ``"nena_nm"``). Each ``evaluate_fn``
        result must carry it (a candidate whose metrics omit the objective is
        skipped, logged).
    evaluate_fn : callable
        ``evaluate_fn(params) -> metrics``. Runs one analysis for a param set
        and returns its metrics dict. In production this fans out a cluster run
        and reads the metrics back from the registry.
    seed : int
        The recorded search seed. Drives every strategy, so two calls with the
        same seed are identical (determinism).
    strategy : str, optional
        ``"grid"`` (cartesian over per-param grids) or ``"coordinate"`` (a
        few-round, dependency-light coordinate-descent with seeded random
        restart). Default ``"grid"``.
    max_evals : int, optional
        Hard cap on the number of ``evaluate_fn`` calls. Default 64.
    rounds : int, optional
        Number of refine rounds for ``"coordinate"`` (ignored for ``"grid"``).
    minimize : bool, optional
        Override the direction. Default: inferred from the A1/C19 table via
        :func:`infer_minimize` (NeNA / precision / drift / FRC are minimized).
    holdout_eval : callable, optional
        A second evaluator on a HELD-OUT split (e.g. a different noise seed).
        When given, the winner is re-scored on it and recorded; if a *different*
        candidate wins the held-out split, that one is preferred — the guard
        against overfitting the search set.
    parallel_map : callable, optional
        ``parallel_map(fn, candidates) -> [metrics, ...]`` — a hook to batch the
        fan-out (e.g. submit all cluster runs, then collect). Default: serial
        ``[evaluate_fn(c) for c in candidates]``. The search logic is agnostic
        to parallelism; this only changes *how* a batch is scored.
    registry : object, optional
        Injected registry client. When given, the chosen params + provenance are
        written back **best-effort** (a failure never raises). picasso-registry
        is imported lazily; omitted ⇒ nothing is persisted.
    run_context : dict, optional
        Extra provenance to persist (``acquisition_run_id`` / ``taxon_id`` /
        ``analysis_run_id`` are read by the write-back; everything else rides in
        ``extra``).

    Returns
    -------
    OptimizeResult
        Best params (as ``(module, param) -> value``) + rationale + provenance.

    Raises
    ------
    ValueError
        If ``param_space`` is malformed or ``strategy`` is unknown (a caller
        programming error, surfaced before any evaluation).
    """
    _validate_space(param_space)
    mini = infer_minimize(objective) if minimize is None else bool(minimize)
    keys = list(param_space.keys())

    def _mapper(
        cands: Sequence[Dict[ParamKey, Any]],
    ) -> List[Mapping[str, Any]]:
        # Enforce the bounds property on EVERY candidate before it is scored.
        for c in cands:
            assert_in_bounds(param_space, c)
        if parallel_map is not None:
            return list(parallel_map(evaluate_fn, list(cands)))
        return [evaluate_fn(c) for c in cands]

    # scores: frozen-candidate -> objective value; trace: ordered candidate log.
    scores: Dict[Tuple, float] = {}
    trace: List[Dict[str, Any]] = []

    def _score_batch(cands: Sequence[Dict[ParamKey, Any]]) -> None:
        # Dedup against already-scored candidates (adaptive strategies may
        # re-propose the current best); never double-spend an eval.
        fresh = [c for c in cands if _frozen(c) not in scores]
        if not fresh:
            return
        metrics_list = _mapper(fresh)
        for cand, metrics in zip(fresh, metrics_list):
            val = (metrics or {}).get(objective)
            if val is None:
                logger.warning(
                    "self_optimize: candidate %s returned no %r; skipped.",
                    _frozen(cand),
                    objective,
                )
                continue
            fval = float(val)
            scores[_frozen(cand)] = fval
            trace.append(
                {
                    "params": {f"{m}.{p}": cand[(m, p)] for (m, p) in keys},
                    objective: fval,
                }
            )

    def _frozen(cand: Dict[ParamKey, Any]) -> Tuple:
        return tuple(cand[k] for k in keys)

    if strategy == "grid":
        _score_batch(_grid_candidates(param_space, max_evals))
    elif strategy == "coordinate":
        first, advance = _coordinate_candidates(
            param_space, seed, max_evals, rounds
        )
        _score_batch(first)
        for _ in range(rounds):
            if len(scores) >= max_evals:
                break
            advance(scores, mini)
            pending = advance._state["pending"]  # type: ignore[attr-defined]
            if not pending:
                break
            _score_batch(pending)
    else:
        raise ValueError(
            f"unknown strategy {strategy!r} (want 'grid' or 'coordinate')"
        )

    if not scores:
        # Nothing scored (every candidate lacked the objective). Return the
        # data-driven defaults unchanged, flagged as un-improved.
        defaults = {k: _default_of(param_space[k]) for k in keys}
        result = OptimizeResult(
            best_params=defaults,
            best_score=None,
            objective=objective,
            minimize=mini,
            strategy=strategy,
            seed=seed,
            n_evals=0,
            rationale={"note": "no candidate produced the objective metric"},
        )
        _persist(registry, result, trace, run_context)
        return result

    pick = min if mini else max
    best_frozen = pick(scores, key=lambda fc: scores[fc])
    best_params = dict(zip(keys, best_frozen))
    best_score = scores[best_frozen]

    # Held-out guard: re-score the winner on the held-out split AND check no
    # OTHER scored candidate wins held-out — prefer the config that also wins
    # held-out, so we don't ship a config that only won the search set.
    holdout_score: Optional[float] = None
    holdout_detail: Dict[str, Any] = {}
    if holdout_eval is not None:
        best_params, best_score, holdout_score, holdout_detail = (
            _apply_holdout_guard(
                param_space,
                keys,
                scores,
                best_params,
                best_score,
                objective,
                mini,
                holdout_eval,
            )
        )

    rationale = {
        "objective": objective,
        "minimize": mini,
        "strategy": strategy,
        "seed": seed,
        "n_evals": len(scores),
        "chosen": {f"{m}.{p}": best_params[(m, p)] for (m, p) in keys},
        "best_score": best_score,
        "defaults": {
            f"{m}.{p}": _default_of(param_space[(m, p)]) for (m, p) in keys
        },
    }
    if holdout_eval is not None:
        rationale["holdout"] = holdout_detail

    result = OptimizeResult(
        best_params=best_params,
        best_score=best_score,
        objective=objective,
        minimize=mini,
        strategy=strategy,
        seed=seed,
        n_evals=len(scores),
        holdout_score=holdout_score,
        rationale=rationale,
        provenance={"trace": trace, **rationale},
    )
    _persist(registry, result, trace, run_context)
    return result


def _apply_holdout_guard(
    param_space: "Mapping[ParamKey, ParamSpecDict]",
    keys: List[ParamKey],
    scores: Dict[Tuple, float],
    best_params: Dict[ParamKey, Any],
    best_score: float,
    objective: str,
    minimize: bool,
    holdout_eval: Callable[[Dict[ParamKey, Any]], Mapping[str, Any]],
) -> Tuple[Dict[ParamKey, Any], float, Optional[float], Dict[str, Any]]:
    """VALIDATE the search winner on the held-out split (no re-selection).

    The SEARCH set selects the winner; the held-out split only *validates* that
    it generalizes. The winner is shipped **only if it beats the data-driven
    DEFAULT on the held-out split**; otherwise we fall back to the default (the
    search win didn't generalize -- don't ship an overfit config).

    Crucially this does NOT re-select by taking the argmin/argmax over held-out
    scores of every candidate: that would just move overfitting from the search
    set to the held-out set (held-out selection bias). Only two configs are
    compared on held-out -- the search winner vs. the default. Returns
    ``(params, search_score, holdout_score, detail)``.
    """

    def _frozen(cand: Dict[ParamKey, Any]) -> Tuple:
        return tuple(cand[k] for k in keys)

    def _ho(cand: Dict[ParamKey, Any]) -> Optional[float]:
        assert_in_bounds(param_space, cand)
        val = (holdout_eval(cand) or {}).get(objective)
        return None if val is None else float(val)

    winner_ho = _ho(best_params)
    if winner_ho is None:
        return (
            best_params,
            best_score,
            None,
            {"note": "no held-out metric for the search winner"},
        )

    default_params = {k: _default_of(param_space[k]) for k in keys}
    default_ho = _ho(default_params)
    # The winner generalizes iff it beats the default on the held-out split
    # (unknown default held-out metric → give the winner the benefit).
    if default_ho is None:
        generalizes = True
    elif minimize:
        generalizes = winner_ho <= default_ho
    else:
        generalizes = winner_ho >= default_ho

    detail = {
        "search_winner": {f"{m}.{p}": best_params[(m, p)] for (m, p) in keys},
        "search_winner_holdout_score": winner_ho,
        "default_holdout_score": default_ho,
        "generalizes": generalizes,
        "preferred": "search_winner" if generalizes else "default",
    }
    if generalizes:
        return best_params, best_score, winner_ho, detail
    # Didn't beat the default on unseen data -> revert to the default rather
    # than ship a config that only won the (noisy) search set.
    return (
        default_params,
        scores.get(_frozen(default_params)),
        default_ho,
        detail,
    )


# ---------------------------------------------------------------------------
# Provenance write-back (best-effort, non-blocking; lazy registry import).
# ---------------------------------------------------------------------------
def _persist(
    registry: Any,
    result: OptimizeResult,
    trace: List[Dict[str, Any]],
    run_context: Optional[Dict[str, Any]],
) -> Optional[dict]:
    """Write the chosen params + search provenance to the registry.

    Best-effort / non-blocking (mirrors WP-6/WP-10): a registry outage logs and
    returns ``None`` — it never fails the optimize. The record is an
    ``analysis_run`` row (``kind="optimization"``) carrying the chosen params in
    ``recommended_spec`` + ``parameters``, the ``optimizer_version`` tag, and the
    full search trace under ``extra`` — all first-class columns the schema
    already exposes (no migration invented). picasso-registry is imported lazily
    only via the injected client (never at module load).
    """
    if registry is None:
        return None
    ctx = run_context or {}
    spec = {
        "optimizer_version": result.optimizer_version,
        "objective": result.objective,
        "minimize": result.minimize,
        "strategy": result.strategy,
        "seed": result.seed,
        "chosen_params": result.params_as_str_keys(),
        "best_score": result.best_score,
        "holdout_score": result.holdout_score,
    }
    extra: Dict[str, Any] = {
        "optimizer": "self_optimize",
        "search_trace": trace,
        "rationale": result.rationale,
    }
    for k, v in ctx.items():
        if k not in ("acquisition_run_id", "analysis_run_id", "taxon_id"):
            extra[k] = v
    if "taxon_id" in ctx:
        extra["taxon_id"] = ctx["taxon_id"]

    fields: Dict[str, Any] = {
        "kind": "optimization",
        "status": "proposed",
        "recommender_version": result.optimizer_version,
        "recommended_spec": spec,
        "parameters": {"chosen_params": result.params_as_str_keys()},
        "extra": extra,
    }
    if ctx.get("analysis_run_id"):
        fields["id"] = ctx["analysis_run_id"]
    if ctx.get("acquisition_run_id"):
        fields["acquisition_run_id"] = ctx["acquisition_run_id"]

    try:
        return registry.log_analysis(**fields)
    except Exception as exc:  # registry must never break the optimize
        logger.warning(
            f"self_optimize: provenance write-back failed ({exc!r}); "
            "returning the result anyway (best-effort)."
        )
        return None


# ---------------------------------------------------------------------------
# Production evaluator seam — the WP-9 cluster fan-out.
# ---------------------------------------------------------------------------
def cluster_evaluator(
    base_workflow: Sequence[Tuple[str, dict]],
    submit_fn: Callable[[List[Tuple[str, dict]]], Mapping[str, Any]],
) -> Callable[[Dict[ParamKey, Any]], Mapping[str, Any]]:
    """Build a production ``evaluate_fn`` that fans out a cluster run per candidate.

    The injected seam made concrete: given a ``base_workflow`` (the recommended
    ``(module, params)`` steps) and a ``submit_fn`` that submits one workflow to
    the cluster (via :mod:`picasso_workflow.cluster_submit`) and returns its
    metrics (read back from the registry once the job's write-back lands), this
    returns an ``evaluate_fn(params) -> metrics`` :func:`optimize` can drive.

    The returned evaluator overlays the candidate's ``(module, param) -> value``
    onto a copy of ``base_workflow`` and submits it; it is intentionally thin so
    the real parallel fan-out (submit all candidates, then collect) lives in the
    caller's ``submit_fn`` / the ``parallel_map`` hook, not in the search loop.
    This function is a convenience wrapper — the real cluster round-trip is
    exercised on the cluster, not in the hermetic unit suite.
    """

    def _evaluate(params: Dict[ParamKey, Any]) -> Mapping[str, Any]:
        wf = _overlay_params(base_workflow, params)
        return submit_fn(wf)

    return _evaluate


def _overlay_params(
    base_workflow: Sequence[Tuple[str, dict]],
    params: "Mapping[ParamKey, Any]",
) -> List[Tuple[str, dict]]:
    """Return a copy of ``base_workflow`` with ``(module, param)`` overrides.

    Each ``(module, param) -> value`` in ``params`` is written onto the FIRST
    step whose module name matches; a param whose module is absent is ignored
    (logged) rather than silently inventing a step. Pure/deterministic — the
    building block the production evaluator uses to turn a candidate into a
    runnable workflow.
    """
    out = [(m, dict(p)) for m, p in base_workflow]
    index: Dict[str, int] = {}
    for i, (m, _) in enumerate(out):
        index.setdefault(m, i)
    for (module, param), value in params.items():
        if module not in index:
            logger.warning(
                "self_optimize: param module %r not in base workflow; "
                "override %r dropped.",
                module,
                param,
            )
            continue
        out[index[module]][1][param] = value
    return out
