#!/usr/bin/env python
"""Sample-aware analysis-workflow recommender (WP-10).

Turn *(conditions, goal, dataset probe, history)* into a **runnable**
picasso-workflow spec — an ordered ``(module, parameters)`` list the existing
``WorkflowRunner`` / ``AggregationWorkflowRunner`` execute **unchanged**. The
recommender is **advisory**: it emits a spec for review, records *why* each
decision was made, and never gates a run.

It composes the four layers of the Part IV sketch, each cheap, pure-ish, and
unit-tested in isolation:

* **Layer 0 — template select** (:func:`select_modules`): pick a
  :class:`~picasso_workflow.templates.Template` from ``TEMPLATE_REGISTRY`` keyed
  by the ``goal`` (question-type) and the *coarse* ratified descriptor axes
  (``sample_type`` / ``modality`` / ``dimensionality`` — NO A2 leaf vocab), and
  resolve it to its concrete default module list via ``Template.resolve``.
* **Layer 1 — resolve choice points** (:func:`resolve_choice_points`): for
  modules that fill a mutually-exclusive ``role`` (``undrift``, ``clusterer``,
  …) choose the variant that performed best in the **cohort ranked by
  sample-taxon tree distance** (``db_client.cohort(taxon_id)``), joined to its
  ``metrics``; keep the template default when there is no usable history
  (explainable).
* **Layer 2 — recommend parameters** (:func:`recommend_parameters`): seed
  data-driven defaults from the taxon node (``db_client.node_defaults``) and the
  WP-7 learned per-cohort ranges (latest ``cohort_metric_range`` rows), used
  *conservatively* — only override a template default when the cohort evidence
  is real.
* **Layer 3 — validate + order** (:func:`topological_order`): order the modules
  so every ``requires`` is satisfied by an earlier ``provides`` and assert the
  result passes :func:`~picasso_workflow.modulespec.validate_workflow`.

Design constraints
------------------
* **picasso-registry is NOT a dependency.** The registry ``db_client`` /
  ``registry`` are **injected** params; the core imports headlessly and is
  tested with the in-memory ``picasso_registry.testing`` double. The persistence
  hook imports the client only lazily, inside the write path.
* **Best-effort registry I/O.** Every cohort/defaults/range read is wrapped so a
  registry outage degrades the recommendation (fewer data-driven choices) rather
  than raising into the caller — same posture as
  :mod:`picasso_workflow.registry_client`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from picasso_workflow import templates as tmpl
from picasso_workflow.modulespec import (
    MODULE_REGISTRY,
    ModuleSpec,
    Scope,
    _step_name,
    _step_params,
    validate_workflow,
)

# A workflow step as the runner / recipes represent it: (module_name, params).
Step = Tuple[str, Dict[str, Any]]
Workflow = List[Step]

#: Recommender version tag persisted alongside every recommended spec. Bumped by
#: hand when the selection/parameter logic changes in a way that should
#: invalidate "this spec came from recommender X" comparisons. (The package
#: version moves on every commit via setuptools-scm, so a dedicated, coarser tag
#: is the stable provenance key for the recommender's *behaviour*.)
RECOMMENDER_VERSION = "wp10-recommender-1"

#: Fallback template per scope when nothing matches goal/conditions. A sane,
#: minimal default keeps the recommender total — it always returns a runnable
#: spec rather than failing.
_DEFAULT_TEMPLATE = {
    Scope.SINGLE: "minimal",
    Scope.AGGREGATION: "minimal_channel_align",
}

#: Ratified A2 descriptor axes the Layer-0 match is allowed to use (NO leaf
#: vocab). Any other key in ``conditions`` (e.g. ``taxon_id``, prep facets) is
#: metadata, not a template-match axis.
_COARSE_AXES = ("sample_type", "modality", "dimensionality")

#: A1/C19 optimize-tier metrics that rank choice-point variants, with the
#: direction that is "better". All current primaries are minimized (lower NeNA /
#: precision / drift residual is better); kept as a table so adding a
#: maximize-metric later is a one-line change.
_LOWER_IS_BETTER = (
    "nena_nm",
    "lp_x_nm",
    "lp_y_nm",
    "drift_residual_nm",
    "frc_nm",
)

#: Minimum cohort samples a variant needs before it may REPLACE the template
#: default (a single lucky run must not flip the pipeline). The template default
#: is the conservative baseline; swap only on real evidence.
MIN_COHORT_SAMPLES = 3


# ---------------------------------------------------------------------------
# Result container.
# ---------------------------------------------------------------------------
@dataclass
class Recommendation:
    """A recommended workflow plus the rationale behind every decision.

    ``workflow`` is the ordered ``(module, params)`` step list — exactly the
    format the runner / recipes consume, so it runs unchanged. ``to_yaml``
    renders the same content as a standard workflow YAML. ``rationale`` records,
    per layer, *which* template / variants / params were chosen and *why*, so
    the recommendation is auditable.
    """

    workflow: Workflow
    scope: Scope
    template_id: Optional[str]
    recommender_version: str = RECOMMENDER_VERSION
    rationale: Dict[str, Any] = field(default_factory=dict)

    def to_yaml(self) -> str:
        """Render the workflow as a standard workflow YAML string.

        The steps are emitted as a list of ``{module, parameters}`` mappings —
        the canonical, review-friendly serialization of the same content the
        runner consumes.
        """
        import yaml

        doc = {
            "scope": self.scope.value,
            "template_id": self.template_id,
            "recommender_version": self.recommender_version,
            "workflow": [
                {"module": _step_name(s), "parameters": _step_params(s)}
                for s in self.workflow
            ],
        }
        return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Layer 0 — template select.
# ---------------------------------------------------------------------------
def _scope_for_goal(goal: str, conditions: Dict[str, Any]) -> Scope:
    """Infer the workflow scope from the goal / conditions.

    A channel-align / aggregate goal (or an explicit ``scope`` in conditions)
    is AGGREGATION; everything else is a SINGLE-dataset workflow.
    """
    explicit = (conditions or {}).get("scope")
    if explicit:
        try:
            return Scope(explicit)
        except ValueError:
            # recommend_workflow never raises: an invalid scope string falls
            # back to goal inference rather than propagating a ValueError.
            logger.warning(
                "recommender: ignoring invalid conditions['scope']=%r",
                explicit,
            )
    g = (goal or "").lower()
    if "align" in g or "aggregate" in g or "aggregation" in g:
        return Scope.AGGREGATION
    return Scope.SINGLE


def _descriptor_tag_match(
    template: tmpl.Template, conditions: Dict[str, Any]
) -> bool:
    """True if a template's coarse descriptor tags are compatible with conditions.

    A template matches when, for every coarse axis it *declares*, the condition
    either agrees or is unspecified. A template with no ``descriptor_tags``
    matches broadly (the common case today). Leaf vocab is never consulted.
    """
    tags = template.descriptor_tags or {}
    for axis in _COARSE_AXES:
        want = tags.get(axis)
        if want is None:
            continue
        have = (conditions or {}).get(axis)
        if have is not None and have != want:
            return False
    return True


def select_modules(
    goal: str, conditions: Dict[str, Any]
) -> Tuple[Workflow, Optional[tmpl.Template]]:
    """Layer 0: pick a template and resolve it to a default module list.

    Keyed by the ``goal`` (question-type, e.g. ``"localize"`` /
    ``"channel-align"``) and the coarse ratified descriptor axes in
    ``conditions``. Falls back to a sane default template for the inferred scope
    when nothing matches, so the recommender always returns a runnable workflow.

    Returns
    -------
    (workflow, template)
        The resolved ``(module, params)`` steps and the chosen
        :class:`~picasso_workflow.templates.Template` (``None`` only if even the
        fallback id is missing from the registry — then ``workflow`` is empty).
    """
    scope = _scope_for_goal(goal, conditions)
    goal_l = (goal or "").lower()

    candidates = [
        t
        for t in tmpl.TEMPLATE_REGISTRY.values()
        if t.scope == scope
        and t.question_type.lower() == goal_l
        and _descriptor_tag_match(t, conditions)
    ]
    # Prefer the template declaring the most coarse-axis tags that the
    # conditions actually pin (most specific wins); deterministic tiebreak on id.
    if candidates:

        def _specificity(t: tmpl.Template) -> int:
            tags = t.descriptor_tags or {}
            return sum(
                1
                for a in _COARSE_AXES
                if tags.get(a) is not None
                and (conditions or {}).get(a) is not None
            )

        candidates.sort(key=lambda t: (-_specificity(t), t.template_id))
        chosen = candidates[0]
    else:
        chosen = tmpl.get_template(_DEFAULT_TEMPLATE.get(scope, "minimal"))

    if chosen is None:
        return [], None
    return list(chosen.resolve()), chosen


# ---------------------------------------------------------------------------
# Layer 1 — resolve choice points (cohort history, taxon-distance ranked).
# ---------------------------------------------------------------------------
def _safe(call, what: str, default):
    """Run a best-effort registry call; log + return ``default`` on failure."""
    try:
        return call()
    except Exception as exc:  # registry must never break the recommendation
        logger.warning(f"recommender: {what} failed ({exc!r}); using default.")
        return default


def _cohort_runs(db_client, taxon_id: str) -> List[dict]:
    """Cohort acquisition runs for ``taxon_id``, ranked by tree distance.

    Thin, best-effort wrapper over ``db_client.cohort`` — the registry already
    ranks exact-node matches first and falls back *up* the sample tree, so a
    sparse exact node still yields a non-empty cohort from its ancestors.
    """
    if db_client is None or not taxon_id:
        return []
    return _safe(lambda: db_client.cohort(taxon_id), "cohort query", [])


def _metrics_by_run(db_client) -> Dict[str, List[dict]]:
    """Map ``analysis_run_id -> [metrics rows]`` from the registry.

    The list endpoints have no server-side filter, so (like the WP-7 backfill)
    we pull the ``metrics`` table once and group client-side. Best-effort: a
    failure yields an empty map and the choice points fall back to defaults.
    """
    rows = _safe(lambda: _list_all(db_client, "metrics"), "list metrics", [])
    out: Dict[str, List[dict]] = {}
    for row in rows:
        rid = row.get("analysis_run_id")
        if rid:
            out.setdefault(rid, []).append(row)
    return out


def _analysis_by_acquisition(db_client) -> Dict[str, List[dict]]:
    """Map ``acquisition_run_id -> [analysis_run rows]`` (best-effort)."""
    rows = _safe(
        lambda: _list_all(db_client, "analysis_run"), "list analysis_run", []
    )
    out: Dict[str, List[dict]] = {}
    for row in rows:
        aid = row.get("acquisition_run_id")
        if aid:
            out.setdefault(aid, []).append(row)
    return out


def _list_all(client, resource: str, page: int = 500) -> List[dict]:
    """Page through a registry list endpoint (limit/offset only, no filter)."""
    out: List[dict] = []
    offset = 0
    while True:
        batch = client.list(resource, limit=page, offset=offset)
        out.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return out


def _module_of(metrics_row: dict) -> Optional[str]:
    """The module a metrics row belongs to.

    ``RegistryLogger.log_module_metrics`` records the module name (and index)
    into the row's ``extra`` JSON (the schema has no first-class ``module``
    column), so that is where the variant identity lives. A top-level
    ``module`` key is also honoured for forward-compatibility.
    """
    if metrics_row.get("module"):
        return metrics_row["module"]
    extra = metrics_row.get("extra") or {}
    return extra.get("module")


def _variant_scores(
    db_client, taxon_id: str, candidate_names: set
) -> Tuple[Dict[str, Tuple[float, int]], Optional[str]]:
    """Per-candidate mean of ONE shared optimize-tier metric, from the cohort.

    Walks the taxon-distance-ranked cohort → its analysis runs → their metrics,
    collecting each candidate module's values **per metric** (not mixed). It
    then picks a single *comparison metric* -- the highest-priority
    ``_LOWER_IS_BETTER`` metric any candidate has data for -- and scores every
    candidate on THAT metric only, so variants are ranked on the same physical
    quantity (NeNA nm vs drift-residual nm are not averaged together). Returns
    ``({module: (mean, n_samples)}, comparison_metric)``; ``({}, None)`` when no
    candidate has any usable metric.
    """
    cohort = _cohort_runs(db_client, taxon_id)
    if not cohort:
        return {}, None
    ana_by_acq = _analysis_by_acquisition(db_client)
    metrics_by_ana = _metrics_by_run(db_client)

    # Per (module, metric) values -- metrics are kept separate, never mixed.
    per: Dict[str, Dict[str, List[float]]] = {}
    for item in cohort:
        acq_id = item.get("acquisition_run_id")
        for ana in ana_by_acq.get(acq_id, []):
            for row in metrics_by_ana.get(ana.get("id"), []):
                module = _module_of(row)
                if module not in candidate_names:
                    continue
                for metric in _LOWER_IS_BETTER:
                    val = row.get(metric)
                    if val is None:
                        continue
                    try:
                        fval = float(val)
                    except (TypeError, ValueError):
                        continue
                    per.setdefault(module, {}).setdefault(metric, []).append(
                        fval
                    )

    # The shared comparison metric: the highest-priority one any candidate has.
    metric = next(
        (m for m in _LOWER_IS_BETTER if any(m in v for v in per.values())),
        None,
    )
    if metric is None:
        return {}, None
    scores = {
        module: (sum(vals[metric]) / len(vals[metric]), len(vals[metric]))
        for module, vals in per.items()
        if metric in vals
    }
    return scores, metric


def _roles(workflow: Workflow) -> Dict[str, List[int]]:
    """Map ``role -> [step indices]`` for steps whose module declares a role."""
    out: Dict[str, List[int]] = {}
    for i, step in enumerate(workflow):
        spec = MODULE_REGISTRY.get(_step_name(step))
        if spec is not None and spec.role:
            out.setdefault(spec.role, []).append(i)
    return out


def _alternatives(role: str, default_spec: ModuleSpec) -> List[str]:
    """Modules interchangeable with ``default_spec`` for its ``role``.

    An alternative must (a) fill the same ``role``, (b) be valid in **every**
    scope the default occupies (``default_spec.scopes <= spec.scopes``) so the
    swap is scope-safe wherever the default sat — crucially, a SINGLE-scope
    undrift in the per-dataset phase of an aggregation workflow picks among
    SINGLE-scope undrift variants, not aggregation ones — and (c) share the
    default's ``module_set`` so a native template's choice points stay among
    native modules (and a picasso-set template's among picasso-set modules),
    never mixing the two families.
    """
    return sorted(
        name
        for name, spec in MODULE_REGISTRY.items()
        if spec.role == role
        and default_spec.scopes <= spec.scopes
        and spec.module_set == default_spec.module_set
        # (d) an alternative must not REQUIRE more than the default: swapping in
        # a variant with extra data deps (e.g. undrift_from_picked needs
        # `picks`, which a plain localize template never provides) would make
        # the graph unorderable and silently revert the whole recommendation.
        and spec.requires <= default_spec.requires
    )


def resolve_choice_points(
    workflow: Workflow,
    conditions: Dict[str, Any],
    db_client: Any,
    dataset_uri: Optional[str] = None,
) -> Tuple[Workflow, List[dict]]:
    """Layer 1: swap each role's default for the cohort's best-performing variant.

    For every module filling a mutually-exclusive ``role`` (``undrift``,
    ``clusterer``, …) with a genuine alternative, rank the alternatives by their
    mean A1/C19 optimize-tier metric over the taxon-distance-ranked cohort and
    switch to the winner (carrying the step's original params, which validate
    against the same ``requires``/``provides``). With no usable history the
    template default is kept — explainable. ``dataset_uri`` is an optional, as
    yet unused, data-probe hook (kept for signature stability).

    Returns the (possibly rewritten) workflow and a list of per-decision
    rationale dicts.
    """
    taxon_id = (conditions or {}).get("taxon_id")
    out = list(workflow)
    decisions: List[dict] = []

    for role, indices in _roles(workflow).items():
        # Only one slot per role in the shipped recipes; resolve the first.
        idx = indices[0]
        default_name = _step_name(out[idx])
        default_spec = MODULE_REGISTRY.get(default_name)
        if default_spec is None:
            continue
        alts = _alternatives(role, default_spec)
        if len(alts) <= 1:
            continue  # no genuine choice

        scores, metric = _variant_scores(db_client, taxon_id, set(alts))
        if not scores:
            decisions.append(
                {
                    "role": role,
                    "chosen": default_name,
                    "source": "template-default",
                    "reason": "no cohort history for the candidate variants",
                }
            )
            continue

        # Lower is better on the shared comparison metric (see _variant_scores).
        best_name, (best_val, n) = min(scores.items(), key=lambda kv: kv[1][0])
        default_score = scores.get(default_name)
        if best_name == default_name:
            decisions.append(
                {
                    "role": role,
                    "chosen": default_name,
                    "source": "cohort-confirmed-default",
                    "metric": metric,
                    "metric_mean": best_val,
                    "n_samples": n,
                }
            )
            continue

        # Guard against a noisy swap: only replace the template default when the
        # winner clears MIN_COHORT_SAMPLES AND (when the default has cohort data)
        # actually beats it. Otherwise keep the explainable template default -- a
        # single lucky run must not flip the pipeline.
        beats_default = default_score is None or best_val < default_score[0]
        if n < MIN_COHORT_SAMPLES or not beats_default:
            decisions.append(
                {
                    "role": role,
                    "chosen": default_name,
                    "source": "template-default-kept",
                    "reason": (
                        f"insufficient cohort evidence to swap in {best_name} "
                        f"(n={n} on {metric}; need n>={MIN_COHORT_SAMPLES} and "
                        "to beat the default)"
                    ),
                    "metric": metric,
                    "candidates": {k: v[0] for k, v in scores.items()},
                }
            )
            continue

        params = _step_params(out[idx])
        out[idx] = (best_name, params)
        decisions.append(
            {
                "role": role,
                "chosen": best_name,
                "replaced": default_name,
                "source": "cohort-history",
                "metric": metric,
                "metric_mean": best_val,
                "n_samples": n,
                "candidates": {k: v[0] for k, v in scores.items()},
            }
        )
    return out, decisions


# ---------------------------------------------------------------------------
# Layer 2 — recommend parameters (node defaults + learned cohort ranges).
# ---------------------------------------------------------------------------
def _latest_cohort_ranges(db_client, taxon_id: str) -> Dict[str, dict]:
    """Latest WP-7 ``cohort_metric_range`` row per metric for ``taxon_id``.

    The store is append-only: a re-derived range inserts a NEW row, so we read
    the LATEST per (taxon, metric). The list endpoint returns rows ordered by
    ULID id (creation time), so the last-seen row per metric is the newest.
    Best-effort; an empty map leaves params at their node/template defaults.
    """
    if db_client is None or not taxon_id:
        return {}
    rows = _safe(
        lambda: _list_all(db_client, "cohort_metric_range"),
        "list cohort_metric_range",
        [],
    )
    latest: Dict[str, dict] = {}
    for row in rows:
        if row.get("taxon_id") != taxon_id:
            continue
        metric = row.get("metric")
        if metric:
            latest[metric] = row  # later row wins (newest by id order)
    return latest


def _node_default_overrides(
    node_defaults: Dict[str, Any], name: str, params: Dict[str, Any]
) -> Dict[str, Tuple[Any, str]]:
    """Resolve the node-default overrides that apply to module ``name``.

    Supports the three shapes the ``node_defaults`` blob may use (so the
    docstring's promise is actually honored):

    * ``{"<module>": {"<param>": val}}`` -- a per-module dict;
    * ``"<module>.<param>": val`` -- a dotted key for one module's param;
    * ``"<param>": val`` (a bare scalar) -- applied to EVERY module that
      already declares that param (safe: never introduces a param the module
      doesn't use).

    Returns ``{param: (value, source_tag)}``; the more specific per-module /
    dotted forms win over a bare scalar on the same param.
    """
    overrides: Dict[str, Tuple[Any, str]] = {}
    # Bare "<param>" first (lowest precedence), only where the module uses it.
    for key, val in node_defaults.items():
        if isinstance(val, dict) or "." in key:
            continue
        if key in params:
            overrides[key] = (val, "node_defaults[<param>]")
    # Dotted "<module>.<param>".
    for key, val in node_defaults.items():
        if isinstance(val, dict) or "." not in key:
            continue
        mod, _, param = key.partition(".")
        if mod == name:
            overrides[param] = (val, f"node_defaults[{key}]")
    # Per-module dict (highest precedence).
    md = node_defaults.get(name)
    if isinstance(md, dict):
        for param, val in md.items():
            overrides[param] = (val, f"node_defaults[{name}]")
    return overrides


def recommend_parameters(
    workflow: Workflow,
    conditions: Dict[str, Any],
    db_client: Any,
) -> Tuple[Workflow, List[dict]]:
    """Layer 2: seed data-driven param defaults from the taxon node + ranges.

    Reads the taxon node's inherited ``defaults`` (``db_client.node_defaults``)
    and the WP-7 learned per-cohort ranges (latest ``cohort_metric_range`` per
    metric). A node default for ``"<module>.<param>"`` (or a bare ``"<param>"``
    applied to every module declaring it) *conservatively* overrides a template
    default — only when the registry evidence is real. Learned ranges are
    recorded as param *hints* in the rationale (e.g. an expected density band),
    not silently written into opaque module params.

    Returns the (possibly updated) workflow and the per-param rationale.
    """
    taxon_id = (conditions or {}).get("taxon_id")
    decisions: List[dict] = []
    if db_client is None or not taxon_id:
        return list(workflow), decisions

    nd = (
        _safe(lambda: db_client.node_defaults(taxon_id), "node_defaults", {})
        or {}
    )
    node_defaults = nd.get("defaults") or {}
    ranges = _latest_cohort_ranges(db_client, taxon_id)

    out: Workflow = []
    for step in workflow:
        name = _step_name(step)
        params = dict(_step_params(step))
        overrides = _node_default_overrides(node_defaults, name, params)
        for key, (val, src) in overrides.items():
            if params.get(key) != val:
                decisions.append(
                    {
                        "module": name,
                        "param": key,
                        "value": val,
                        "source": src,
                        "taxon_id": taxon_id,
                    }
                )
                params[key] = val
        out.append((name, params))

    # Learned ranges are advisory hints (the recommender stays conservative and
    # does not guess which opaque module param a density band maps onto).
    for metric, row in ranges.items():
        decisions.append(
            {
                "hint": "learned-cohort-range",
                "metric": metric,
                "p05": row.get("p05"),
                "p50": row.get("p50"),
                "p95": row.get("p95"),
                "n": row.get("n"),
                "source": "cohort_metric_range",
                "taxon_id": taxon_id,
            }
        )
    return out, decisions


# ---------------------------------------------------------------------------
# Layer 3 — validate + topological order.
# ---------------------------------------------------------------------------
def _spec_for(step) -> Optional[ModuleSpec]:
    return MODULE_REGISTRY.get(_step_name(step))


def _order_segment(steps: Workflow) -> Workflow:
    """Stable Kahn topological sort of one same-scope segment.

    A step becomes ready once every capability it ``requires`` is provided by an
    already-emitted step and all its ``after`` predecessors are emitted. Ties
    keep input order (stable), so a segment that is already a valid order is
    returned unchanged. Raises ``ValueError`` on an unsatisfiable requirement /
    cycle.
    """
    steps = list(steps)
    emitted: List[int] = []
    available: set = set()
    emitted_names: set = set()
    remaining = list(range(len(steps)))

    while remaining:
        progressed = False
        for pos in list(remaining):
            spec = _spec_for(steps[pos])
            if spec is None:
                # Unknown module: no declared requirements to satisfy; emit in
                # place so validate_workflow can report it rather than hanging.
                ready = True
            else:
                ready = (
                    spec.requires <= available and spec.after <= emitted_names
                )
            if ready:
                emitted.append(pos)
                remaining.remove(pos)
                if spec is not None:
                    available |= spec.provides
                emitted_names.add(_step_name(steps[pos]))
                progressed = True
        if not progressed:
            raise ValueError(
                "cannot topologically order workflow; unsatisfiable "
                f"requirements remain for steps {remaining}"
            )
    return [steps[i] for i in emitted]


def _split_aggregation(workflow: Workflow) -> Tuple[Workflow, Workflow]:
    """Split an aggregation workflow into (single-dataset phase, aggregation phase).

    An aggregation template resolves to the per-dataset ``single_dataset_modules``
    (SINGLE scope) followed by the ``aggregation_modules`` (AGGREGATION scope),
    concatenated (see ``Template.resolve``). The two run in *different* scopes,
    so they must be ordered and validated independently. The boundary is the
    first module that is valid ONLY in aggregation scope (``load_datasets_to_
    aggregate`` in the shipped recipes); everything before it is the SINGLE
    phase. With no such module the whole workflow is the aggregation phase.
    """
    for i, step in enumerate(workflow):
        spec = _spec_for(step)
        if spec is not None and spec.scopes == frozenset({Scope.AGGREGATION}):
            return list(workflow[:i]), list(workflow[i:])
    return [], list(workflow)


def topological_order(workflow: Workflow, scope: Scope) -> Workflow:
    """Layer 3: order the modules so every ``requires`` is met by an earlier step.

    A stable Kahn topological sort over the ModuleSpec ``requires``/``provides``
    DAG (plus ``after`` ordering edges). The template recipes are already valid
    orders, so this is a validate-and-preserve pass in practice; it reorders
    only if a Layer-1/2 rewrite perturbed the order, and keeps input order on
    ties (the familiar recipe sequence is retained).

    An **aggregation** workflow is two phases in two scopes — the per-dataset
    SINGLE modules then the AGGREGATION modules (``Template.resolve`` concatenates
    them). Each phase is ordered and validated **in its own scope** (a
    single-dataset module is not valid in aggregation scope and vice-versa), then
    re-concatenated — so the whole spec is checked, not just one half.

    Raises
    ------
    ValueError
        If the modules cannot be ordered into a valid workflow (a dependency
        cycle, an unsatisfiable requirement, or a scope violation) — caught by
        the caller, which falls back to the template default order.
    """
    if scope == Scope.AGGREGATION:
        single_phase, agg_phase = _split_aggregation(workflow)
        single_ordered = _order_segment(single_phase)
        agg_ordered = _order_segment(agg_phase)
        _assert_valid(single_ordered, Scope.SINGLE)
        _assert_valid(agg_ordered, Scope.AGGREGATION)
        return single_ordered + agg_ordered

    ordered = _order_segment(workflow)
    _assert_valid(ordered, scope)
    return ordered


def _assert_valid(workflow: Workflow, scope: Scope) -> None:
    """Raise ``ValueError`` if ``workflow`` does not validate in ``scope``."""
    errors = validate_workflow(workflow, scope)
    if errors:
        raise ValueError(
            "recommended workflow failed validation: " + "; ".join(errors)
        )


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def recommend_workflow(
    conditions: Dict[str, Any],
    goal: str,
    registry: Any = None,
    db_client: Any = None,
    dataset_uri: Optional[str] = None,
) -> Recommendation:
    """Recommend a runnable, ordered workflow for ``(conditions, goal)``.

    Composes the four layers (Part IV). The result is **advisory** — it runs
    unchanged through ``WorkflowRunner`` and the recommender never gates.

    Parameters
    ----------
    conditions : dict
        Coarse ratified descriptor axes (``sample_type`` / ``modality`` /
        ``dimensionality``) plus ``taxon_id`` (the cohort key for Layers 1–2)
        and an optional ``scope``. Leaf vocab is never consulted.
    goal : str
        The question-type (e.g. ``"localize"`` / ``"channel-align"``) keying the
        Layer-0 template select.
    registry : object, optional
        Injected registry client used by the persistence hook (``log_analysis``
        surface). Accepted for symmetry with the design signature; cohort /
        param reads use ``db_client``. If omitted, no spec is persisted here.
    db_client : object, optional
        Injected registry client exposing ``cohort`` / ``node_defaults`` /
        ``list`` (the real ``RegistryClient`` or the in-memory test double).
        ``None`` disables Layers 1–2's data-driven choices (template defaults
        are kept) — the recommender still returns a valid workflow.
    dataset_uri : str, optional
        Optional data-probe hook (currently unused; kept for signature
        stability with the Part IV sketch).

    Returns
    -------
    Recommendation
        The ordered workflow + per-layer rationale + ``recommender_version``.
    """
    scope = _scope_for_goal(goal, conditions)
    # Thread goal through for the scope inference the inner layers re-run.
    cond = dict(conditions or {})
    cond.setdefault("goal", goal)

    # Layer 0 — template select.
    workflow, template = select_modules(goal, cond)
    rationale: Dict[str, Any] = {
        "goal": goal,
        "conditions": conditions,
        "template_id": template.template_id if template else None,
        "template_version": (template.template_version if template else None),
    }

    # Layer 1 — resolve choice points (cohort history).
    workflow, choice_rationale = resolve_choice_points(
        workflow, cond, db_client, dataset_uri
    )
    rationale["choice_points"] = choice_rationale

    # Layer 2 — data-driven parameters.
    workflow, param_rationale = recommend_parameters(workflow, cond, db_client)
    rationale["parameters"] = param_rationale

    # Layer 3 — validate + order. Fall back to the template default order if a
    # rewrite produced an unorderable graph (advisory: never raise at the user).
    try:
        workflow = topological_order(workflow, scope)
        rationale["validated"] = True
    except ValueError as exc:
        logger.warning(
            f"recommender: {exc}; reverting to the template default order."
        )
        default_wf = list(template.resolve()) if template else []
        workflow = default_wf
        rationale["validated"] = False
        rationale["validation_error"] = str(exc)

    return Recommendation(
        workflow=workflow,
        scope=scope,
        template_id=template.template_id if template else None,
        recommender_version=RECOMMENDER_VERSION,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Persistence + operator-edit feedback (best-effort, advisory).
# ---------------------------------------------------------------------------
def persist_recommendation(
    registry: Any,
    recommendation: Recommendation,
    *,
    analysis_run_id: Optional[str] = None,
    acquisition_run_id: Optional[str] = None,
    taxon_id: Optional[str] = None,
) -> Optional[dict]:
    """Persist a recommended spec to the registry (best-effort, non-blocking).

    Writes an ``analysis_run`` row carrying ``recommended_spec`` (the ordered
    workflow + rationale) and ``recommender_version`` — the first-class fields
    the registry schema already exposes (mirroring how WP-9 attaches a run's
    spec), so no migration is invented. Returns the server's response, or
    ``None`` if there is no registry or the write failed. A registry hiccup
    never breaks the recommendation.
    """
    if registry is None:
        return None
    spec = {
        "scope": recommendation.scope.value,
        "template_id": recommendation.template_id,
        "workflow": [
            {"module": _step_name(s), "parameters": _step_params(s)}
            for s in recommendation.workflow
        ],
        "rationale": recommendation.rationale,
    }
    fields: Dict[str, Any] = {
        "kind": "recommendation",
        "status": "recommended",
        "recommender_version": recommendation.recommender_version,
        "recommended_spec": spec,
    }
    if analysis_run_id is not None:
        fields["id"] = analysis_run_id
    if acquisition_run_id is not None:
        fields["acquisition_run_id"] = acquisition_run_id
    if taxon_id is not None:
        fields["extra"] = {"taxon_id": taxon_id}
    return _safe(
        lambda: registry.log_analysis(**fields),
        "persist recommended_spec",
        None,
    )


def capture_operator_feedback(
    registry: Any,
    recommendation: Recommendation,
    operator_workflow: Workflow,
    *,
    analysis_run_id: Optional[str] = None,
    acquisition_run_id: Optional[str] = None,
    used_in_final: bool = True,
) -> Optional[dict]:
    """Record the operator's edits to a recommended spec as a feedback row.

    Diffs the operator's final workflow against the recommended one (reusing the
    WP-TEMPLATE ``compute_config_diff`` — the same ``{param_overrides,
    modules_added, modules_removed}`` shape the deviation log already speaks) and
    POSTs it as a ``feedback`` row (``recommender_edit`` = the diff). This is the
    recommender's eval signal (acceptance / edit rate). Best-effort; returns the
    server response or ``None``.
    """
    if registry is None:
        return None
    diff = tmpl.compute_config_diff(
        list(operator_workflow), list(recommendation.workflow)
    )
    edited = bool(
        diff.get("param_overrides")
        or diff.get("modules_added")
        or diff.get("modules_removed")
    )
    fields: Dict[str, Any] = {
        "operator_decision": "edited" if edited else "accepted",
        "recommender_edit": diff,
        "used_in_final": used_in_final,
    }
    if analysis_run_id is not None:
        fields["analysis_run_id"] = analysis_run_id
    if acquisition_run_id is not None:
        fields["acquisition_run_id"] = acquisition_run_id
    return _safe(
        lambda: registry.create("feedback", **fields),
        "capture operator feedback",
        None,
    )
