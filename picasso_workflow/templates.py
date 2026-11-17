#!/usr/bin/env python
"""Versioned analysis templates + deviation logging (WP-TEMPLATE-CONFORMANCE).

Standardize analysis **without** mandating one workflow (A19 ratified as C30).
A *template* is a VERSIONED, loadable recipe: it resolves to a concrete default
(an ordered module list + params) and carries a stable ``template_version`` so
that every later run can be compared against the exact template content it
started from. Running a template is never gated; the run simply records how it
*differed* from the template — a ``config_diff`` plus a derived
``conformance`` — so deviations are cheap, allowed, and attributable.

This module is **additive / opt-in**. The existing "assemble any workflow at
will" path is untouched: templates wrap the existing recipes in
``standard_singledataset_workflows`` / ``standard_aggregation_workflows`` as
registered defaults, and the deviation log is a pure side-effect of running
(written into the existing ``qc.json`` sidecar) — it never blocks or alters a
run.

Phase map (this module implements Phases 1 and 2 only)
------------------------------------------------------
* **Phase 1 — versioned templates.** :data:`TEMPLATE_REGISTRY` registers the
  existing recipes as :class:`Template` objects carrying a ``template_id``, a
  content-hash ``template_version``, a ``question_type`` tag, and OPTIONAL
  coarse descriptor-match tags (ratified A2 axes only; NO leaf vocab). Each
  resolves to a concrete default via :meth:`Template.resolve`.
* **Phase 2 — deviation logging.** :func:`compute_config_diff` /
  :func:`derive_conformance` mechanically diff a run's resolved workflow against
  the template default; :func:`template_conformance_record` assembles the record
  and :func:`write_conformance_to_qc` appends it to the qc.json sidecar.
* **Phase 3 (NOT built here)** — registry-backed deviation sink. Hook:
  :func:`template_conformance_record` returns a plain dict ready to POST to the
  ``picasso-registry`` service once WP-6 is merged. See the module-level
  ``PHASE_3_HOOK`` note below.
* **Phase 4 (NOT built here)** — recommender consumption (Layer 2/3 node
  cascade + auto-selection; folds into WP-10/WP-11). The deviation corpus this
  module produces is that work's training input.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import picasso_workflow.standard_aggregation_workflows as saw
import picasso_workflow.standard_singledataset_workflows as ssw
from picasso_workflow.modulespec import Scope, _step_name, _step_params

# A workflow step as the runner / recipes represent it: (module_name, params).
Step = Tuple[str, Dict[str, Any]]
Workflow = List[Step]

# ---------------------------------------------------------------------------
# Canonicalization: what the content hash (and the diff) see.
#
# Template content is "the ordered module list + params", but a few parameter
# values are per-RUN placeholders injected by the recipe (the dataset filepath,
# save targets, map/result-reference directives). If those rode into the hash,
# two runs of the SAME template on different files would carry different
# template_versions and every config_diff would be uninterpretable — exactly
# the failure the brief warns against ("version from the first run"). So the
# canonical form NORMALIZES those run-specific holes to a stable sentinel.
#
# This normalization is applied symmetrically to BOTH sides of a diff, so it
# never hides a genuine user override (which lands in some OTHER key).
# ---------------------------------------------------------------------------
_PLACEHOLDER = "<run-specific>"

# Parameter keys (dotted paths into a step's params) whose values are filled in
# per run by the recipe / runner, not chosen by the user. Normalized out of the
# canonical form so they affect neither the version hash nor the diff.
_RUN_SPECIFIC_KEYS: frozenset[str] = frozenset(
    {
        "filename",  # dataset path / per-channel save target
        "filepath",  # dataset path
        "filepaths",  # aggregation: per-channel dataset paths
        "tags",  # aggregation: per-channel tags
    }
)


def _is_directive(value: Any) -> bool:
    """True for a recipe result-reference / map directive tuple.

    The recipes encode "fill this in from a prior result / per-channel map" as a
    tuple whose first element is a ``$``-prefixed command string (e.g.
    ``("$$map", "#tags", None)`` or ``("$get_previous_module_result", ...)``).
    Those are run-plumbing, not user parameter choices.
    """
    return (
        isinstance(value, (tuple, list))
        and len(value) >= 1
        and isinstance(value[0], str)
        and value[0].startswith("$")
    )


def _canonicalize(value: Any, key: Optional[str] = None) -> Any:
    """Recursively normalize a param value to its content-comparable form.

    * run-specific keys → the ``<run-specific>`` sentinel,
    * recipe directives (``$``-prefixed tuples) → the sentinel,
    * tuples → lists (JSON has no tuples; keeps hashing stable),
    * dicts → key-sorted dicts (handled by the JSON encoder at serialize time).
    """
    if key is not None and key in _RUN_SPECIFIC_KEYS:
        return _PLACEHOLDER
    if _is_directive(value):
        return _PLACEHOLDER
    if isinstance(value, dict):
        return {k: _canonicalize(v, k) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonicalize(v) for v in value]
    return value


def canonical_workflow(workflow: Workflow) -> List[Dict[str, Any]]:
    """Return a workflow in canonical, content-comparable form.

    A list (order preserved — module order is semantically significant) of
    ``{"module": name, "params": <canonicalized params>}`` dicts. This is the
    single representation that both the version hash and the diff consume, so
    they can never disagree about what "the content" is.
    """
    out: List[Dict[str, Any]] = []
    for step in workflow:
        name = _step_name(step)
        params = _step_params(step)
        out.append(
            {
                "module": name,
                "params": _canonicalize(copy.deepcopy(params)),
            }
        )
    return out


def _canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def content_hash(workflow: Workflow) -> str:
    """Stable content hash of a resolved workflow.

    SHA-256 over the canonical JSON of :func:`canonical_workflow`, truncated to
    16 hex chars. Stable because the canonical form is key-sorted and strips
    per-run placeholders; it changes **iff** the template's module list or its
    (non-run-specific) params change. Order-of-modules IS content and is
    preserved, so reordering modules changes the hash.
    """
    digest = hashlib.sha256(
        _canonical_json(canonical_workflow(workflow)).encode("utf-8")
    ).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# Phase 1 — versioned templates.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Template:
    """A versioned, loadable analysis template.

    Parameters
    ----------
    template_id : str
        Stable identifier (recipe name), e.g. ``"minimal"``.
    scope : Scope
        Workflow scope the template's default is valid in.
    question_type : str
        Coarse goal tag (e.g. ``"localize"``, ``"channel-align"``). NOT an A2
        descriptor leaf — a question/goal label.
    builder : callable
        Returns the concrete default workflow (ordered module list + params)
        when called with ``**builder_kwargs``. For a single-scope template this
        is the recipe function; it returns a list of ``(name, params)`` steps.
    builder_kwargs : dict
        Default arguments passed to ``builder`` to materialize the default.
    descriptor_tags : dict
        OPTIONAL coarse descriptor-match tags, ratified A2 axes only
        (``sample_type`` / ``modality`` / ``dimensionality`` — NO leaf vocab).
        Empty by default; a template matches broadly when unset.
    """

    template_id: str
    scope: Scope
    question_type: str
    builder: Callable[..., Any]
    builder_kwargs: Dict[str, Any] = field(default_factory=dict)
    descriptor_tags: Dict[str, Any] = field(default_factory=dict)

    def resolve(self) -> Workflow:
        """Materialize the concrete default workflow (ordered steps).

        Single-scope recipes return the step list directly. Aggregation
        recipes return a ``{single_dataset_modules, ...,
        aggregation_modules}`` mapping; the template default is the
        **whole** pipeline — the per-dataset ``single_dataset_modules``
        followed by the ``aggregation_modules`` — so a change to *either* half
        (e.g. the per-dataset ``box_size`` or an added aggregation step) is
        visible to both ``template_version`` and ``config_diff``. (Diffing only
        the aggregation half would make per-dataset deviations invisible, since
        an aggregation run has no separate single-dataset template run to catch
        them.) The two phases' module names are disjoint in the shipped
        recipes, so the flat concatenation is unambiguous; a run diffed against
        this default must likewise supply its single+aggregation steps.
        """
        built = self.builder(**self.builder_kwargs)
        if isinstance(built, dict):
            return list(built.get("single_dataset_modules", [])) + list(
                built.get("aggregation_modules", [])
            )
        return list(built)

    @property
    def template_version(self) -> str:
        """Content-hash version of the resolved default (see :func:`content_hash`)."""
        return content_hash(self.resolve())


# Placeholder dataset paths used only to MATERIALIZE a default for hashing /
# diffing. They are run-specific keys, so _canonicalize normalizes them away —
# the concrete string here never reaches the hash.
_PH_FILE = "<template-default>"
_PH_FILES = ["<template-default>"]


_TEMPLATES: List[Template] = [
    Template(
        template_id="minimal",
        scope=Scope.SINGLE,
        question_type="localize",
        builder=ssw.minimal,
        builder_kwargs={"filepath": _PH_FILE},
    ),
    Template(
        template_id="picasso_native",
        scope=Scope.SINGLE,
        question_type="localize",
        builder=ssw.picasso_native,
        builder_kwargs={"filepath": _PH_FILE},
    ),
    Template(
        template_id="minimal_channel_align",
        scope=Scope.AGGREGATION,
        question_type="channel-align",
        builder=saw.minimal_channel_align,
        builder_kwargs={"filepaths": _PH_FILES},
    ),
    Template(
        template_id="cluster_aggregation",
        scope=Scope.AGGREGATION,
        question_type="cluster-undrift-aggregate",
        builder=saw.cluster_aggregation,
        builder_kwargs={"filepaths": _PH_FILES},
    ),
]

TEMPLATE_REGISTRY: Dict[str, Template] = {}
for _t in _TEMPLATES:
    if _t.template_id in TEMPLATE_REGISTRY:
        raise ValueError(f"duplicate template id: {_t.template_id}")
    TEMPLATE_REGISTRY[_t.template_id] = _t
del _t


def get_template(template_id: str) -> Optional[Template]:
    """Return the registered :class:`Template`, or ``None`` if unknown."""
    return TEMPLATE_REGISTRY.get(template_id)


# ---------------------------------------------------------------------------
# Phase 2 — deviation logging.
# ---------------------------------------------------------------------------
def _flatten_params(
    params: Dict[str, Any], prefix: str = ""
) -> Dict[str, Any]:
    """Flatten a (canonicalized) params dict to dotted-path leaf values.

    Nested dicts become dotted keys (``auto_netgrad.box_size``); lists/scalars
    are leaves. Operates on the canonical form, so values are already
    JSON-plain and run-specific holes are sentinels.
    """
    flat: Dict[str, Any] = {}
    for key, value in params.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_params(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


def compute_config_diff(
    run_workflow: Workflow, template_default: Workflow
) -> Dict[str, Any]:
    """Mechanically diff a run's resolved workflow vs a template default.

    Both sides are canonicalized first (so per-run filepaths etc. never read as
    deviations). The diff is computed per module position by name:

    * ``modules_added`` — module names present in the run but not the template
      (at a given position / count), in run order;
    * ``modules_removed`` — module names in the template but not the run;
    * ``param_overrides`` — for modules common to both, a map
      ``"<module>.<dotted.param>" -> {"template": <old>, "run": <new>}`` for
      every differing (or added/removed) leaf param.

    The comparison walks the two module sequences in order. A position whose
    module name matches is treated as "the same module" and its params are
    leaf-diffed; name mismatches are reported as an add + a remove. This is a
    deterministic, order-independent-of-dict-iteration computation: identical
    inputs always yield an identical diff.
    """
    run_canon = canonical_workflow(run_workflow)
    tmpl_canon = canonical_workflow(template_default)

    modules_added: List[str] = []
    modules_removed: List[str] = []
    param_overrides: Dict[str, Dict[str, Any]] = {}

    # Align by module name, position-sensitively, using a simple two-pointer
    # longest-common-subsequence-free walk: match equal names greedily; a
    # mismatch means the template module was removed and/or the run module was
    # added. We detect which by looking ahead one step.
    i = j = 0
    while i < len(run_canon) and j < len(tmpl_canon):
        rname = run_canon[i]["module"]
        tname = tmpl_canon[j]["module"]
        if rname == tname:
            _diff_params(
                rname,
                tmpl_canon[j]["params"],
                run_canon[i]["params"],
                param_overrides,
            )
            i += 1
            j += 1
            continue
        # Names differ: decide add vs remove by minimal look-ahead.
        run_has_tname = any(s["module"] == tname for s in run_canon[i + 1 :])
        tmpl_has_rname = any(s["module"] == rname for s in tmpl_canon[j + 1 :])
        if tmpl_has_rname and not run_has_tname:
            modules_removed.append(tname)
            j += 1
        elif run_has_tname and not tmpl_has_rname:
            modules_added.append(rname)
            i += 1
        else:
            # Pure substitution at this position: one added, one removed.
            modules_added.append(rname)
            modules_removed.append(tname)
            i += 1
            j += 1
    while i < len(run_canon):
        modules_added.append(run_canon[i]["module"])
        i += 1
    while j < len(tmpl_canon):
        modules_removed.append(tmpl_canon[j]["module"])
        j += 1

    return {
        "param_overrides": param_overrides,
        "modules_added": modules_added,
        "modules_removed": modules_removed,
    }


def _diff_params(
    module: str,
    tmpl_params: Dict[str, Any],
    run_params: Dict[str, Any],
    out: Dict[str, Dict[str, Any]],
) -> None:
    """Leaf-diff two (canonical) param dicts into ``out`` keyed by module.param."""
    tflat = _flatten_params(tmpl_params)
    rflat = _flatten_params(run_params)
    for key in sorted(set(tflat) | set(rflat)):
        tval = tflat.get(key, None)
        rval = rflat.get(key, None)
        t_has = key in tflat
        r_has = key in rflat
        if t_has and r_has and tval == rval:
            continue
        out[f"{module}.{key}"] = {
            "template": tval if t_has else None,
            "run": rval if r_has else None,
        }


# Conformance levels, ordered least-to-most deviation.
CONFORMANCE_EXACT = "exact"
CONFORMANCE_PARAM = "param-override"
CONFORMANCE_STRUCTURAL = "structural-deviation"
CONFORMANCE_ADHOC = "ad-hoc"


def derive_conformance(
    config_diff: Optional[Dict[str, Any]], has_template: bool
) -> str:
    """Derive the conformance level from a config_diff.

    * no template → ``ad-hoc``;
    * modules added/removed → ``structural-deviation`` (dominates params);
    * only param overrides → ``param-override``;
    * empty diff → ``exact``.
    """
    if not has_template:
        return CONFORMANCE_ADHOC
    diff = config_diff or {}
    if diff.get("modules_added") or diff.get("modules_removed"):
        return CONFORMANCE_STRUCTURAL
    if diff.get("param_overrides"):
        return CONFORMANCE_PARAM
    return CONFORMANCE_EXACT


def template_conformance_record(
    run_workflow: Workflow,
    template: Optional[Template] = None,
) -> Dict[str, Any]:
    """Assemble the deviation record for a run.

    Returns a plain, JSON-serializable dict with the four logged fields:
    ``template_id``, ``template_version`` (both ``None`` for an ad-hoc run with
    no template), ``config_diff`` (``None`` when ad-hoc), and the derived
    ``conformance``.

    PHASE_3_HOOK: this dict is exactly the payload a future registry client
    POSTs to ``picasso-registry`` (WP-6, not yet merged). Until then the MVP
    sink is the qc.json sidecar (:func:`write_conformance_to_qc`). Keeping the
    record a pure dict means the Phase-3 swap is a sink change, not a format
    change.
    """
    if template is None:
        return {
            "template_id": None,
            "template_version": None,
            "config_diff": None,
            "conformance": derive_conformance(None, has_template=False),
        }
    config_diff = compute_config_diff(run_workflow, template.resolve())
    return {
        "template_id": template.template_id,
        "template_version": template.template_version,
        "config_diff": config_diff,
        "conformance": derive_conformance(config_diff, has_template=True),
    }


# ---------------------------------------------------------------------------
# Phase 2 sink — the existing qc.json sidecar (MVP; no registry, no recommender).
#
# Mirrors picasso_workflow.qc_advisor.filter_advisor's load/merge/dump pattern:
# additive, never clobbers existing keys. The record lands under a dedicated
# top-level ``template_conformance`` section, leaving ``recommendations`` and
# every other field untouched.
# ---------------------------------------------------------------------------
def _load_qc(path: str) -> Dict[str, Any]:
    import os

    if path and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def write_conformance_to_qc(
    qc_path: str, record: Dict[str, Any]
) -> Dict[str, Any]:
    """Append a conformance ``record`` to the qc.json sidecar, additively.

    Writes under the top-level ``template_conformance`` key, creating the
    file/section as needed and preserving every existing field. Returns the
    updated qc dict. A pure side-effect of running — it never gates or alters
    the run.
    """
    qc = _load_qc(qc_path)
    qc["template_conformance"] = record
    with open(qc_path, "w", encoding="utf-8") as fh:
        json.dump(qc, fh, indent=2)
    return qc


def log_run_conformance(
    qc_path: str,
    run_workflow: Workflow,
    template: Optional[Template] = None,
) -> Dict[str, Any]:
    """Convenience: build the record and write it to the qc.json sidecar.

    Returns the record (not the whole qc dict). The one call a runner makes
    after resolving its workflow; opt-in and side-effect-only.
    """
    record = template_conformance_record(run_workflow, template)
    write_conformance_to_qc(qc_path, record)
    return record
