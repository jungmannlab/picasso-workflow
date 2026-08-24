#!/usr/bin/env python
"""Declarative metadata for every analysis module.

Each module that the workflow can run is described here by a :class:`ModuleSpec`
capturing three things: its data dependencies (``requires`` / ``provides``), its
relationship to the ``picasso`` library (:class:`PicassoRelation` +
``picasso_symbol`` + ``outpost``), and the workflow scopes it is valid in
(:class:`Scope`). The specs are collected into :data:`MODULE_REGISTRY`, keyed by
module name.

Design notes
------------
* **Identity is the module name.** That is the key the runner dispatches on:
  :meth:`WorkflowRunner.run` validates requested modules against
  :class:`~picasso_workflow.util.AbstractModuleCollection`, and
  :meth:`WorkflowRunner.call_module` calls both the analysis implementation and
  every reporter via ``getattr(obj, name)``. So a spec is a property of the
  *logical* module, not of any one implementation. ``MODULE_REGISTRY`` is
  reconciled against ``AbstractModuleCollection`` by a completeness test (see
  ``tests/test_modulespec.py``), so it cannot silently drift from the contract.
* **Dependency-free on purpose.** This module imports nothing from ``picasso``,
  ``matplotlib`` or ``PyQt6`` so it can back a cheap pre-flight workflow
  validation (e.g. before submitting a cluster job) and feed docs/GUI tooling
  without dragging in the analysis or GUI stacks.

The ``requires`` / ``provides`` annotations below are a best-effort starting
point to be reconciled against the live source as modules evolve; the
completeness and vocabulary tests gate names and tokens, not the precise
correctness of every edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Scope(str, Enum):
    """A workflow type a module may be valid in."""

    SINGLE = "single"
    AGGREGATION = "aggregation"


class PicassoRelation(str, Enum):
    """How a module relates to the ``picasso`` library."""

    WRAPS = "wraps"  # thin representative of a single picasso API call
    EXTENDS = "extends"  # builds on picasso with extra orchestration/logic
    NATIVE = "native"  # no picasso dependency (plumbing / control flow)


# ---------------------------------------------------------------------------
# Controlled capability vocabulary -- single source of truth.
# New tokens are added here via PR review; specs referencing an unknown token
# fail at construction (see ModuleSpec.__post_init__) and in CI.
# ---------------------------------------------------------------------------
CAPABILITIES: frozenset[str] = frozenset(
    {
        # --- single-dataset data flow ---
        "raw_movie",  # loaded image stack + camera/acquisition info
        "picasso_config",  # picasso settings/config object
        "identifications",  # detected spots (pre-fit)
        "locs",  # fitted localization table (base capability)
        "locs_z",  # localizations with z (after zfit)
        "locs_undrifted",  # drift-corrected localizations
        "drift",  # estimated drift trace
        "density",  # per-localization local density
        "clusters",  # cluster labels + cluster properties
        "picks",  # picked localization groups (gold/similar/structures)
        "mask",  # cell / density mask
        "nn_distances",  # nearest-neighbour distance distribution
        "binding_kinetics",  # qPAINT / binding-event results
        "resolution",  # resolution estimate(s) (FRC / decorr / autocorr)
        "ripleys_k",  # Ripley's K spatial statistics
        "protein_interactions",  # protein-protein interaction metrics
        "cluster_motifs",  # binary-barcode / motif results from molint DBSCAN
        "spinna_results",  # SPINNA stoichiometry results
        "labeling_efficiency",  # labeling-efficiency estimate
        "render_image",  # rendered super-resolution image
        "brightfield_image",  # processed brightfield/overview image
        "dataset_summary",  # per-dataset summary statistics
        "report_items",  # items appended to the report (side-effect output)
        "saved_dataset",  # persisted result on disk
        # --- aggregation / multi-channel data flow ---
        "dataset_collection",  # set of single-dataset results gathered to aggregate
        "pooled_locs",  # localizations pooled across datasets
        "channel_locs",  # per-channel localizations (multi-channel)
        "combined_locs",  # channels combined into one dataset (e.g. RESI)
    }
)


@dataclass(frozen=True)
class ModuleSpec:
    """Declarative metadata for one analysis module.

    Parameters
    ----------
    name : str
        Module name; must match a method of
        :class:`~picasso_workflow.util.AbstractModuleCollection`.
    requires, provides, optional : frozenset[str]
        Capability tokens (from :data:`CAPABILITIES`) the module needs (AND
        semantics), makes available afterward, and consumes only if present.
    role : str | None
        Slot name for mutually exclusive alternatives (e.g. ``"undrift"``,
        ``"clusterer"``, ``"loader"``).
    after : frozenset[str]
        Explicit ordering escape hatch: names of modules that must precede this
        one, for control-flow cases capabilities cannot express.
    relation : PicassoRelation
        Relationship to ``picasso``.
    picasso_symbol : str | None
        For ``WRAPS``/``EXTENDS``, the wrapped/extended picasso entry point
        (e.g. ``"picasso.aim.aim"``). Required when ``relation`` is ``WRAPS``.
    outpost : bool
        Orthogonal flag: implementation currently lives in ``picasso_outpost``;
        a migration candidate for core ``picasso``.
    scopes : frozenset[Scope]
        Non-empty set of workflow scopes the module is valid in.
    summary : str
        One-line human description (mirrors the contract docstring).
    params : object | None
        Reserved for the future per-module parameter schema (see proposal
        Sec. 10); ``None`` until that follow-up lands.
    """

    name: str
    # --- data dependencies ---
    requires: frozenset[str] = frozenset()
    provides: frozenset[str] = frozenset()
    role: str | None = None
    optional: frozenset[str] = frozenset()
    after: frozenset[str] = frozenset()
    # --- picasso relation ---
    relation: PicassoRelation = PicassoRelation.NATIVE
    picasso_symbol: str | None = None
    outpost: bool = False
    # --- workflow scope ---
    scopes: frozenset[Scope] = frozenset({Scope.SINGLE})
    # --- docs ---
    summary: str = ""
    # --- reserved for the parameter-schema follow-up (proposal Sec. 10) ---
    params: object | None = None

    def __post_init__(self):
        if not self.scopes:
            raise ValueError(f"{self.name}: scopes must be non-empty")
        unknown = (
            self.requires | self.provides | self.optional
        ) - CAPABILITIES
        if unknown:
            raise ValueError(
                f"{self.name}: unknown capability tokens {sorted(unknown)}"
            )
        if self.relation is PicassoRelation.WRAPS and not self.picasso_symbol:
            raise ValueError(f"{self.name}: WRAPS requires a picasso_symbol")


def _s(
    name,
    *,
    requires=(),
    provides=(),
    role=None,
    optional=(),
    after=(),
    relation=PicassoRelation.NATIVE,
    picasso_symbol=None,
    outpost=False,
    scopes=(Scope.SINGLE,),
    summary="",
):
    """Terse constructor: accepts iterables, freezes them into a ModuleSpec."""
    return ModuleSpec(
        name=name,
        requires=frozenset(requires),
        provides=frozenset(provides),
        role=role,
        optional=frozenset(optional),
        after=frozenset(after),
        relation=relation,
        picasso_symbol=picasso_symbol,
        outpost=outpost,
        scopes=frozenset(scopes),
        summary=summary,
    )


_BOTH = (Scope.SINGLE, Scope.AGGREGATION)
_SINGLE = (Scope.SINGLE,)
_AGG = (Scope.AGGREGATION,)
W = PicassoRelation.WRAPS
E = PicassoRelation.EXTENDS
N = PicassoRelation.NATIVE


# ---------------------------------------------------------------------------
# The registry. One entry per module in AbstractModuleCollection (58 total).
# ---------------------------------------------------------------------------
_SPECS = [
    # --- plumbing / control flow -------------------------------------------
    _s(
        "dummy_module",
        relation=N,
        scopes=_BOTH,
        summary="Do nothing; placeholder to disable a module without renumbering.",
    ),
    _s(
        "analysis_documentation",
        provides=["report_items"],
        relation=N,
        scopes=_BOTH,
        summary="Document where and how the analysis is being performed.",
    ),
    _s(
        "conditional_branch",
        relation=N,
        scopes=_BOTH,
        summary="Execute different sub-module sequences based on a condition.",
    ),
    _s(
        "manual",
        relation=N,
        scopes=_BOTH,
        summary="Handle a manual step that waits for user-provided files.",
    ),
    _s(
        "pairwise_module_executor",
        relation=N,
        scopes=_AGG,
        summary="Call another module as a sub-module for all channel pairs.",
    ),
    _s(
        "random_val",
        relation=N,
        scopes=_BOTH,
        summary="Generate a random value and test plot for debugging.",
    ),
    # --- loaders ------------------------------------------------------------
    _s(
        "load_dataset_movie",
        provides=["raw_movie"],
        role="loader",
        relation=W,
        picasso_symbol="picasso.io.load_movie",
        scopes=_SINGLE,
        summary="Load a DNA-PAINT movie dataset in a picasso-supported format.",
    ),
    _s(
        "load_dataset_localizations",
        # A loaded locs file is a finished product, assumed already
        # drift-corrected; surface both the base and undrifted capability so
        # downstream analysis (which requires locs_undrifted) validates.
        provides=["locs", "locs_undrifted"],
        role="loader",
        relation=W,
        picasso_symbol="picasso.io.load_locs",
        scopes=_SINGLE,
        summary="Load a DNA-PAINT localizations dataset in a picasso format.",
    ),
    _s(
        "convert_zeiss_movie",
        provides=["raw_movie"],
        role="loader",
        relation=E,
        scopes=_SINGLE,
        summary="Convert a DNA-PAINT movie into picasso-supported .raw.",
    ),
    _s(
        "load_picassoconfig",
        provides=["picasso_config"],
        relation=N,
        scopes=_SINGLE,
        summary="Load a specific picasso configuration file.",
    ),
    # --- identify / localize / refine --------------------------------------
    _s(
        "identify",
        requires=["raw_movie"],
        provides=["identifications"],
        optional=["picasso_config"],
        relation=W,
        picasso_symbol="picasso.localize.identify",
        scopes=_SINGLE,
        summary="Identify localization sites in a loaded movie.",
    ),
    _s(
        "localize",
        requires=["identifications", "raw_movie"],
        provides=["locs"],
        optional=["picasso_config"],
        relation=W,
        picasso_symbol="picasso.localize.fit",
        scopes=_SINGLE,
        summary="Localize the spots previously identified.",
    ),
    _s(
        "zfit",
        requires=["locs"],
        provides=["locs_z"],
        relation=W,
        picasso_symbol="picasso.zfit.zfit",
        scopes=_SINGLE,
        summary="Fit z coordinates of localized spots via astigmatic calibration.",
    ),
    _s(
        "filter_locs",
        requires=["locs"],
        provides=["locs"],
        relation=E,
        scopes=_BOTH,
        summary="Filter localizations to a min-max range of a metric.",
    ),
    _s(
        "filter_transient_binding",
        requires=["locs"],
        provides=["locs"],
        relation=E,
        scopes=_BOTH,
        summary="Filter molecule positions for transient binding.",
    ),
    _s(
        "link_locs",
        requires=["locs"],
        provides=["locs"],
        relation=W,
        picasso_symbol="picasso.postprocess.link",
        scopes=_BOTH,
        summary="Link localizations across frames.",
    ),
    # --- drift correction (role: undrift) ----------------------------------
    _s(
        "undrift_rcc",
        requires=["locs"],
        provides=["locs_undrifted", "drift"],
        role="undrift",
        relation=W,
        picasso_symbol="picasso.postprocess.undrift",
        scopes=_SINGLE,
        summary="Undrift localized data using redundant cross-correlation (RCC).",
    ),
    _s(
        "undrift_aim",
        requires=["locs"],
        provides=["locs_undrifted", "drift"],
        role="undrift",
        relation=W,
        picasso_symbol="picasso.aim.aim",
        scopes=_SINGLE,
        summary="Undrift localized data using the AIM algorithm.",
    ),
    _s(
        "undrift_rsso",
        requires=["locs"],
        provides=["locs_undrifted", "drift"],
        role="undrift",
        relation=E,
        outpost=True,
        scopes=_SINGLE,
        summary="Undrift localized data using iterative RSSO drift correction.",
    ),
    _s(
        "undrift_from_picked",
        requires=["locs", "picks"],
        provides=["locs_undrifted", "drift"],
        role="undrift",
        relation=E,
        scopes=_SINGLE,
        summary="Undrift using picked localizations.",
    ),
    # --- rendering / brightfield -------------------------------------------
    _s(
        "render",
        requires=["locs_undrifted"],
        provides=["render_image", "report_items"],
        relation=W,
        picasso_symbol="picasso.render.plot_scene",
        scopes=_BOTH,
        summary="Render localizations on the full FOV and a center-of-mass zoom.",
    ),
    _s(
        "export_brightfield",
        optional=["raw_movie"],
        provides=["brightfield_image", "report_items"],
        relation=E,
        scopes=_SINGLE,
        summary="Open single-plane tiff image(s) and save as PNG with contrast.",
    ),
    # --- density / clustering (role: clusterer) ----------------------------
    _s(
        "density",
        requires=["locs_undrifted"],
        provides=["density"],
        relation=W,
        picasso_symbol="picasso.postprocess.compute_local_density",
        scopes=_BOTH,
        summary="Calculate the local localization density.",
    ),
    _s(
        "dbscan",
        requires=["locs_undrifted"],
        provides=["clusters"],
        role="clusterer",
        relation=W,
        picasso_symbol="picasso.clusterer.dbscan",
        scopes=_BOTH,
        summary="Cluster localizations using DBSCAN.",
    ),
    _s(
        "hdbscan",
        requires=["locs_undrifted"],
        provides=["clusters"],
        role="clusterer",
        relation=E,
        scopes=_BOTH,
        summary="Cluster localizations using HDBSCAN.",
    ),
    _s(
        "smlm_clusterer",
        requires=["locs_undrifted"],
        provides=["clusters"],
        role="clusterer",
        relation=W,
        picasso_symbol="picasso.clusterer.cluster",
        scopes=_BOTH,
        summary="Cluster localizations using the SMLM clusterer.",
    ),
    _s(
        "gaussian_mixture_cluster",
        requires=["locs_undrifted"],
        provides=["clusters"],
        role="clusterer",
        relation=W,
        picasso_symbol="picasso.g5m.g5m",
        scopes=_BOTH,
        summary="Cluster localizations using Gaussian mixture models.",
    ),
    # --- nearest-neighbour / CSR -------------------------------------------
    _s(
        "nneighbor",
        requires=["locs_undrifted"],
        provides=["nn_distances", "report_items"],
        relation=E,
        scopes=_BOTH,
        summary="Compute nearest-neighbour distances.",
    ),
    _s(
        "fit_csr",
        requires=["nn_distances"],
        provides=["report_items"],
        relation=E,
        scopes=_BOTH,
        summary="Fit a complete-spatial-randomness model to nearest neighbours.",
    ),
    # --- quality metrics / resolution --------------------------------------
    _s(
        "summarize_dataset",
        requires=["locs_undrifted"],
        provides=["dataset_summary", "report_items"],
        relation=E,
        scopes=_SINGLE,
        summary="Summarize a dataset using various quality-metric methods.",
    ),
    _s(
        "binding_event_analysis",
        requires=["locs_undrifted"],
        provides=["binding_kinetics", "report_items"],
        relation=E,
        scopes=_SINGLE,
        summary="Evaluate binding events following Steen et al.",
    ),
    _s(
        "resolution_analysis",
        requires=["locs_undrifted"],
        provides=["resolution", "report_items"],
        relation=E,
        scopes=_SINGLE,
        summary="Estimate spatial resolution via point-pattern autocorrelation.",
    ),
    _s(
        "resolution_frc_spatial",
        requires=["locs_undrifted"],
        provides=["resolution", "report_items"],
        relation=E,
        scopes=_SINGLE,
        summary="Calculate resolution using a spatial FRC approach.",
    ),
    # --- Ripley's K ---------------------------------------------------------
    _s(
        "ripleysk",
        requires=["locs_undrifted"],
        provides=["ripleys_k", "report_items"],
        relation=E,
        scopes=_BOTH,
        summary="Compute Ripley's K spatial statistics for the dataset.",
    ),
    _s(
        "ripleysk2",
        requires=["locs_undrifted"],
        provides=["ripleys_k", "report_items"],
        relation=E,
        scopes=_BOTH,
        summary="Compute Ripley's K statistics (second implementation).",
    ),
    _s(
        "ripleysk_average",
        requires=["ripleys_k"],
        provides=["ripleys_k", "report_items"],
        relation=E,
        scopes=_AGG,
        summary="Average Ripley's K curves across datasets.",
    ),
    _s(
        "ripleysk_average2",
        requires=["ripleys_k"],
        provides=["ripleys_k", "report_items"],
        relation=E,
        scopes=_AGG,
        summary="Average Ripley's K curves across datasets (second variant).",
    ),
    # --- protein interactions ----------------------------------------------
    _s(
        "protein_interactions",
        requires=["locs_undrifted"],
        provides=["protein_interactions", "report_items"],
        relation=E,
        scopes=_BOTH,
        summary="Quantify protein-protein interactions from the localizations.",
    ),
    _s(
        "protein_interactions_average",
        requires=["protein_interactions"],
        provides=["protein_interactions", "report_items"],
        relation=E,
        scopes=_AGG,
        summary="Average protein-interaction metrics across datasets.",
    ),
    _s(
        "interaction_graph",
        requires=["protein_interactions"],
        provides=["report_items"],
        relation=E,
        scopes=_BOTH,
        summary="Plot the target-interaction graph.",
    ),
    # --- masks / molecular-interactions workflow ---------------------------
    _s(
        "create_mask",
        requires=["locs_undrifted"],
        provides=["mask"],
        role="mask",
        relation=E,
        scopes=_BOTH,
        summary="Calculate a cell mask (Susanne's original DC-Atlas implementation).",
    ),
    _s(
        "create_mask2",
        requires=["locs_undrifted"],
        provides=["mask"],
        role="mask",
        relation=E,
        scopes=_BOTH,
        summary="Calculate a cell mask (Rafal's DC-Atlas v3 implementation).",
    ),
    _s(
        "refine_mask_by_density",
        requires=["mask"],
        optional=["density"],
        provides=["mask", "report_items"],
        relation=E,
        scopes=_BOTH,
        summary="Analyse and refine a previously created mask by density.",
    ),
    _s(
        "dbscan_molint",
        requires=["locs_undrifted"],
        provides=["clusters"],
        role="clusterer",
        relation=E,
        scopes=_BOTH,
        summary="Run DBSCAN for the molecular-interactions workflow.",
    ),
    _s(
        "CSR_sim_in_mask",
        requires=["mask"],
        provides=["clusters", "report_items"],
        relation=E,
        scopes=_BOTH,
        summary="Simulate CSR within a density mask and run DBSCAN on it.",
    ),
    _s(
        "find_cluster_motifs",
        requires=["clusters"],
        provides=["cluster_motifs", "report_items"],
        relation=E,
        scopes=_BOTH,
        summary="Analyse the binary barcode results of the molint DBSCAN.",
    ),
    _s(
        "plot_densities",
        requires=["density"],
        provides=["report_items"],
        relation=E,
        scopes=_AGG,
        summary="Aggregate and plot densities and cell areas across datasets.",
    ),
    # --- picking ------------------------------------------------------------
    _s(
        "find_gold",
        requires=["locs_undrifted"],
        provides=["picks"],
        relation=E,
        scopes=_SINGLE,
        summary="Find localizations from gold beads via blinking kinetics.",
    ),
    _s(
        "find_similar",
        requires=["locs_undrifted"],
        provides=["picks"],
        relation=E,
        scopes=_SINGLE,
        summary="Pick-similar in nlocs/rmsd space within specified limits.",
    ),
    _s(
        "find_structures",
        requires=["clusters"],
        provides=["picks"],
        relation=E,
        scopes=_SINGLE,
        summary="Pick-similar on clusters in nlocs/rmsd space.",
    ),
    # --- SPINNA / labeling efficiency --------------------------------------
    _s(
        "spinna",
        requires=["locs_undrifted"],
        provides=["spinna_results", "report_items"],
        relation=E,
        outpost=True,
        scopes=_BOTH,
        summary="Run a direct SPINNA batch analysis.",
    ),
    _s(
        "spinna_batch",
        requires=["locs_undrifted"],
        provides=["spinna_results", "report_items"],
        relation=E,
        outpost=True,
        scopes=_BOTH,
        summary="Run a SPINNA batch analysis from a pre-existing config file.",
    ),
    _s(
        "labeling_efficiency_analysis",
        requires=["locs_undrifted"],
        provides=["labeling_efficiency", "report_items"],
        relation=E,
        outpost=True,
        scopes=_BOTH,
        summary="Analyse labeling efficiency via a 3-component SPINNA analysis.",
    ),
    # --- persistence --------------------------------------------------------
    _s(
        "save_single_dataset",
        requires=["locs", "locs_undrifted"],
        provides=["saved_dataset"],
        relation=W,
        picasso_symbol="picasso.io.save_locs",
        scopes=_SINGLE,
        summary="Save the locs and info of a single dataset.",
    ),
    # --- aggregation: collection / channels / persistence ------------------
    _s(
        "load_datasets_to_aggregate",
        # Aggregation consumes saved single-dataset results, which are already
        # drift-corrected (save_single_dataset sits downstream of undrift), so
        # locs / locs_undrifted are available for both-scope analysis modules.
        provides=[
            "dataset_collection",
            "pooled_locs",
            "channel_locs",
            "locs",
            "locs_undrifted",
        ],
        role="loader",
        relation=N,
        scopes=_AGG,
        summary="Load the results of single-dataset workflows for aggregation.",
    ),
    _s(
        "align_channels",
        requires=["channel_locs"],
        provides=["channel_locs", "report_items"],
        relation=E,
        scopes=_AGG,
        summary="Align multiple channels to each other (aggregation workflow).",
    ),
    _s(
        "register_channels",
        requires=["channel_locs"],
        provides=["channel_locs", "report_items"],
        relation=E,
        picasso_symbol=(
            "picasso.registration.calibrate_channel_registration_from_beads"
        ),
        scopes=_AGG,
        summary="Register channels via bead-based affine/projective/polynomial"
        " transforms (aggregation workflow).",
    ),
    _s(
        "combine_channels",
        requires=["channel_locs"],
        provides=["combined_locs"],
        relation=E,
        scopes=_AGG,
        summary="Combine multiple channels into one dataset (e.g. for RESI).",
    ),
    _s(
        "save_datasets_aggregated",
        requires=["dataset_collection"],
        provides=["saved_dataset"],
        relation=N,
        scopes=_AGG,
        summary="Save data of all single-dataset workflows in an aggregation.",
    ),
]

MODULE_REGISTRY: dict[str, ModuleSpec] = {}
for _spec in _SPECS:
    if _spec.name in MODULE_REGISTRY:
        raise ValueError(f"duplicate module spec: {_spec.name}")
    MODULE_REGISTRY[_spec.name] = _spec
del _spec


# ---------------------------------------------------------------------------
# Pre-flight workflow validation.
# ---------------------------------------------------------------------------
def _step_name(step):
    """Return the module name from a workflow step.

    Accepts the runner's native ``(name, parameters)`` tuple, a bare module
    name, or a ``{"module": ...}`` / ``{"name": ...}`` mapping.
    """
    if isinstance(step, str):
        return step
    if isinstance(step, dict):
        return step.get("module") or step.get("name")
    # (name, parameters) tuple/list, as used by WorkflowRunner.
    return step[0]


def _initial_available(scope):
    """Capabilities available before any step runs, per scope.

    Both scopes start empty: a ``loader`` module must provide the base
    capabilities (``raw_movie``/``locs`` for single, ``dataset_collection``
    for aggregation). Kept as a hook for future per-scope seeding.
    """
    return set()


def validate_workflow(steps, scope, registry=None):
    """Check an ordered workflow against the module registry.

    A pre-flight, execution-free check intended to run before a workflow is
    submitted (e.g. to a cluster). It does not import or call any module.

    Parameters
    ----------
    steps : iterable
        Ordered workflow steps, each a ``(module_name, parameters)`` tuple (the
        :class:`~picasso_workflow.workflow.WorkflowRunner` format), a bare
        module-name string, or a ``{"module": ...}`` mapping.
    scope : Scope or str
        The workflow scope (``"single"`` or ``"aggregation"``).
    registry : dict[str, ModuleSpec], optional
        Registry to validate against. Defaults to :data:`MODULE_REGISTRY`.

    Returns
    -------
    list[str]
        Human-readable error messages, one per problem, prefixed with the step
        index. Empty if the workflow is valid. Checks, in order per step:
        module is registered, its ``scopes`` contains ``scope``, its
        ``requires`` are satisfied by capabilities provided earlier, and any
        ``after`` ordering constraints hold. ``optional`` inputs are never
        required. Mutually-exclusive ``role`` collisions are intentionally not
        flagged here (an advisory recommender concern, not a correctness error).
    """
    if registry is None:
        registry = MODULE_REGISTRY
    if isinstance(scope, str):
        scope = Scope(scope)

    available = _initial_available(scope)
    prior_names: list[str] = []
    errors: list[str] = []
    for i, step in enumerate(steps):
        name = _step_name(step)
        spec = registry.get(name)
        if spec is None:
            errors.append(f"[{i}] unknown module '{name}'")
            prior_names.append(name)
            continue
        if scope not in spec.scopes:
            valid = ", ".join(sorted(s.value for s in spec.scopes))
            errors.append(
                f"[{i}] {spec.name} not valid in {scope.value} workflow "
                f"(valid in: {valid})"
            )
        missing = spec.requires - available
        if missing:
            errors.append(
                f"[{i}] {spec.name} missing required inputs: {sorted(missing)}"
            )
        for required_predecessor in sorted(spec.after):
            if required_predecessor not in prior_names:
                errors.append(
                    f"[{i}] {spec.name} must come after "
                    f"'{required_predecessor}'"
                )
        available |= spec.provides
        prior_names.append(name)
    return errors
