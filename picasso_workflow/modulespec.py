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

from picasso_workflow.picasso_set.params import (
    PICASSO_SET_PARAMS,
    PICASSO_SET_SUMMARIES,
)


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
        "fret_results",  # FRET efficiency results
        "z_calibration",  # 3D (astigmatism) z-calibration file
        "camera_calibration",  # sCMOS camera calibration file
        "spline_calibration",  # cubic-spline PSF calibration file
        "converted_file",  # localizations exported to a foreign format
        "dataset_summary",  # per-dataset summary statistics
        "report_items",  # items appended to the report (side-effect output)
        "saved_dataset",  # persisted result on disk
        # --- control flow ---
        "branches",  # per-branch results produced by the branch module
        # --- aggregation / multi-channel data flow ---
        "dataset_collection",  # set of single-dataset results gathered to aggregate
        "pooled_locs",  # localizations pooled across datasets
        "channel_locs",  # per-channel localizations (multi-channel)
        "combined_locs",  # channels combined into one dataset (e.g. RESI)
    }
)


# ---------------------------------------------------------------------------
# Capabilities that live only in process memory. Everything else is either
# encoded in the localizations themselves (and thus restored when a saved
# locs file is re-loaded on resume) or exchanged between modules via file
# paths recorded in the results dict, which survive in WorkflowRunner.yaml.
# These do not: restarting from a saved-locs checkpoint cannot restore them.
# ---------------------------------------------------------------------------
MEMORY_ONLY_CAPABILITIES: frozenset[str] = frozenset(
    {
        "raw_movie",  # AutoPicasso.movie: the loaded image stack
        "identifications",  # AutoPicasso.identifications: pre-fit spots
        "drift",  # AutoPicasso.drift: estimated drift trace
        "picasso_config",  # process-global picasso CONFIG set at load time
    }
)

# Capabilities carried by the in-memory localization state. Unlike the
# memory-only set above, these CAN be restored on resume -- by re-loading a
# saved locs file into AutoPicasso.locs (single) or AutoPicasso.channel_locs
# (channels). Which subset a given checkpoint restores depends on which
# parts it saved.
SINGLE_LOCS_CAPABILITIES: frozenset[str] = frozenset(
    {"locs", "locs_z", "locs_undrifted"}
)
CHANNEL_LOCS_CAPABILITIES: frozenset[str] = frozenset(
    {
        "channel_locs",
        "dataset_collection",
        "pooled_locs",
        "combined_locs",  # combine_channels stores into channel_locs
        # in aggregation scope the base locs tokens ride on the channel
        # state (cf. load_datasets_to_aggregate's provides)
        "locs",
        "locs_undrifted",
    }
)
LOCS_STATE_CAPABILITIES: frozenset[str] = (
    SINGLE_LOCS_CAPABILITIES | CHANNEL_LOCS_CAPABILITIES
)

# Parameter keys (dotted paths for nested dicts) that modules overwrite in
# their own parameter dict at run time (estimates, resolved output paths,
# defaults). Needed only when resuming runs recorded before the pristine
# parameter snapshot existed: those yamls persisted the overwritten values,
# which must not read as user edits. Runs recorded since compare against
# the exact snapshot and never consult this map.
RUNTIME_PARAMETER_WRITE_BACKS: dict[str, frozenset[str]] = {
    "load_dataset_movie": frozenset({"sample_movie.filename"}),
    "identify": frozenset({"min_gradient", "auto_netgrad.filename"}),
    "undrift_rcc": frozenset({"segmentation", "dimensions"}),
    "smlm_clusterer": frozenset({"basic_fa", "radius_z"}),
    "align_channels": frozenset({"align_pars.plot_dir"}),
    "labeling_efficiency_analysis": frozenset(
        {"nn_nth", "pair_distance", "labeling_uncertainty"}
    ),
}


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
    module_set : str
        Which module set the module belongs to: ``"native"`` for the classic
        picasso-workflow modules, ``"picasso"`` for the picasso-set modules
        that recapitulate native picasso 1:1 (``picasso_*`` names).
    params : object | None
        Per-module parameter schema. For picasso-set modules this is the
        ``(parameters_spec, results_spec)`` tuple from
        :mod:`picasso_workflow.picasso_set.params`, consumed by the GUI;
        ``None`` for native modules until the schema follow-up lands.
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
    # --- module set ---
    module_set: str = "native"
    # --- per-module parameter schema (filled for the picasso-set) ---
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
        if self.module_set not in ("native", "picasso"):
            raise ValueError(
                f"{self.name}: unknown module_set {self.module_set!r}"
            )
        if self.module_set == "picasso":
            if not self.name.startswith("picasso_"):
                raise ValueError(
                    f"{self.name}: picasso-set modules must be named "
                    "'picasso_*'"
                )
            if self.relation is not PicassoRelation.WRAPS:
                raise ValueError(
                    f"{self.name}: picasso-set modules must WRAP a picasso "
                    "symbol"
                )


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
    module_set="native",
    params=None,
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
        module_set=module_set,
        params=params,
    )


_BOTH = (Scope.SINGLE, Scope.AGGREGATION)
_SINGLE = (Scope.SINGLE,)
_AGG = (Scope.AGGREGATION,)
W = PicassoRelation.WRAPS
E = PicassoRelation.EXTENDS
N = PicassoRelation.NATIVE


# ---------------------------------------------------------------------------
# The registry. One entry per module in AbstractModuleCollection (59 total),
# plus the picasso-set modules (module_set="picasso", implemented in
# picasso_workflow/picasso_set/).
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
        "branch",
        provides=["branches"],
        relation=N,
        scopes=_BOTH,
        summary="Fan out into per-branch sub-workflows, then optionally re-join.",
    ),
    _s(
        "summarize_branches",
        provides=["report_items"],
        relation=N,
        scopes=_BOTH,
        summary="Summarize per-branch results as a figure (box/strip or vs arg).",
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
    _s(
        "pick_origami",
        requires=["locs_undrifted"],
        provides=["picks"],
        relation=E,
        outpost=True,
        scopes=_SINGLE,
        summary="Design-aware picking of origami structures, tolerant of "
        "missing sites.",
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
    # --- picasso-set: 1:1 recapitulation of native picasso ------------------
    # These modules mirror picasso CLI commands / library calls with the
    # exact picasso parameter names and defaults. They are implemented in
    # picasso_workflow/picasso_set/ (not in AbstractModuleCollection) and
    # carry their GUI parameter schema in `params`. Summaries come from the
    # same source (picasso_set.params) as the schemas.
    _s(
        "picasso_localize",
        requires=["raw_movie"],
        provides=["locs"],
        relation=W,
        picasso_symbol="picasso.localize.localize",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_localize"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_localize"],
    ),
    _s(
        "picasso_undrift_rcc",
        requires=["locs"],
        provides=["locs_undrifted", "drift"],
        role="undrift",
        relation=W,
        picasso_symbol="picasso.postprocess.undrift",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_undrift_rcc"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_undrift_rcc"],
    ),
    _s(
        "picasso_undrift_aim",
        requires=["locs"],
        provides=["locs_undrifted", "drift"],
        role="undrift",
        relation=W,
        picasso_symbol="picasso.aim.aim",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_undrift_aim"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_undrift_aim"],
    ),
    _s(
        "picasso_undrift_fiducials",
        requires=["locs"],
        provides=["locs_undrifted", "drift"],
        role="undrift",
        relation=W,
        picasso_symbol="picasso.postprocess.undrift_from_fiducials",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_undrift_fiducials"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_undrift_fiducials"],
    ),
    _s(
        "picasso_link",
        requires=["locs_undrifted"],
        provides=["locs"],
        relation=W,
        picasso_symbol="picasso.postprocess.link",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_link"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_link"],
    ),
    _s(
        "picasso_dark",
        requires=["locs"],
        provides=["binding_kinetics"],
        relation=W,
        picasso_symbol="picasso.postprocess.compute_dark_times",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_dark"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_dark"],
    ),
    _s(
        "picasso_groupprops",
        requires=["locs"],
        provides=["binding_kinetics", "saved_dataset"],
        relation=W,
        picasso_symbol="picasso.postprocess.groupprops",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_groupprops"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_groupprops"],
    ),
    _s(
        "picasso_density",
        requires=["locs_undrifted"],
        provides=["density"],
        relation=W,
        picasso_symbol="picasso.postprocess.compute_local_density",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_density"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_density"],
    ),
    _s(
        "picasso_pair_correlation",
        requires=["locs_undrifted"],
        provides=["report_items"],
        relation=W,
        picasso_symbol="picasso.postprocess.pair_correlation",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_pair_correlation"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_pair_correlation"],
    ),
    _s(
        "picasso_dbscan",
        requires=["locs_undrifted"],
        provides=["clusters"],
        role="clusterer",
        relation=W,
        picasso_symbol="picasso.clusterer.dbscan",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_dbscan"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_dbscan"],
    ),
    _s(
        "picasso_hdbscan",
        requires=["locs_undrifted"],
        provides=["clusters"],
        role="clusterer",
        relation=W,
        picasso_symbol="picasso.clusterer.hdbscan",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_hdbscan"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_hdbscan"],
    ),
    _s(
        "picasso_smlm_cluster",
        requires=["locs_undrifted"],
        provides=["clusters"],
        role="clusterer",
        relation=W,
        picasso_symbol="picasso.clusterer.cluster",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_smlm_cluster"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_smlm_cluster"],
    ),
    _s(
        "picasso_nneighbor",
        requires=["clusters"],
        provides=["nn_distances"],
        relation=W,
        picasso_symbol="picasso.__main__._nneighbor",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_nneighbor"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_nneighbor"],
    ),
    _s(
        "picasso_clusterfilter",
        requires=["locs", "clusters"],
        provides=["locs", "saved_dataset"],
        relation=W,
        picasso_symbol="picasso.__main__._clusterfilter",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_clusterfilter"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_clusterfilter"],
    ),
    _s(
        "picasso_cluster_combine",
        requires=["clusters"],
        provides=["locs"],
        relation=W,
        picasso_symbol="picasso.postprocess.cluster_combine",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_cluster_combine"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_cluster_combine"],
    ),
    _s(
        "picasso_cluster_combine_dist",
        requires=["clusters"],
        provides=["locs", "nn_distances"],
        relation=W,
        picasso_symbol="picasso.postprocess.cluster_combine_dist",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_cluster_combine_dist"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_cluster_combine_dist"],
    ),
    _s(
        "picasso_g5m",
        requires=["clusters"],
        provides=["locs", "saved_dataset"],
        relation=W,
        picasso_symbol="picasso.g5m.g5m",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_g5m"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_g5m"],
    ),
    _s(
        "picasso_render",
        requires=["locs_undrifted"],
        provides=["render_image"],
        relation=W,
        picasso_symbol="picasso.render.render",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_render"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_render"],
    ),
    _s(
        "picasso_align",
        provides=["saved_dataset"],
        relation=W,
        picasso_symbol="picasso.postprocess.align",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_align"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_align"],
    ),
    _s(
        "picasso_join",
        # like load_dataset_localizations: joined saved files are assumed
        # finished (drift-corrected) products
        provides=["locs", "locs_undrifted"],
        relation=W,
        picasso_symbol="picasso.lib.merge_locs",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_join"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_join"],
    ),
    # --- picasso-set tier 2: pick-based postprocessing ----------------------
    # Picks enter as a picks_file parameter (external file or a prior
    # module's result), so the `picks` capability is optional, not required.
    _s(
        "picasso_picked_locs",
        requires=["locs"],
        optional=["picks"],
        provides=["locs"],
        relation=W,
        picasso_symbol="picasso.postprocess.picked_locs",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_picked_locs"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_picked_locs"],
    ),
    _s(
        "picasso_pick_similar",
        requires=["locs"],
        optional=["picks"],
        provides=["picks"],
        relation=W,
        picasso_symbol="picasso.postprocess.pick_similar",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_pick_similar"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_pick_similar"],
    ),
    _s(
        "picasso_remove_locs_in_picks",
        requires=["locs"],
        optional=["picks"],
        provides=["locs"],
        relation=W,
        picasso_symbol="picasso.postprocess.remove_locs_in_picks",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_remove_locs_in_picks"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_remove_locs_in_picks"],
    ),
    _s(
        "picasso_pick_properties",
        requires=["locs"],
        optional=["picks"],
        provides=["binding_kinetics", "saved_dataset"],
        relation=W,
        picasso_symbol="picasso.postprocess.pick_properties",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_pick_properties"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_pick_properties"],
    ),
    _s(
        "picasso_pick_kinetics",
        requires=["locs"],
        optional=["picks"],
        provides=["locs", "binding_kinetics"],
        relation=W,
        picasso_symbol="picasso.postprocess.pick_kinetics",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_pick_kinetics"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_pick_kinetics"],
    ),
    _s(
        "picasso_fret",
        provides=["fret_results", "saved_dataset"],
        relation=W,
        picasso_symbol="picasso.postprocess.calculate_fret",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_fret"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_fret"],
    ),
    _s(
        "picasso_mask_locs",
        requires=["locs_undrifted"],
        provides=["locs", "mask"],
        relation=W,
        picasso_symbol="picasso.masking.mask_locs",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_mask_locs"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_mask_locs"],
    ),
    _s(
        "picasso_nena",
        requires=["locs"],
        provides=["resolution"],
        relation=W,
        picasso_symbol="picasso.postprocess.nena",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_nena"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_nena"],
    ),
    _s(
        "picasso_frc",
        requires=["locs_undrifted"],
        provides=["resolution"],
        relation=W,
        picasso_symbol="picasso.postprocess.frc",
        scopes=_SINGLE,
        summary=PICASSO_SET_SUMMARIES["picasso_frc"],
        module_set="picasso",
        params=PICASSO_SET_PARAMS["picasso_frc"],
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


def _step_params(step):
    """Return the parameters dict from a workflow step (``{}`` if none)."""
    if isinstance(step, dict):
        return step.get("parameters") or step.get("params") or {}
    if (
        isinstance(step, (tuple, list))
        and len(step) > 1
        and isinstance(step[1], dict)
    ):
        return step[1]
    return {}


def _initial_available(scope):
    """Capabilities available before any step runs, per scope.

    Both scopes start empty: a ``loader`` module must provide the base
    capabilities (``raw_movie``/``locs`` for single, ``dataset_collection``
    for aggregation). Kept as a hook for future per-scope seeding.
    """
    return set()


def validate_workflow(steps, scope, registry=None, available=None):
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

    if available is None:
        available = _initial_available(scope)
    else:
        available = set(available)
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
        if name == "branch":
            errors.extend(
                _validate_branch_step(
                    i, _step_params(step), scope, registry, available
                )
            )
        available |= spec.provides
        prior_names.append(name)
    return errors


def _validate_branch_step(i, params, scope, registry, available):
    """Validate a ``branch`` step's sub-workflows.

    ``branch_modules`` are validated starting from the trunk's current
    capabilities (so they can see the shared prefix); their ``provides`` are
    kept branch-local and do not leak back to the trunk. ``join_modules`` are
    then validated with ``branches`` added. A runtime branch requires a
    ``branch_over`` value/command (its resolved length sets the branch count).
    """
    errors = []

    branch_type = params.get("branch_type")
    if branch_type not in ("explicit", "runtime"):
        # Missing or unknown (e.g. the removed "screen"): AutoPicasso.branch
        # would KeyError / ValueError at run time after the prefix ran, so
        # reject it here instead.
        errors.append(
            f"[{i}.branch] branch_type must be 'explicit' or 'runtime' "
            f"(got {branch_type!r})"
        )
    elif branch_type == "runtime" and params.get("branch_over") is None:
        errors.append(
            f"[{i}.branch] runtime branch requires 'branch_over' (a value or "
            "command that resolves to a list; its length sets the branch count)"
        )
    elif branch_type == "explicit" and (
        params.get("n_branches") is None
        and params.get("branch_labels") is None
    ):
        errors.append(
            f"[{i}.branch] explicit branch requires 'n_branches' or "
            "'branch_labels'"
        )

    # branch_modules see the shared-prefix capabilities (available); their
    # provides stay branch-local and do not leak back to the trunk.
    branch_modules = params.get("branch_modules") or []
    sub_errors = validate_workflow(
        branch_modules, scope, registry, available=available
    )
    errors.extend(e.replace("[", f"[{i}.branch.", 1) for e in sub_errors)

    # join_modules run after the branches on the restored prefix state and may
    # pool per-branch results (the "branches" token).
    join_modules = params.get("join_modules") or []
    join_errors = validate_workflow(
        join_modules, scope, registry, available=available | {"branches"}
    )
    errors.extend(e.replace("[", f"[{i}.join.", 1) for e in join_errors)

    return errors


def _branch_trunk_requires(params, registry) -> frozenset[str]:
    """Capabilities a ``branch`` step's sub-workflows need from the trunk.

    A nested module's requirement counts only if no earlier module of the
    same sub-workflow provides it (mirroring how the branch executes its
    sub-modules in order on the trunk state).

    Parameters
    ----------
    params : dict
        The branch step's parameters (``branch_modules``/``join_modules``).
    registry : dict[str, ModuleSpec]
        Registry to look nested modules up in.

    Returns
    -------
    frozenset[str]
        The capabilities required from outside the branch.
    """
    needed: set[str] = set()
    for key in ("branch_modules", "join_modules"):
        provided: set[str] = set()
        for step in params.get(key) or []:
            spec = registry.get(_step_name(step))
            if spec is None:
                continue
            needed |= spec.requires - provided
            provided |= spec.provides
    return frozenset(needed)


def restart_conflicts(
    steps, restart_index: int, lost_capabilities=None, registry=None
) -> tuple[list[str], list[str]]:
    """Check whether restarting a workflow mid-way runs against lost state.

    Used by the checkpoint-aware resume: when a workflow is restarted at
    ``restart_index``, every module from there to the end executes in the
    resumed run. Any of them requiring a capability from
    ``lost_capabilities`` that was produced *before* the restart point (and
    is not re-produced within the re-run range) is guaranteed to run
    against missing state.

    Parameters
    ----------
    steps : iterable
        Ordered workflow steps in any format accepted by
        :func:`validate_workflow`. ``branch`` steps are inspected
        recursively: their sub-workflows' unmet requirements count as
        requirements of the branch step itself.
    restart_index : int
        Index of the first module that will be executed; everything before
        it is skipped.
    lost_capabilities : frozenset[str], optional
        The capabilities considered lost at the restart point. Defaults to
        ``MEMORY_ONLY_CAPABILITIES | LOCS_STATE_CAPABILITIES`` (nothing
        restored); a caller restoring a checkpoint passes the memory-only
        set plus whatever locs state the checkpoint does not cover.
    registry : dict[str, ModuleSpec], optional
        Registry to check against. Defaults to :data:`MODULE_REGISTRY`.

    Returns
    -------
    hard : list[str]
        Required-capability conflicts -- the restart point is not viable.
    soft : list[str]
        Advisory messages: lost ``optional`` inputs and modules unknown to
        the registry (best-effort specs).
    """
    if registry is None:
        registry = MODULE_REGISTRY
    if lost_capabilities is None:
        lost_capabilities = MEMORY_ONLY_CAPABILITIES | LOCS_STATE_CAPABILITIES
    producers: dict[str, int] = {}
    hard: list[str] = []
    soft: list[str] = []
    for i, step in enumerate(steps):
        name = _step_name(step)
        spec = registry.get(name)
        if i >= restart_index:
            if spec is None:
                soft.append(
                    f"[{i}] unknown module '{name}': cannot verify "
                    "in-memory requirements"
                )
            else:
                requires = spec.requires
                if name == "branch":
                    requires = requires | _branch_trunk_requires(
                        _step_params(step), registry
                    )
                for cap in sorted(requires & lost_capabilities):
                    p = producers.get(cap)
                    if p is not None and p < restart_index:
                        hard.append(
                            f"[{i}] {name} requires in-memory '{cap}' "
                            f"produced at [{p}], before the restart point "
                            f"[{restart_index}]"
                        )
                for cap in sorted(spec.optional & lost_capabilities):
                    p = producers.get(cap)
                    if p is not None and p < restart_index:
                        soft.append(
                            f"[{i}] {name} optionally uses in-memory "
                            f"'{cap}' produced at [{p}], before the restart "
                            f"point [{restart_index}]"
                        )
        if spec is not None:
            for cap in spec.provides:
                producers[cap] = i
    return hard, soft
