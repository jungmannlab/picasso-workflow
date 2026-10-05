#!/usr/bin/env python
"""GUI parameter/result schemas and docstrings for the picasso-set modules.

One entry per picasso-set module, keyed by module name:

* :data:`PICASSO_SET_PARAMS` -- ``(parameters_spec, results_spec)`` tuples in
  the exact dict shape the GUI's ``ModuleDescriptor`` methods return (see
  ``gui.py``), consumed via ``ModuleSpec.params``.
* :data:`PICASSO_SET_SUMMARIES` -- one-line summaries, reused by the
  ``modulespec`` registry entries so there is a single source of truth.
* :func:`docstring` -- renders the GUI hover/help docstring from the above.

Parameter names and defaults replicate the picasso CLI
(``picasso/__main__.py``) / library signatures 1:1; any deliberate deviation
is called out in the parameter description.

This module is dependency-free on purpose (plain dicts only): it is imported
by :mod:`picasso_workflow.modulespec`, which must stay importable without the
analysis or GUI stacks.

Author: Heinrich Grabmayr
Initial date: October 5, 2026
"""

from __future__ import annotations

# Result keys every module gets from the module decorator.
_RESULTS_COMMON: dict = {
    "folder": {
        "type": "str",
        "description": "Output folder for module results",
    },
    "start time": {
        "type": "str",
        "description": "Module execution start timestamp",
    },
    "end time": {
        "type": "str",
        "description": "Module execution end timestamp",
    },
    "duration": {
        "type": "float",
        "description": "Module execution duration in seconds",
        "min": 0.0,
    },
}

_NLOCS_RESULT: dict = {
    "type": "int",
    "description": "Number of localizations after the module",
    "min": 0,
}


def _results(**extra) -> dict:
    """Return a results_spec: the common decorator keys plus ``extra``."""
    spec = dict(_RESULTS_COMMON)
    spec.update(extra)
    return spec


def _fp(description) -> dict:
    """A file-path result leaf."""
    return {"type": "str", "description": description}


# The CLI's fit-method choices (picasso/__main__.py localize parser).
_FIT_METHOD_CHOICES = [
    "mle",
    "mle-gpu",
    "mle-spherical",
    "mle-spherical-gpu",
    "mle-rotated",
    "mle-rotated-gpu",
    "mle-3d",
    "lq",
    "lq-spherical",
    "lq-spherical-gpu",
    "lq-rotated",
    "lq-rotated-gpu",
    "lq-3d",
    "lq-gpu",
    "lq-gpu-3d",
    "spline",
    "spline-mle",
    "spline-gpu",
    "spline-mle-gpu",
    "avg",
]


PICASSO_SET_SUMMARIES: dict[str, str] = {
    "picasso_localize": (
        "Identify and fit single-molecule spots "
        "(native picasso CLI: localize)."
    ),
    "picasso_undrift_rcc": (
        "Correct localization coordinates for drift using RCC "
        "(native picasso CLI: undrift)."
    ),
    "picasso_undrift_aim": (
        "Correct localization coordinates for drift with AIM "
        "(native picasso CLI: aim)."
    ),
    "picasso_undrift_fiducials": (
        "Correct localization coordinates for drift with fiducials "
        "(native picasso CLI: undrift_fiducials)."
    ),
    "picasso_link": (
        "Link localizations in consecutive frames "
        "(native picasso CLI: link)."
    ),
    "picasso_dark": (
        "Compute the dark time for grouped localizations "
        "(native picasso CLI: dark)."
    ),
    "picasso_groupprops": (
        "Calculate kinetics and properties of localization groups "
        "(native picasso CLI: groupprops)."
    ),
    "picasso_density": (
        "Compute the local localization density "
        "(native picasso CLI: density)."
    ),
    "picasso_pair_correlation": (
        "Calculate the pair-correlation of localizations "
        "(native picasso CLI: pc)."
    ),
    "picasso_dbscan": (
        "Cluster localizations with the DBSCAN algorithm "
        "(native picasso CLI: dbscan)."
    ),
    "picasso_hdbscan": (
        "Cluster localizations with the HDBSCAN algorithm "
        "(native picasso CLI: hdbscan)."
    ),
    "picasso_smlm_cluster": (
        "Cluster localizations with the custom SMLM clustering algorithm "
        "(native picasso CLI: smlm_cluster)."
    ),
    "picasso_nneighbor": (
        "Calculate nearest neighbor distances of a clustered dataset "
        "(native picasso CLI: nneighbor)."
    ),
    "picasso_clusterfilter": (
        "Filter localizations by properties of their clusters "
        "(native picasso CLI: clusterfilter)."
    ),
    "picasso_cluster_combine": (
        "Combine localizations in each cluster "
        "(native picasso CLI: cluster_combine)."
    ),
    "picasso_cluster_combine_dist": (
        "Calculate the nearest-neighbor distance for combined clusters "
        "(native picasso CLI: cluster_combine_dist)."
    ),
    "picasso_g5m": (
        "Gaussian Mixture Modeling for Molecular Mapping of clustered "
        "localizations (native picasso CLI: g5m)."
    ),
    "picasso_render": (
        "Render a localization-based image " "(native picasso CLI: render)."
    ),
    "picasso_align": (
        "Align one localization file to another "
        "(native picasso CLI: align)."
    ),
    "picasso_join": (
        "Join hdf5 localization lists with frame reindexing "
        "(native picasso CLI: join)."
    ),
    "picasso_picked_locs": (
        "Keep only the localizations inside the given picks "
        "(native picasso Render: picked locs)."
    ),
    "picasso_pick_similar": (
        "Find regions similar to the given picks "
        "(native picasso Render: pick similar)."
    ),
    "picasso_remove_locs_in_picks": (
        "Remove the localizations inside the given picks "
        "(native picasso Render: remove picked locs)."
    ),
    "picasso_pick_properties": (
        "Calculate statistical properties per pick "
        "(native picasso Render: save pick properties)."
    ),
    "picasso_pick_kinetics": (
        "Estimate binding kinetics per pick "
        "(native picasso Render: pick kinetics)."
    ),
    "picasso_fret": (
        "Calculate FRET efficiencies from a donor and an acceptor "
        "dataset (native picasso Render: calculate FRET)."
    ),
    "picasso_mask_locs": (
        "Split localizations by a density mask "
        "(native picasso Render: mask image)."
    ),
    "picasso_nena": (
        "Estimate the localization precision via NeNA "
        "(native picasso Render: NeNA)."
    ),
    "picasso_frc": (
        "Estimate the image resolution via Fourier Ring Correlation "
        "(native picasso Render: FRC)."
    ),
}


PICASSO_SET_PARAMS: dict[str, tuple[dict, dict]] = {
    "picasso_localize": (
        {
            "fit_method": {
                "type": "str",
                "description": "Fit method (CLI: --fit-method)",
                "options": _FIT_METHOD_CHOICES,
                "default": "mle",
                "required": False,
            },
            "box_side_length": {
                "type": "int",
                "description": (
                    "Side length of the fit box " "(CLI: --box-side-length)"
                ),
                "default": 7,
                "required": False,
            },
            "gradient": {
                "type": "float",
                "description": (
                    "Minimum net gradient for spot detection "
                    "(CLI: --gradient)"
                ),
                "default": 5000,
                "required": False,
            },
            "roi": {
                "type": "list",
                "description": (
                    "Region of interest [y_min, x_min, y_max, x_max] "
                    "(CLI: --roi; one region)"
                ),
                "required": False,
            },
            "frame_bounds": {
                "type": "list",
                "description": (
                    "[start_frame, end_frame] segment(s), 0-indexed "
                    "inclusive (CLI: --frame-bounds)"
                ),
                "required": False,
            },
            "temporal_median": {
                "type": "int",
                "description": (
                    "Rolling temporal median background filter window "
                    "(CLI: --temporal-median; 0 = off)"
                ),
                "default": 0,
                "required": False,
            },
            "gaussian_filter": {
                "type": "float",
                "description": (
                    "Spatial Gaussian pre-filter sigma "
                    "(CLI: --gaussian-filter; 0 = off)"
                ),
                "default": 0.0,
                "required": False,
            },
            "convergence": {
                "type": "float",
                "description": (
                    "Fit tolerance; 0 = the fit method's own default "
                    "(CLI: --convergence)"
                ),
                "default": 0,
                "required": False,
            },
            "max_iterations": {
                "type": "int",
                "description": (
                    "Max iterations per spot; 0 = the method's own "
                    "default (CLI: --max-iterations)"
                ),
                "default": 0,
                "required": False,
            },
            "baseline": {
                "type": "float",
                "description": (
                    "Camera baseline (CLI: --baseline, literal 0; here "
                    "None = from loaded metadata/config)"
                ),
                "required": False,
            },
            "sensitivity": {
                "type": "float",
                "description": (
                    "Camera sensitivity (CLI: --sensitivity, literal 1; "
                    "here None = from loaded metadata/config)"
                ),
                "required": False,
            },
            "gain": {
                "type": "float",
                "description": (
                    "Camera gain (CLI: --gain, literal 1; here None = "
                    "from loaded metadata/config)"
                ),
                "required": False,
            },
            "qe": {
                "type": "float",
                "description": (
                    "Quantum efficiency (CLI: --qe, literal 1; here "
                    "None = from loaded metadata/config)"
                ),
                "required": False,
            },
            "pixelsize": {
                "type": "float",
                "description": (
                    "Camera pixel size in nm (CLI: --pixelsize, literal "
                    "130; here None = from loaded metadata/config)"
                ),
                "required": False,
            },
            "spline_calibration": {
                "type": "file",
                "description": (
                    "Spline PSF calibration .hdf5, required for spline "
                    "methods (CLI: --spline-calibration)"
                ),
                "required": False,
            },
            "affine_calibration": {
                "type": "list",
                "description": (
                    "Lateral-correction calibration file(s) "
                    "(CLI: --affine-calibration)"
                ),
                "required": False,
            },
            "camera_calibration": {
                "type": "file",
                "description": (
                    "sCMOS camera calibration .hdf5 "
                    "(CLI: --camera-calibration)"
                ),
                "required": False,
            },
            "zc": {
                "type": "file",
                "description": (
                    "3D z-calibration file, required for -3d methods "
                    "(CLI: --zc)"
                ),
                "required": False,
            },
            "mf": {
                "type": "float",
                "description": (
                    "Magnification factor, required for -3d methods "
                    "(CLI: --mf)"
                ),
                "required": False,
            },
            "fit_z_gpu": {
                "type": "bool",
                "description": (
                    "Fit z on the GPU; falls back to CPU if unavailable "
                    "(CLI: --fit-z-gpu)"
                ),
                "default": False,
                "required": False,
            },
            "drift": {
                "type": "int",
                "description": (
                    "RCC segmentation for post-fit undrift; 0 = off "
                    "(CLI: --drift, default 1000 - deviation: the "
                    "workflow has dedicated undrift modules)"
                ),
                "default": 0,
                "required": False,
            },
            "suffix": {
                "type": "str",
                "description": (
                    "Inserted into the output file name (CLI: --suffix)"
                ),
                "required": False,
            },
        },
        _results(
            nlocs=_NLOCS_RESULT,
            filepath_locs=_fp("Fitted localizations hdf5"),
        ),
    ),
    "picasso_undrift_rcc": (
        {
            "segmentation": {
                "type": "float",
                "description": (
                    "Number of frames combined for one temporal segment "
                    "(CLI: --segmentation)"
                ),
                "default": 1000,
                "required": False,
            },
            "fromfile": {
                "type": "file",
                "description": (
                    "Apply drift from this drift .txt file instead of "
                    "computing it (CLI: --fromfile)"
                ),
                "required": False,
            },
        },
        _results(
            nlocs=_NLOCS_RESULT,
            filepath_locs_undrift=_fp("Undrifted localizations hdf5"),
            filepath_driftfile=_fp("Drift table text file"),
            fp_fig_drift=_fp("Drift plot figure"),
        ),
    ),
    "picasso_undrift_aim": (
        {
            "segmentation": {
                "type": "float",
                "description": (
                    "Number of frames combined for one temporal segment "
                    "(CLI: --segmentation)"
                ),
                "default": 100,
                "required": False,
            },
            "intersectdist": {
                "type": "float",
                "description": (
                    "Max. distance (camera pixels) between localizations "
                    "in consecutive segments to be considered intersecting "
                    "(CLI: --intersectdist)"
                ),
                "default": 20 / 130,
                "required": False,
            },
            "roiradius": {
                "type": "float",
                "description": (
                    "Max. drift (camera pixels) between two consecutive "
                    "segments (CLI: --roiradius)"
                ),
                "default": 60 / 130,
                "required": False,
            },
        },
        _results(
            nlocs=_NLOCS_RESULT,
            filepath_locs_aim=_fp("Undrifted localizations hdf5"),
            filepath_driftfile=_fp("Drift table text file"),
            fp_fig_drift=_fp("Drift plot figure"),
        ),
    ),
    "picasso_undrift_fiducials": (
        {},
        _results(
            nlocs=_NLOCS_RESULT,
            filepath_locs_undrift_fiducials=_fp(
                "Undrifted localizations hdf5"
            ),
            filepath_driftfile=_fp("Drift table text file"),
            fp_fig_drift=_fp("Drift plot figure"),
        ),
    ),
    "picasso_link": (
        {
            "distance": {
                "type": "float",
                "description": (
                    "Maximum distance (camera pixels) between "
                    "localizations to consider them the same binding "
                    "event (CLI: --distance)"
                ),
                "default": 1.0,
                "required": False,
            },
            "tolerance": {
                "type": "int",
                "description": (
                    "Maximum dark time between localizations to still "
                    "consider them the same binding event "
                    "(CLI: --tolerance)"
                ),
                "default": 1,
                "required": False,
            },
        },
        _results(
            nlocs=_NLOCS_RESULT,
            filepath_locs_link=_fp("Linked localizations hdf5"),
        ),
    ),
    "picasso_dark": (
        {},
        _results(
            nlocs=_NLOCS_RESULT,
            filepath_locs_dark=_fp("Localizations with dark times hdf5"),
        ),
    ),
    "picasso_groupprops": (
        {},
        _results(
            n_groups={
                "type": "int",
                "description": "Number of localization groups",
                "min": 0,
            },
            filepath_locs_groupprops=_fp(
                "Locs + group properties hdf5 (io.save_datasets)"
            ),
        ),
    ),
    "picasso_density": (
        {
            "radius": {
                "type": "float",
                "description": (
                    "Maximum distance (camera pixels) for localizations "
                    "to count as local (CLI: radius)"
                ),
                "min": 0.0,
                "required": True,
            },
        },
        _results(
            nlocs=_NLOCS_RESULT,
            filepath_locs_density=_fp(
                "Saved density-annotated localizations "
                "(mirrors the CLI's _density.hdf5 output)"
            ),
        ),
    ),
    "picasso_pair_correlation": (
        {
            "binsize": {
                "type": "float",
                "description": "The bin size in camera pixels (CLI: -b)",
                "default": 0.1,
                "required": False,
            },
            "rmax": {
                "type": "float",
                "description": (
                    "The maximum distance for the pair-correlation "
                    "(CLI: -r)"
                ),
                "default": 10,
                "required": False,
            },
        },
        _results(
            fp_fig_pair_correlation=_fp("Pair-correlation curve figure"),
            filepath_pair_correlation=_fp("Pair-correlation curve data"),
        ),
    ),
    "picasso_dbscan": (
        {
            "radius": {
                "type": "float",
                "description": (
                    "Maximal distance (camera pixels) between two "
                    "localizations to be considered local (CLI: radius)"
                ),
                "required": True,
            },
            "density": {
                "type": "int",
                "description": (
                    "Minimum local density for localizations to be "
                    "assigned to a cluster (CLI: density)"
                ),
                "required": True,
            },
            "pixelsize": {
                "type": "int",
                "description": (
                    "Camera pixel size in nm (required for 3D "
                    "localizations only)"
                ),
                "required": False,
            },
            "radius_z": {
                "type": "float",
                "description": (
                    "DBSCAN epsilon in z (camera pixels); enables "
                    "anisotropic 3D clustering (CLI: --radius_z)"
                ),
                "required": False,
            },
        },
        _results(
            nlocs=_NLOCS_RESULT,
            nclusters={
                "type": "int",
                "description": "Number of clusters found",
                "min": 0,
            },
            filepath_locs_dbscan=_fp("Clustered localizations hdf5"),
            filepath_locs_dbclusters=_fp("Cluster centers hdf5"),
        ),
    ),
    "picasso_hdbscan": (
        {
            "min_cluster": {
                "type": "int",
                "description": (
                    "Smallest size grouping considered a cluster "
                    "(CLI: min_cluster)"
                ),
                "required": True,
            },
            "min_samples": {
                "type": "int",
                "description": (
                    "The higher, the more points are considered noise "
                    "(CLI: min_samples)"
                ),
                "required": True,
            },
            "pixelsize": {
                "type": "int",
                "description": (
                    "Camera pixel size in nm (required for 3D "
                    "localizations only)"
                ),
                "required": False,
            },
        },
        _results(
            nlocs=_NLOCS_RESULT,
            nclusters={
                "type": "int",
                "description": "Number of clusters found",
                "min": 0,
            },
            filepath_locs_hdbscan=_fp("Clustered localizations hdf5"),
            filepath_locs_hdbclusters=_fp("Cluster centers hdf5"),
        ),
    ),
    "picasso_smlm_cluster": (
        {
            "radius": {
                "type": "float",
                "description": "Clustering radius in camera pixels",
                "required": True,
            },
            "min_locs": {
                "type": "int",
                "description": (
                    "Minimum number of localizations in a cluster"
                ),
                "required": True,
            },
            "pixelsize": {
                "type": "int",
                "description": (
                    "Camera pixel size in nm (required for 3D "
                    "localizations only)"
                ),
                "required": False,
            },
            "basic_fa": {
                "type": "bool",
                "description": (
                    "Whether to perform basic frame analysis (sticking "
                    "event removal)"
                ),
                "default": False,
                "required": False,
            },
            "radius_z": {
                "type": "float",
                "description": (
                    "Clustering radius in axial direction (must be set "
                    "for 3D)"
                ),
                "required": False,
            },
        },
        _results(
            nlocs=_NLOCS_RESULT,
            nclusters={
                "type": "int",
                "description": "Number of clusters found",
                "min": 0,
            },
            filepath_locs_clusters=_fp("Clustered localizations hdf5"),
            filepath_locs_cluster_centers=_fp("Cluster centers hdf5"),
        ),
    ),
    "picasso_nneighbor": (
        {
            "files": {
                "type": "file",
                "description": (
                    "The hdf5 cluster file (e.g. the cluster-centers "
                    "output of a picasso-set clusterer, via "
                    "$get_prior_result)"
                ),
                "required": True,
            },
        },
        _results(
            filepath_minval=_fp(
                "Nearest-neighbor distances text file "
                "(the CLI's _minval.txt)"
            ),
            n_points={
                "type": "int",
                "description": "Number of cluster centers",
                "min": 0,
            },
            nn_mean={
                "type": "float",
                "description": "Mean nearest-neighbor distance",
            },
        ),
    ),
    "picasso_clusterfilter": (
        {
            "clusterfile": {
                "type": "file",
                "description": "A hdf5 clusterfile (CLI: --clusterfile)",
                "required": True,
            },
            "parameter": {
                "type": "str",
                "description": (
                    "Cluster parameter to be filtered (CLI: --parameter)"
                ),
                "required": True,
            },
            "minval": {
                "type": "float",
                "description": "Lower boundary (CLI: --minval)",
                "required": True,
            },
            "maxval": {
                "type": "float",
                "description": "Upper boundary (CLI: --maxval)",
                "required": True,
            },
        },
        _results(
            filepath_locs_filter_in=_fp("In-range localizations hdf5"),
            filepath_locs_filter_out=_fp("Out-of-range localizations hdf5"),
            nlocs_in={
                "type": "int",
                "description": "Number of in-range localizations",
                "min": 0,
            },
            nlocs_out={
                "type": "int",
                "description": "Number of out-of-range localizations",
                "min": 0,
            },
        ),
    ),
    "picasso_cluster_combine": (
        {},
        _results(
            nlocs=_NLOCS_RESULT,
            filepath_locs_comb=_fp("Combined localizations hdf5"),
        ),
    ),
    "picasso_cluster_combine_dist": (
        {},
        _results(
            nlocs=_NLOCS_RESULT,
            filepath_locs_cdist=_fp(
                "Combined localizations with distances hdf5"
            ),
        ),
    ),
    "picasso_g5m": (
        {
            "min_locs": {
                "type": "int",
                "description": (
                    "Min. number of locs per molecule (CLI: --min-locs)"
                ),
                "default": 10,
                "required": False,
            },
            "loc_prec_handle": {
                "type": "str",
                "description": (
                    "Localization precision handle " "(CLI: --loc-prec-handle)"
                ),
                "options": ["local", "abs"],
                "default": "local",
                "required": False,
            },
            "min_sigma": {
                "type": "float",
                "description": (
                    "Minimum sigma factor/value (CLI: --min-sigma)"
                ),
                "default": 0.8,
                "required": False,
            },
            "max_sigma": {
                "type": "float",
                "description": (
                    "Maximum sigma factor/value (CLI: --max-sigma)"
                ),
                "default": 1.5,
                "required": False,
            },
            "max_rounds": {
                "type": "int",
                "description": (
                    "Max. rounds without BIC improvement to terminate "
                    "(CLI: --max-rounds)"
                ),
                "default": 3,
                "required": False,
            },
            "bootstrap_sem": {
                "type": "bool",
                "description": (
                    "Bootstrap to estimate SEM of molecule positions "
                    "(CLI: --bootstrap-sem)"
                ),
                "default": False,
                "required": False,
            },
            "calibration": {
                "type": "file",
                "description": (
                    "Astigmatism calibration file; required only for "
                    "astigmatism 3D data (CLI: --calibration)"
                ),
                "required": False,
            },
            "mode": {
                "type": "str",
                "description": (
                    "3D fitting mode of the input localizations "
                    "(CLI: --mode)"
                ),
                "options": ["astigmatism", "spline"],
                "default": "astigmatism",
                "required": False,
            },
            "covariance_type": {
                "type": "str",
                "description": (
                    "Shape of the G5M components (CLI: --covariance-type)"
                ),
                "options": ["auto", "spherical", "diagonal", "rotated"],
                "default": "auto",
                "required": False,
            },
            "postprocess": {
                "type": "bool",
                "description": (
                    "Postprocess results to remove sticking events and "
                    "low-quality fits (CLI: -p disables)"
                ),
                "default": True,
                "required": False,
            },
            "max_locs": {
                "type": "int",
                "description": (
                    "Maximum number of localizations per cluster "
                    "(CLI: --max-locs)"
                ),
                "default": 100000,
                "required": False,
            },
            "asynch": {
                "type": "bool",
                "description": (
                    "Fit asynchronously via multiprocessing "
                    "(CLI: -a disables)"
                ),
                "default": True,
                "required": False,
            },
            "group_column": {
                "type": "str",
                "description": (
                    "Column used to group localizations into clusters "
                    "(CLI: --group-column)"
                ),
                "options": ["group", "group_input"],
                "default": "group",
                "required": False,
            },
        },
        _results(
            n_molecules={
                "type": "int",
                "description": "Number of molecules in the molecule map",
                "min": 0,
            },
            filepath_locs_molmap=_fp(
                "Molecule map hdf5 (the CLI's _molmap.hdf5)"
            ),
        ),
    ),
    "picasso_render": (
        {
            "disp_px_size": {
                "type": "float",
                "description": (
                    "The size of the rendered pixel in nm "
                    "(CLI: --disp-px-size)"
                ),
                "default": 10.0,
                "required": False,
            },
            "blur_method": {
                "type": "str",
                "description": "Blur method (CLI: --blur-method)",
                "options": ["none", "convolve", "gaussian"],
                "default": "convolve",
                "required": False,
            },
            "min_blur_width": {
                "type": "float",
                "description": (
                    "Minimum blur width if blur is applied "
                    "(CLI: --min-blur-width)"
                ),
                "default": 0.0,
                "required": False,
            },
            "vmin": {
                "type": "float",
                "description": (
                    "Minimum colormap level in range 0-100 or absolute "
                    "value (CLI: --vmin)"
                ),
                "default": 0.0,
                "required": False,
            },
            "vmax": {
                "type": "float",
                "description": (
                    "Maximum colormap level in range 0-100 or absolute "
                    "value (CLI: --vmax)"
                ),
                "default": 20.0,
                "required": False,
            },
            "scaling": {
                "type": "str",
                "description": (
                    "If 'yes', vmin/vmax are relative in the range 0-100 "
                    "(CLI: --scaling)"
                ),
                "options": ["yes", "no"],
                "default": "yes",
                "required": False,
            },
            "cmap": {
                "type": "str",
                "description": (
                    "The colormap to be applied (CLI: --cmap; the CLI "
                    "default comes from user settings, here: viridis)"
                ),
                "options": [
                    "viridis",
                    "inferno",
                    "plasma",
                    "magma",
                    "hot",
                    "gray",
                ],
                "default": "viridis",
                "required": False,
            },
        },
        _results(
            fp_fig_render=_fp("Rendered image PNG"),
            n_rendered={
                "type": "int",
                "description": "Number of localizations rendered",
                "min": 0,
            },
        ),
    ),
    "picasso_align": (
        {
            "files": {
                "type": "list",
                "description": (
                    "The hdf5 localization files to align (CLI: file)"
                ),
                "required": True,
            },
        },
        _results(
            filepaths_aligned={
                "type": "list",
                "description": "The aligned localization files",
            },
        ),
    ),
    "picasso_join": (
        {
            "files": {
                "type": "list",
                "description": (
                    "The hdf5 localization files to be joined (CLI: file)"
                ),
                "required": True,
            },
            "keepindex": {
                "type": "bool",
                "description": (
                    "Do not change frame numbers (CLI: --keepindex)"
                ),
                "default": False,
                "required": False,
            },
        },
        _results(
            nlocs=_NLOCS_RESULT,
            filepath_locs_join=_fp("Joined localizations hdf5"),
        ),
    ),
    "picasso_picked_locs": (
        {
            "picks_file": {
                "type": "file",
                "description": (
                    "Picasso pick-region .yaml file (e.g. from "
                    "picasso_pick_similar via $get_prior_result)"
                ),
                "required": True,
            },
            "add_group": {
                "type": "bool",
                "description": "Add a group column indexing the picks",
                "default": True,
                "required": False,
            },
        },
        _results(
            nlocs=_NLOCS_RESULT,
            n_picks={
                "type": "int",
                "description": "Number of picks",
                "min": 0,
            },
            filepath_locs_picked=_fp("Picked localizations hdf5"),
        ),
    ),
    "picasso_pick_similar": (
        {
            "picks_file": {
                "type": "file",
                "description": (
                    "Picasso pick-region .yaml file with the seed picks "
                    "(Circle, Rectangle, Square or Box)"
                ),
                "required": True,
            },
            "std_range": {
                "type": "float",
                "description": (
                    "Allowed deviation (in standard deviations) of locs "
                    "count and RMSD from the seed picks' mean"
                ),
                "default": 2.0,
                "required": False,
            },
        },
        _results(
            n_picks_input={
                "type": "int",
                "description": "Number of seed picks",
                "min": 0,
            },
            n_picks_similar={
                "type": "int",
                "description": "Number of similar picks found",
                "min": 0,
            },
            filepath_picks=_fp("Pick-region yaml with the found picks"),
        ),
    ),
    "picasso_remove_locs_in_picks": (
        {
            "picks_file": {
                "type": "file",
                "description": "Picasso pick-region .yaml file",
                "required": True,
            },
        },
        _results(
            nlocs=_NLOCS_RESULT,
            filepath_locs_picks_removed=_fp(
                "Localizations outside the picks hdf5"
            ),
        ),
    ),
    "picasso_pick_properties": (
        {
            "picks_file": {
                "type": "file",
                "description": "Picasso pick-region .yaml file",
                "required": True,
            },
            "max_dark_time": {
                "type": "int",
                "description": (
                    "Maximum dark time for linking binding events"
                ),
                "default": 3,
                "required": False,
            },
            "influx_rate": {
                "type": "float",
                "description": "Influx rate for qPAINT unit calibration",
                "default": 0.03,
                "required": False,
            },
        },
        _results(
            n_picks={
                "type": "int",
                "description": "Number of picks",
                "min": 0,
            },
            filepath_pick_properties=_fp(
                "Per-pick property table hdf5 (io.save_datasets)"
            ),
        ),
    ),
    "picasso_pick_kinetics": (
        {
            "picks_file": {
                "type": "file",
                "description": "Picasso pick-region .yaml file",
                "required": True,
            },
            "max_dark_time": {
                "type": "int",
                "description": (
                    "Maximum dark time for linking binding events"
                ),
                "default": 3,
                "required": False,
            },
        },
        _results(
            nlocs=_NLOCS_RESULT,
            n_picks={
                "type": "int",
                "description": "Number of picks",
                "min": 0,
            },
            n_picks_kept={
                "type": "int",
                "description": "Picks with estimable kinetics",
                "min": 0,
            },
            mean_length_frames={
                "type": "float",
                "description": "Mean binding-event length (frames)",
            },
            mean_dark_frames={
                "type": "float",
                "description": "Mean dark time (frames)",
            },
            filepath_locs_pick_kinetics=_fp(
                "Picked localizations with kinetics columns hdf5"
            ),
        ),
    ),
    "picasso_fret": (
        {
            "acc_locs_file": {
                "type": "file",
                "description": "The acceptor localizations hdf5 file",
                "required": True,
            },
            "don_locs_file": {
                "type": "file",
                "description": "The donor localizations hdf5 file",
                "required": True,
            },
        },
        _results(
            n_fret_events={
                "type": "int",
                "description": "Number of FRET events",
                "min": 0,
            },
            mean_fret={
                "type": "float",
                "description": "Mean FRET efficiency",
            },
            filepath_locs_fret=_fp("FRET localizations hdf5"),
            filepath_fret_events=_fp(
                "FRET events text file (frame, efficiency)"
            ),
        ),
    ),
    "picasso_mask_locs": (
        {
            "disp_px_size": {
                "type": "float",
                "description": ("Size of the rendered mask pixel in nm"),
                "required": True,
            },
            "blur": {
                "type": "float",
                "description": (
                    "Gaussian blur sigma applied to the rendered image "
                    "(display pixels)"
                ),
                "required": True,
            },
            "method": {
                "type": "str",
                "description": (
                    "Thresholding method or an explicit threshold in " "(0, 1)"
                ),
                "options": [
                    "isodata",
                    "li",
                    "mean",
                    "minimum",
                    "otsu",
                    "triangle",
                    "yen",
                    "local_gaussian",
                    "local_mean",
                    "local_median",
                ],
                "default": "otsu",
                "required": False,
            },
        },
        _results(
            nlocs_in={
                "type": "int",
                "description": "Localizations inside the mask",
                "min": 0,
            },
            nlocs_out={
                "type": "int",
                "description": "Localizations outside the mask",
                "min": 0,
            },
            threshold={
                "type": "float",
                "description": "Applied threshold (scalar methods only)",
                "required": False,
            },
            filepath_mask=_fp("Binary mask .npy"),
            fp_fig_mask=_fp("Mask image PNG"),
            filepath_locs_mask_in=_fp("In-mask localizations hdf5"),
            filepath_locs_mask_out=_fp("Out-of-mask localizations hdf5"),
        ),
    ),
    "picasso_nena": (
        {},
        _results(
            nena_px={
                "type": "float",
                "description": (
                    "Estimated localization precision (camera pixels)"
                ),
            },
            nena_nm={
                "type": "float",
                "description": (
                    "Estimated localization precision (nm, if the pixel "
                    "size is known)"
                ),
                "required": False,
            },
        ),
    ),
    "picasso_frc": (
        {
            "random_seed": {
                "type": "int",
                "description": (
                    "Seed for the random split of the localizations"
                ),
                "default": 42,
                "required": False,
            },
        },
        _results(
            resolution_nm={
                "type": "float",
                "description": "Estimated FRC resolution (nm)",
            },
            fp_fig_frc=_fp("FRC curve figure"),
            filepath_frc=_fp("FRC curve data"),
        ),
    ),
}


def docstring(name: str) -> str:
    """Render the GUI docstring for a picasso-set module.

    Parameters
    ----------
    name : str
        The picasso-set module name (e.g. ``"picasso_density"``).

    Returns
    -------
    str
        A NumPy-style docstring built from the summary and parameter spec.
    """
    parameters_spec, _ = PICASSO_SET_PARAMS[name]
    lines = [
        PICASSO_SET_SUMMARIES[name],
        "",
        "Parameters",
        "----------",
    ]
    if not parameters_spec:
        lines.append("(none)")
    for pname, pspec in parameters_spec.items():
        required = "required" if pspec.get("required") else "optional"
        default = pspec.get("default")
        default_txt = "" if default is None else f", default: {default!r}"
        lines.append(f"{pname} : {pspec.get('type', 'any')}")
        lines.append(f"    {pspec.get('description', '')}")
        lines.append(f"    ({required}{default_txt})")
    return "\n".join(lines)
