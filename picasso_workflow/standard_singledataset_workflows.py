#!/usr/bin/env python
"""Predefined standard workflows for analyzing single datasets.

Module Name: standard_singledataset_workflows.py
Author: Heinrich Grabmayr
Initial Date: March 20, 2024
"""

from __future__ import annotations

from picasso_workflow.live_locs import reuse_live_locs
from picasso_workflow.modulespec import Scope, validate_workflow


def preprocess(filepath, box_size=7, live_locs=None, movie=None):
    """Provide the modules for the authoritative "preprocess" reduction.

    The authoritative localization pass run when a FOV/round completes. It
    produces authoritative localizations and a render, saves the
    single-dataset result, and -- crucially -- does **no drift correction**:
    undrift (and aggregation) run on the SLURM cluster, and only the
    localizations cross to it (recommendation §"hand-off"; C33/C15). So this
    recipe is deliberately missing an ``undrift`` module.

    Two paths, decided by :func:`picasso_workflow.live_locs.reuse_live_locs`:

    - **Re-localize (default):** ``load_dataset_movie → identify → localize →
      render → save_single_dataset``.
    - **Reuse the lossless live locs:** when the live pass was lossless and
      still valid for this movie (frame-range checksum matches), the live
      localizations *are* authoritative, so identify+localize are replaced by
      a single ``load_dataset_localizations`` of the live locs -- no
      re-localization at all.

    Parameters
    ----------
    filepath : str
        Path to the movie to analyse.
    box_size : int
        The (odd) analysis box size.
    live_locs : LiveLocsDescriptor or dict, optional
        Descriptor of localizations produced by the live pass. When present,
        valid and lossless (checked against ``movie``), they are reused and
        the recipe skips re-localization. Absent/invalid → re-localize.
    movie : sequence of ndarray-like, optional
        The already-loaded movie, used only to validate ``live_locs`` (the
        frame-range checksum). When omitted, ``live_locs`` cannot be validated
        and the recipe re-localizes (the safe default).

    Returns
    -------
    list of tuple
        The ordered ``(module, parameters)`` workflow.
    """
    reusable = None
    if live_locs is not None and movie is not None:
        reusable = reuse_live_locs(movie, live_locs)

    if reusable is not None:
        # The lossless live locs are authoritative; load them instead of
        # re-identifying/re-localizing. A loaded locs file is treated as a
        # finished product (see load_dataset_localizations' ModuleSpec), so
        # the downstream render/save validate without a local undrift.
        load_step = (
            "load_dataset_localizations",
            {"filename": reusable.locs_path},
        )
        workflow_modules = [load_step]
    else:
        workflow_modules = [
            (
                "load_dataset_movie",
                {
                    "filename": filepath,
                    "sample_movie": {
                        "filename": "selected_frames.mp4",
                        "n_sample": 40,
                        "max_quantile": 0.9998,
                        "fps": 2,
                    },
                },
            ),
            (
                "identify",
                {
                    "auto_netgrad": {
                        "filename": "ng_histogram.png",
                        "frame_numbers": (
                            "$get_previous_module_result",
                            "sample_movie, sample_frame_idx",
                        ),
                        "box_size": box_size,
                        "start_ng": -3000,
                        "zscore": 5,
                    },
                    "ids_vs_frame": {"filename": "ids_vs_frame.png"},
                    "box_size": box_size,
                },
            ),
            (
                "localize",
                {
                    "fit_method": "lsq",
                    "box_size": box_size,
                    "fit_parallel": True,
                },
            ),
        ]

    # NO undrift here: the cluster does drift correction. Render + save then
    # operate on the (authoritative but not drift-corrected) localizations.
    workflow_modules += [
        ("render", {}),
        (
            "save_single_dataset",
            {
                "filename": ("$$map", "#tags", None),
            },
        ),
    ]
    return workflow_modules


# Capability the preprocess recipe treats as satisfied without a local
# undrift module: the cluster drift-corrects downstream, and only
# localizations cross to it. Render/save declare a ``locs_undrifted``
# requirement (they predate the acquire→cluster split), so validating the
# preprocess recipe seeds it as available -- it is not a missing dependency
# here but a deliberately-deferred step. See ``validate_preprocess``.
_PREPROCESS_CLUSTER_DEFERRED = {"locs_undrifted"}


def validate_preprocess(workflow_modules):
    """Validate a ``preprocess`` workflow, accounting for cluster-side undrift.

    Thin wrapper over
    :func:`picasso_workflow.modulespec.validate_workflow` for the
    single-dataset scope that seeds the cluster-deferred ``locs_undrifted``
    capability (see ``_PREPROCESS_CLUSTER_DEFERRED``) into the initial
    available set. Without that seed the generic validator would -- correctly
    for the classic local pipeline, wrongly for this split -- flag
    ``render`` / ``save_single_dataset`` as missing ``locs_undrifted``, since
    the preprocess recipe omits undrift on purpose.

    Parameters
    ----------
    workflow_modules : list of tuple
        The preprocess workflow (as returned by :func:`preprocess`).

    Returns
    -------
    list of str
        Validation error messages (empty when the recipe is valid).
    """
    return validate_workflow(
        workflow_modules,
        Scope.SINGLE,
        available=_PREPROCESS_CLUSTER_DEFERRED,
    )


def minimal(filepath, box_size=7):
    """Provide the modules for a minimal single-dataset workflow.

    The workflow consists of ``load_dataset``, ``identify``, ``localize`` and
    ``undrift_rcc``.

    Parameters
    ----------
    filepath : str
        The name of the file to analyze.
    box_size : int
        The (odd) analysis box size.
    """
    workflow_modules = [
        (
            "load_dataset_movie",
            {
                "filename": filepath,
                # "load_camera_info": True,
                "sample_movie": {
                    "filename": "selected_frames.mp4",
                    "n_sample": 40,
                    "max_quantile": 0.9998,
                    "fps": 2,
                },
            },
        ),
        (
            "identify",
            {
                "auto_netgrad": {
                    "filename": "ng_histogram.png",
                    "frame_numbers": (
                        "$get_previous_module_result",  # from prior results
                        "sample_movie, sample_frame_idx",
                    ),
                    "box_size": box_size,
                    "start_ng": -3000,
                    "zscore": 5,
                },
                "ids_vs_frame": {"filename": "ids_vs_frame.png"},
                "box_size": box_size,
            },
        ),
        # ('identify', {
        #     'net_gradient': 5000,
        #     'ids_vs_frame': {
        #         'filename': 'ids_vs_frame.png'
        #     },
        #     'box_size': box_size,
        #     },
        # ),
        (
            "localize",
            {"fit_method": "lsq", "box_size": box_size, "fit_parallel": True},
        ),
        (
            "undrift_rcc",
            {
                "segmentation": 500,
                "max_iter_segmentations": 4,
                "filename": "drift.csv",
                "save_locs": {"filename": "locs_undrift.hdf5"},
            },
        ),
        (
            "save_single_dataset",
            {
                # Save as "<tag>.hdf5" per channel in an aggregation run;
                # falls back to "locs.hdf5" standalone (tag-less).
                "filename": ("$$map", "#tags", None),
            },
        ),
    ]
    return workflow_modules


def picasso_native(filepath, fit_method="mle", gradient=5000):
    """Provide the modules for a native-picasso single-dataset workflow.

    Uses the picasso-set modules, which mirror the picasso CLI with its
    exact parameter names and defaults: ``load_dataset_movie`` →
    ``picasso_localize`` → ``picasso_undrift_rcc`` → ``picasso_render`` →
    ``save_single_dataset``.

    Parameters
    ----------
    filepath : str
        The name of the file to analyze.
    fit_method : str
        The picasso CLI fit method (default "mle").
    gradient : float
        Minimum net gradient for spot detection (default 5000).
    """
    workflow_modules = [
        (
            "load_dataset_movie",
            {
                "filename": filepath,
            },
        ),
        (
            "picasso_localize",
            {
                "fit_method": fit_method,
                "gradient": gradient,
            },
        ),
        (
            "picasso_undrift_rcc",
            {},
        ),
        (
            "picasso_render",
            {},
        ),
        (
            "save_single_dataset",
            {
                # Save as "<tag>.hdf5" per channel in an aggregation run;
                # falls back to "locs.hdf5" standalone (tag-less).
                "filename": ("$$map", "#tags", None),
            },
        ),
    ]
    return workflow_modules
