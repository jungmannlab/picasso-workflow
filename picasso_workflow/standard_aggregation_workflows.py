#!/usr/bin/env python
"""Predefined standard workflows for analyzing multiple datasets.

Module Name: standard_aggregation_workflows.py
Author: Heinrich Grabmayr
Initial Date: March 20, 2024
"""

from __future__ import annotations

import picasso_workflow.standard_singledataset_workflows as ssw


def cluster_aggregation(filepaths, undrift="aim", segmentation=500):
    """Undrift-first cluster aggregation recipe (WP-9 hand-off default).

    The authoritative localization pass already ran on the acquisition side and
    did **no** drift correction -- only localizations crossed to the cluster
    (recommendation §"hand-off"; C33/C15). So the cluster job STARTS WITH
    undrift: each channel's saved ``_locs.hdf5`` is loaded, drift-corrected
    (AIM by default, RCC optional), re-saved, then the channels are aggregated.

    This is the "recommended aggregation that STARTS WITH undrift" the WP-9
    cluster-submit worker submits by default. The per-dataset leg therefore
    begins at ``load_dataset_localizations`` (not a movie) and its first real
    processing module is the undrift, so a template/conformance diff against
    this recipe sees undrift-first explicitly.

    Parameters
    ----------
    filepaths : list of str
        Per-channel saved localization files (``_locs.hdf5``) to aggregate.
    undrift : str
        Undrift algorithm: ``"aim"`` (default) or ``"rcc"``.
    segmentation : int
        Frames per undrift segment.
    """
    if undrift not in ("aim", "rcc"):
        raise ValueError(f"unknown undrift algorithm: {undrift!r}")
    if undrift == "aim":
        undrift_step = (
            "undrift_aim",
            {
                "segmentation": segmentation,
                "intersect_d": 20.0,
                "roi_r": 60.0,
                "dimensions": ["x", "y"],
                "filename": "drift.csv",
                "save_locs": {"filename": "locs_undrift.hdf5"},
            },
        )
    else:
        undrift_step = (
            "undrift_rcc",
            {
                "segmentation": segmentation,
                "max_iter_segmentations": 4,
                "filename": "drift.csv",
                "save_locs": {"filename": "locs_undrift.hdf5"},
            },
        )
    sgl_dataset_workflow = [
        (
            "load_dataset_localizations",
            {"filename": ("$$map", "filepath")},
        ),
        undrift_step,
        (
            "save_single_dataset",
            {"filename": ("$$map", "#tags", None)},
        ),
    ]
    idx_last_sgl_module = len(sgl_dataset_workflow) - 1
    workflow_modules_agg = [
        (
            "load_datasets_to_aggregate",
            {
                "tags": ("$$map", "#tags"),
                "filepaths": (
                    "$$get_prior_result",
                    "all_results, single_dataset, $all, "
                    + f"{idx_last_sgl_module:02d}_save_single_dataset, "
                    + "filepath",
                ),
            },
        ),
        (
            "align_channels",
            {},
        ),
        (
            "save_datasets_aggregated",
            {},
        ),
    ]
    workflow_modules_multi = {
        "single_dataset_tileparameters": {
            "#tags": [f"channel {i}" for i in range(len(filepaths))],
            "filepath": filepaths,
        },
        "single_dataset_modules": sgl_dataset_workflow,
        "aggregation_modules": workflow_modules_agg,
    }
    return workflow_modules_multi


def minimal_channel_align(filepaths, box_size=7):
    """Provide the modules for a minimal multi-dataset align workflow.

    Each dataset is processed with ``load_dataset``, ``identify``,
    ``localize`` and ``undrift_rcc``, after which the channels are aligned.

    Parameters
    ----------
    filepaths : list of str
        The names of the files to analyze.
    box_size : int
        The (odd) analysis box size.
    """
    sgl_dataset_workflow = ssw.minimal(
        filepath=("$$map", "filepath"), box_size=box_size
    )
    idx_last_sgl_module = len(sgl_dataset_workflow) - 1
    workflow_modules_agg = [
        (
            "load_datasets_to_aggregate",
            {
                "tags": ("$$map", "#tags"),
                "filepaths": (
                    "$$get_prior_result",
                    "all_results, single_dataset, $all, "
                    + f"{idx_last_sgl_module:02d}_save_single_dataset, "
                    + "filepath",
                ),
            },
        ),
        (
            "align_channels",
            {},
        ),
        (
            "save_datasets_aggregated",
            {},
        ),
    ]
    workflow_modules_multi = {
        "single_dataset_tileparameters": {
            "#tags": [f"channel {i}" for i in range(len(filepaths))],
            "filepath": filepaths,
        },
        "single_dataset_modules": sgl_dataset_workflow,
        "aggregation_modules": workflow_modules_agg,
    }
    return workflow_modules_multi
