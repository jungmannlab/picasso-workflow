#!/usr/bin/env python
"""Predefined standard workflows for analyzing multiple datasets.

Module Name: standard_aggregation_workflows.py
Author: Heinrich Grabmayr
Initial Date: March 20, 2024
"""

from __future__ import annotations

import picasso_workflow.standard_singledataset_workflows as ssw


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
