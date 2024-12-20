#!/usr/bin/env python
"""
Module Name: stanard_aggregation_workflows.py
Author: Heinrich Grabmayr
Initial Date: March 20, 2024
Description: This module provides predefined standard workflows for
    analyzing multiple datasets
"""
import picasso_workflow.standard_singledataset_workflows as ssw


def minimal_channel_align(filepaths, box_size=7):
    """Provides workflow modules for a minimal workflow for multiple
    files which are eventually aligned.
    Each dataset is processed with the follwing:
    - load_dataset
    - identify
    - localize
    - undrift_rcc
    Args:
        filepaths : list of str
            the names of the files to analyze
        box_size : uneven int
            the analysis box size
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
