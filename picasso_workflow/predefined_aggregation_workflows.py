#!/usr/bin/env python
"""
Module Name: predefined_aggregation_workflows.py
Author: Heinrich Grabmayr
Initial Date: March 20, 2024
Description: This module provides predefined standard workflows for
    analyzing multiple datasets
"""
import picasso_workflow.predefined_singledataset_workflows as psw


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
    sgl_dataset_workflow = psw.minimal(
        filepath=("$map", "filename"), box_size=box_size
    )
    workflow_modules_agg = [
        (
            "load_datasets_to_aggregate",
            {
                "tags": ("$map", "#tags"),
                "filenames": (
                    "$get_prior_results",
                    "all_results, $all, save_single_dataset, filepath",
                ),
            },
        ),
        (
            "align_channels",
            {
                "evaldirs": (
                    "$get_prior_results",
                    "all_results, $all, undrift_rcc, locs_undrift.hdf5",
                )
            },
        ),
    ]
    workflow_modules_multi = {
        "single_dataset_tileparameters": {
            "#tags": [f"channel {i}" for i in len(filepaths)],
            "filename": filepaths,
        },
        "single_dataset_modules": sgl_dataset_workflow,
        "aggregation_modules": workflow_modules_agg,
    }
    return workflow_modules_multi
