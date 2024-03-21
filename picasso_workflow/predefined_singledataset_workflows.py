#!/usr/bin/env python
"""
Module Name: predefined_singledataset_workflows.py
Author: Heinrich Grabmayr
Initial Date: March 20, 2024
Description: This module provides predefined standard workflows for
    analyzing single datasets
"""


def minimal(filepath, box_size=7):
    """Provides workflow modules for a minimal workflow, consisting of
    - load_dataset
    - identify
    - localize
    - undrift_rcc
    Args:
        filepath : str
            the name of the file to analyze
        box_size : uneven int
            the analysis box size
    """
    workflow_modules = [
        (
            "load_dataset",
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
                        "$get_prior_result",  # get from prior results
                        "results, 00_load_dataset, "
                        + "sample_movie, sample_frame_idx",
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
                "filename": "locs.hdf5",
            },
        ),
    ]
    return workflow_modules
