#!/usr/bin/env python
"""Predefined standard workflows for analyzing single datasets.

Module Name: standard_singledataset_workflows.py
Author: Heinrich Grabmayr
Initial Date: March 20, 2024
"""

from __future__ import annotations


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
                "filename": "locs.hdf5",
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
                "filename": "locs.hdf5",
            },
        ),
    ]
    return workflow_modules
