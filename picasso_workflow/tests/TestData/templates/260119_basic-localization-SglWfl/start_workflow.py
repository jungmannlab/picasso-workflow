#!/usr/bin/env python
"""
Script Name: start_workflow.py
Author: Heinrich Grabmayr
Initial Date: February 19, 2025
Description: This script defines a "single-target" workflow for analysing
    all datasets present in subfolders of the file location.
    The usecase is that datasets acquired on a day can be automatically
    analysed until e.g. molecule positions, regardless of the specific question
    and downstream analysis across exchange rounds.
    The analysis is compatible with using a SLURM manager on a computing
    cluster.
    )
"""

import os
from picasso_workflow.metaworkflow import (
    find_dnapaint_raw,
    SingleWorkflowCoordinator,
)

# # machine the analysis will run on; must have file drives defined in .env
# dest_machine = "hpcl8"

# Personalized confluence access information for documentation is stored as
# environment variables
# UNIX:
#   export CONFLUENCE_URL="https://mibwiki.biochem.mpg.de" etc
#     MacOS: in ~/.bash_profile
#     linux cluster (hpcl8001): same in ~/.bashrc,
# Windows: in PowerShell, enter:
#  [Environment]::SetEnvironmentVariable(
#      "CONFLUENCE_BASE_PAGE",
#      "test reporting", [EnvironmentVariableTarget]::User)
confluence_url = os.getenv("CONFLUENCE_URL")
confluence_token = os.getenv("CONFLUENCE_TOKEN")
confluence_space = os.getenv("CONFLUENCE_SPACE")
base_page = os.getenv("CONFLUENCE_BASE_PAGE")


workflow_modules_sgl = [
    (  # 00
        "analysis_documentation",
        {},
    ),
    (  # 01
        "load_dataset_movie",
        {
            "filename": ("$$map", "filepath"),
            # "load_camera_info": True,
            "sample_movie": {
                "filename": "selected_frames.mp4",
                "n_sample": 40,
                "max_quantile": 0.9998,
                "fps": 2,
            },
        },
    ),
    (  # 02
        "identify",
        {
            "auto_netgrad": {
                "filename": "ng_histogram.png",
                "frame_numbers": (
                    "$get_previous_module_result",  # get from prior results
                    "sample_movie, sample_frame_idx",
                ),
                "box_size": 7,
                "start_ng": -3000,
                "zscore": 10,
            },
            "ids_vs_frame": {"filename": "ids_vs_frame.png"},
            "box_size": 7,
        },
    ),
    (  # 03
        "localize",
        {"fit_method": "lsq", "box_size": 7, "fit_parallel": True},
    ),
    (  # 04
        "undrift_aim",
        {
            "segmentation": 50,
            "dimensions": ["x", "y"],
            "intersect_d": 20,
            "roi_r": 60,
        },
    ),
    (  # 05
        "filter_locs",
        {
            "field": ["sx", "sy"],
            "minval": [0.8, 0.8],
            "maxval": [1.15, 1.15],
        },
    ),
    (  # 06
        "filter_locs",
        {
            "field": "ellipticity",
            "minval": 0,
            "maxval": 0.1,
        },
    ),
    (  # 07
        "render",
        {
            "ctrmass_fov_nm": 1000,
            "ctrmass_pixelsize": 10,
        },
    ),
    (  # 08
        "find_structures",
        {
            "diameter": 1.5,
            "min_n_locs_per_frame": 0.01,
            "display_pixelsize": 1,
            "n_plot_structures": 8,
            "xi": 0.01,
        },
    ),
    (  # 09
        "summarize_dataset",
        {
            "methods": {"nena": {}},
        },
    ),
    (  # 10
        "dbscan",
        {
            "radius": ("$get_previous_module_result *1.5", "nena, nena-nm"),
            "min_samples": 3,
            "continue_with_centers": False,
        },
    ),
    (  # 11
        "gaussian_mixture_cluster",
        {
            "min_locs": 10,
            "callback_parent": "silent",
        },
    ),
    (  # 12
        "nneighbor",
        {
            "dims": ["x", "y"],
            "nth_NN": 4,
            "nth_rdf": 12,
            "subsample_1stNN": 20,
        },
    ),
    (  # 13
        "save_single_dataset",
        {
            "filename": "locs.hdf5",
        },
    ),
]


if __name__ == "__main__":
    # parse file location, and find base DNA-PAINT files to analyse
    working_folder = os.path.dirname(os.path.abspath(__file__))
    datasets, src_loc_file = find_dnapaint_raw(working_folder)

    print("datasets", datasets)
    print("src_loc_file", src_loc_file)
    analysis_name = os.path.split(working_folder)[-1]

    coordinator = SingleWorkflowCoordinator(
        src_loc_file,
        analysis_name,
        working_folder,
        confluence_url,
        confluence_space,
        confluence_token,
        base_page,
        always_save=True,
    )

    coordinator.run_analysis(workflow_modules_sgl)
