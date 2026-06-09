#!/usr/bin/env python
"""
Script Name: start_workflow.py
Author: Heinrich Grabmayr
Initial Date: January 30, 2025
Description: This script defines parameters and runs a multi-level analysis
    of an investigation. An investigation consists of the following levels:
    Multiple targets (receptors are imaged per cell/FOV); multiple cells/FOVs
    are imaged per condition/sample type.
    The analysis is compatible with using a SLURM manager on a computing
    cluster, and intended to be run from a bash script twice - first for the
    analysis workflow of the single cells, and in a second step for the
    aggregation of the single cells to the sample/condition level. Each time, a
    command line argument specifies the stage
    (e.g.
        srun python3 start_workflow.py cell
        or
        srun python3 start_workflow.py condition
    )
"""

import os
import shutil

# DNA-PAINT targets imaged
receptors = 2

analysis_name = "LE-investigation-gold-SMLM"  # -2"

investigation_description = f"""
<b>{analysis_name}</b>

This is an investigation of multiple labeling efficiency measurements,
of different targets.
"""
# In this test 2, we are trying channel alignment with too few/no gold beads.
# """

# yaml file with a list (len 1) of dicts with keys describing the dataset
# with A_B_C_D, A_B describing the sample/condition, C the cell/FOV ID,
# and D the target, and values are filepaths as valid on one of the machines
# defined in .env
src_loc = "raw_locs_list.yaml"


# machine the analysis will run on; must have file drives defined in .env
dest_machine = "hpcl8"
# dest_machine = "pcju55.local"

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
confluence_token = os.getenv("CONFLUENCE_BEARER")
confluence_space = os.getenv("CONFLUENCE_SPACE")
base_page = os.getenv("CONFLUENCE_BASE_PAGE")


def get_workflow(datasets):

    reference_name = datasets["#tags"][0]
    target_name = datasets["#tags"][1]

    workflow_modules_sgl = [
        (  # 00
            "analysis_documentation",
            {},
        ),
        (  # 01
            "load_dataset_localizations",
            {
                "filename": ("$$map", "filepath"),
            },
        ),
        # (  # 02
        #    "undrift_aim",
        #    {
        #        "segmentation": 10,
        #        "intersect_d": 20,
        #        "roi_r": 60,
        #        "dimensions": ["x", "y"],
        #    }
        # ),
        (  # 02
            "undrift_rcc",
            {
                "segmentation": 1000,
                "dimensions": ["x", "y"],
                "filename": "drift_rcc.txt",
            },
        ),
        (  # 03
            "find_gold",
            {
                "diameter": 2,
                "std_range": 1.4,
                "mean_rmsd": 0.4,
                "remove_gold": False,
            },
        ),
        (  # 04
            "undrift_from_picked",
            {
                "fp_picked_locs": (
                    "$get_previous_module_result",  # get from previous module res
                    "fp_gold",
                ),
            },
        ),
        (  # 05
            "find_gold",
            {
                "diameter": 2,
                "std_range": 1.4,
                "mean_rmsd": 0.4,
                "remove_gold": True,
            },
        ),
        (  # 06
            "filter_locs",
            {
                "field": ["sx", "sy"],
                "minval": [0.8, 0.8],
                "maxval": [1.15, 1.15],
            },
        ),
        (  # 07
            "filter_locs",
            {
                "field": "ellipticity",
                "minval": 0,
                "maxval": 0.1,
            },
        ),
        (  # 08
            "render",
            {
                "fullfov_pixelsize": 400,
                "ctrmass_fov_nm": 20000,
                "ctrmass_pixelsize": 200,
            },
        ),
        (  # 09
            "summarize_dataset",
            {
                "methods": {
                    "nena": {},
                    "median-loc-precision": {},
                },
            },
        ),
        # (  # 10
        #     "dbscan",
        #     {
        #         "radius": (
        #             "$get_previous_module_result *1.4142",
        #             "nena, nena-nm"),
        #         "min_samples": 4,
        #         "continue_with_centers": False,
        #     },
        # ),
        # (  # 11
        #     "gaussian_mixture_cluster",
        #     {
        #         "min_locs": 15,
        #         "sigma_bounds": (0.8, 1.5),
        #         "callback_parent": "silent",
        #     }
        # ),
        (  # 10
            "smlm_clusterer",
            {
                "radius": (
                    "$get_previous_module_result *2.2",
                    "nena, nena-nm",
                ),
                "min_locs": 16,
                "basic_fa": False,
            },
        ),
        (  # 11
            "filter_transient_binding",
            {
                "meanframe_cutoff": 0.1,
                "stdframe_cutoff": 0.16,
                "fp_locs": (
                    "$get_previous_module_result",
                    "fp_clustered_locs",
                ),
            },
        ),
        (  # 12
            "nneighbor",
            {
                "dims": ["x", "y"],
                "nth_NN": 4,
                "nth_rdf": 6,
                "subsample_1stNN": 20,
                "add_column": True,
                "save_locs": True,
            },
        ),
        (  # 13
            "fit_csr",
            {
                "nneighbors": ("$get_previous_module_result", "nneighbors"),
                "dimensionality": 2,
                "min_dist": 50,
                "max_dist": 300,
                "bkg_fraction": 0.01,
                "kmin": 2,
                "save_locs": True,
            },
        ),
        (  # 14
            "save_single_dataset",
            {
                "filename": "locs.hdf5",
            },
        ),
    ]
    idx_last_sgl_module = len(workflow_modules_sgl) - 1
    workflow_modules_agg = [
        (  # 00
            "analysis_documentation",
            {},
        ),
        (  # 01
            "load_datasets_to_aggregate",
            {
                "tags": ("$$map", "#tags"),
                "filepaths": (
                    "$$get_prior_result",
                    "all_results, single_dataset, $$all,"
                    + f"{idx_last_sgl_module:02d}_save_single_dataset, filepath",
                ),
            },
        ),
        # (  # 02
        #     "align_channels",
        #     {
        #         "fp_fiducials": (
        #             "$$get_prior_result",
        #             "all_results, single_dataset, $$all, "
        #             + "05_find_gold, fp_gold"),
        #         "save_locs": True,
        #         "align_pars": {"force_method": "RCC"},
        #         "crop_boundaries": False,
        #     },
        # ),
        (  # 03
            "align_channels",
            {
                "fp_fiducials": (
                    "$get_previous_module_result",
                    "fp_fiducials",
                ),
                "save_locs": True,
                "align_pars": {
                    "max_shift": 4,
                    "force_method": "picked",
                },
                "crop_boundaries": True,
            },
        ),
        # (  # 02
        #     "dummy_module",
        #     {},
        # ),
        # (  # 03
        #     "align_channels",
        #     {
        #         "save_locs": True,
        #         "align_pars": {
        #             "force_method": "RSSO",
        #             "max_shift": 10,
        #             "plot_histogram": True,
        #         },
        #         "crop_boundaries": False,
        #     },
        # ),
        (  # 04
            "create_mask2",
            {
                "binsize": 20,
                "blursize": 400,
                "mask_pixel_size": 10,
                "threshold": 0.85,
                "select_cell": True,
                "fill_holes": True,
                "nth_largest_cell": 1,  # i
                "dilate_nm": 500,
                "apply_to_locs": True,
                "save_locs": True,
            },
        ),
        (  # 05
            "create_mask2",
            {
                "binsize": 1500,
                "blursize": 10,
                "mask_pixel_size": 1500,
                "threshold": 0.5,
                "select_cell": False,
                # "fill_holes": False,
                "apply_to_locs": True,
                "save_locs": False,
            },
        ),
        ("save_datasets_aggregated", {}),  # 05
        (  # 06
            "load_datasets_to_aggregate",
            {
                "tags": ("$$map", "#tags"),
                "filepaths": ("$get_previous_module_result", "filepaths"),
            },
        ),
        (
            "refine_mask_by_density",
            {
                "fp_mask": (
                    "$get_prior_result",
                    "results, 04_create_mask2, fp_mask",
                ),
                # "min_density": (
                #     "$sum *0.8",
                #     (
                #         "$$get_prior_result",
                #         "all_results, single_dataset, $$all, "
                #         + "11_nneighbor, density_rdf",
                #     )),
                # "max_density": (
                #     "$sum *1.2",
                #     (
                #         "$$get_prior_result",
                #         "all_results, single_dataset, $$all, "
                #         + "11_nneighbor, density_rdf",
                #     )),
                "density_std_cutoff": 1.5,
                "nbins": 30,
                "nth_largest": 0,
                "apply_to_locs": True,
                "smoothe_nm": 1500,
            },
        ),
        (  # 07
            "nneighbor",
            {
                "dims": ["x", "y"],
                "nth_NN": 4,
                "nth_rdf": 6,
                "subsample_1stNN": 10,
                "add_column": False,
                "save_locs": True,
            },
        ),
        (  # 08
            "fit_csr",
            {
                "nneighbors": ("$get_previous_module_result", "nneighbors"),
                "dimensionality": 2,
                "min_dist": 30,
                # "max_dist": 350,
                "bkg_fraction": 0.01,
                "kmin": 2,
                "save_locs": True,
                "fit_bkg": True,
            },
        ),
        (  # 09
            "labeling_efficiency_analysis",
            {
                "reference_name": ("$$index 0", "#tags"),
                "target_name": ("$$index 1", "#tags"),
                # "density": (
                #     "$get_prior_result",
                #     "results, 09_nneighbor, density_rdf",
                # ),
                "density": (
                    "$get_previous_module_result",
                    "density",
                ),
                "pair_distance": 10,
                "labeling_uncertainty": {
                    ("$$index 0", "#tags"): 6,
                    ("$$index 1", "#tags"): 4,
                },
                "n_simulate": 500000,
                "granularity": 100,
                "sim_repeats": 10,
            },
        ),
    ]

    # e.g. for multi dataset evaluation and aggregation
    workflow_modules_multi = {
        "single_dataset_tileparameters": datasets,
        "single_dataset_modules": workflow_modules_sgl,
        "aggregation_modules": workflow_modules_agg,
    }
    return workflow_modules_multi


if __name__ == "__main__":
    working_folder = os.path.dirname(os.path.abspath(__file__))

    coordinator = AggregationWorkflowCoordinator(
        src_loc,
        analysis_name,
        working_folder,
        confluence_url,
        confluence_space,
        confluence_token,
        base_page,
        always_save=False,
    )

    datasets = io.load_info(src_loc)
    workflow_modules_multi = get_workflow(datasets)

    coordinator.run_analysis(workflow_modules_sgl, workflow_modules_agg)
    # copy this file to save the settings/parameters
    srcloc = os.path.abspath(__file__)
    destloc = os.path.join(coordinator.root_folder, os.path.split(srcloc)[1])
    shutil.copyfile(srcloc, destloc)
