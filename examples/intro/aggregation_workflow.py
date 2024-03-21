#!/usr/bin/env python
"""
Module Name: 240318_aggregation_workflow.py
Author: Heinrich Grabmayr
Initial Date: March 18, 2024
Description: This module shows an example of how to use an aggregation
    workflow of picasso-workflow. It uses the test data of the package.
    In order to get the example to work, do the following:
    - install the package picasso-workflow and its dependencies, e.g.
        into a conda environment.
    - in your Confluence Space, generate an API token (depending on
        the Confluence version e.g. in your profile -> settings ->
        Personal Access Tokens -> Create Token)
    - For long term usage: Save the API token as an environment variable.
        - in Windows, open the command line window and enter
            setx CONFLUENCE_BEARER <your_confluence_api_token>
        - on MacOS, open terminal and enter
            echo $SHELL
            if the output is /bin/zsh, enter:
                open -a TextEdit ~/.zshrc
                and set the environment variable
            if the output is /bin/bash, enter:
                open -a TextEdit ~/.bash_profile
                and set the environment variable
            in either case, the environment variable is set by the command
            export CONFLUENCE_BEARER="<your_confluence_api_token>"
            as an alternative to TextEdit, and on linux, you can use vi
            to open the file within the terminal.
    - For quick tests, add the field 'token': '<your_confluence_api_token>'
        in reporter_config/ConfluenceReporter.
    - enter the values required below.
    - in terminal, with the conda environment active, go to the examples
        folder and execute
        python 240318_aggregation_workflow.py
    - see your confluence page for the results.
"""
from picasso_workflow import AggregationWorkflowRunner


# the URL of your confluence instance
confluence_url = ""
# your confluence space (can be seen in Space Tools -> Overview -> Key)
confluence_space = ""
# the page under which the report should be generated
parent_page_title = ""
# the name under which the report should be generated
report_name = ""


# the directory where the analysis files should be stored
result_location = ""


reporter_config = {
    "ConfluenceReporter": {
        "base_url": confluence_url,
        "space_key": confluence_space,
        "parent_page_title": parent_page_title,
        "report_name": report_name,
    }
}

analysis_config = {
    "result_location": result_location,
    "camera_info": {
        "gain": 1,
        "sensitivity": 0.45,
        "baseline": 100,
        "qe": 0.82,
        "pixelsize": 130,  # nm
    },
    "gpufit_installed": False,
}

# e.g. for single dataset evaluation
workflow_modules_sgl = [
    (
        "load",
        {
            "filename": ("$map", "filename"),
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
                    "results, load, sample_movie, sample_frame_idx",
                ),
                "start_ng": -3000,
                "zscore": 5,
            },
            "ids_vs_frame": {"filename": "ids_vs_frame.png"},
            "box_size": 7,
        },
    ),
    # ('identify', {
    #     'net_gradient': 5000,
    #     'ids_vs_frame': {
    #         'filename': 'ids_vs_frame.png'
    #     },
    #     'box_size': 7,
    #     },
    # ),
    ("localize", {"fit_method": "lsq", "box_size": 7, "fit_parallel": True}),
    (
        "undrift_rcc",
        {
            "segmentation": 1000,
            "max_iter_segmentations": 4,
            "filename": "drift.csv",
            "save_locs": {"filename": "locs_undrift.hdf5"},
        },
    ),
    (
        "manual",
        {
            "prompt": "Please manually undrift.",
            "filename": "locs_undrift.hdf5",
        },
    ),
    # ('segmentation', {
    #     'method': 'brightfield',
    #     'parameters': {
    #         'filename': 'BF.png'}
    #     },
    # ),
    # ('cluster_smlm', {
    #     'min_locs': 10,
    #     'cluster_radius': 4
    #     },
    # ),
]


# for dataset aggregation, after they have been analyzed separately
workflow_modules_agg = [
    (
        "load_datasets_to_aggregate",
        {
            "tags": ("$map", "#tags"),
            "evaldirs": ["resiround1/eval", "resiround2/eval"],
        },
    ),
    ("RESI", {"evaldirs": ["resiround1/eval", "resiround2/eval"]}),
]


# e.g. for multi dataset evaluation and aggregation
workflow_modules_multi = {
    "single_dataset_tileparameters": {
        "#tags": ["RESIround1", "RESIround2"],
        "filename": ["fn1", "fn2"],
    },
    "single_dataset_modules": workflow_modules_sgl,
    "aggregation_modules": [
        (
            "load",
            {
                "evaldirs": (
                    "$get_prior_results",
                    "all_results, $all, undrift_rcc, locs_undrift.hdf5",
                )
            },
        )
    ],
}


if __name__ == "__main__":
    awr = AggregationWorkflowRunner.config_from_dicts(
        reporter_config, analysis_config, workflow_modules_multi
    )
    awr.run()
    awr.save()
