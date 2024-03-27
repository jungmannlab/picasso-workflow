#!/usr/bin/env python
"""
Module Name: 01_sgl_dataset_workflow.py
Author: Heinrich Grabmayr
Initial Date: March 18, 2024
Description: This module shows an example of how to use a single dataset
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
        in reporter_config/ConfluenceReporter. However, saving tokens in plain
        text is not recommended.
    - enter the values required below.
    - in terminal, with the conda environment active, go to the examples
        folder and execute
        python 240318_aggregation_workflow.py
    - see your confluence page for the results.

    Analysis is performed according to the modules listed in the
    workflow_modules, together with the respective parameters.
    For a list of available modules, check out
    util.AbstractModuleCollection
"""
import os
from picasso_workflow import WorkflowRunner


# the URL of your confluence instance
confluence_url = ""
# your confluence space (can be seen in Space Tools -> Overview -> Key)
confluence_space = ""
# the page under which the report should be generated
parent_page_title = ""
# the name under which the report should be generated
report_name = "test_report"


# the directory where the analysis files should be stored
result_location = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"
)

# if you have the .env file set up (copy from .env_template),
# you can use its information instead of manually inputting it here
# if not, comment this out
confluence_url = os.getenv("TEST_CONFLUENCE_URL")
confluence_token = os.getenv("TEST_CONFLUENCE_TOKEN")
confluence_space = os.getenv("TEST_CONFLUENCE_SPACE")
parent_page_title = os.getenv("TEST_CONFLUENCE_PAGE")


data_location = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "picasso_workflow",
    "tests",
    "TestData",
    "integration",
)


reporter_config = {
    "report_name": report_name,
    "ConfluenceReporter": {
        "base_url": confluence_url,
        "space_key": confluence_space,
        "parent_page_title": parent_page_title,
        "token": confluence_token,
    },
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
        "load_dataset_movie",
        {
            "filename": os.path.join(
                data_location,
                "3C_30px_1kframes_1",
                "3C_30px_1kframes_MMStack_Pos0.ome.tif",
            ),
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
                    "results, 00_load_dataset_movie, sample_movie, "
                    + "sample_frame_idx",
                ),
                "box_size": 7,
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
        "save_single_dataset",
        {
            "filename": "locs.hdf5",
        },
    ),
    (
        "manual",
        {
            "prompt": "Please manually undrift.",
            "filename": "locs_undrift.hdf5",
        },
    ),
    (
        "load_dataset_localizations",
        {
            "filename": (
                "$get_prior_result",  # get from prior results
                "results, 04_manual, filepath",
            ),
        },
    ),
    (
        "save_single_dataset",
        {
            "filename": "locs.hdf5",
        },
    ),
]


if __name__ == "__main__":
    wr = WorkflowRunner.config_from_dicts(
        reporter_config,
        analysis_config,
        workflow_modules_sgl,
        continue_previous_runner=True,
    )
    wr.run()
    # wr.save()
