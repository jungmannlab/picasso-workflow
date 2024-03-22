#!/usr/bin/env python
"""
Module Name: 01_sgl_dataset_workflow.py
Author: Heinrich Grabmayr
Initial Date: March 21, 2024
Description: This module shows an example of how to use a predefined standard
    workflow of picasso-workflow. Use your own data
    In order to get the example to work, you need to have the system set up
    as described in the headers in the 01_intro examples.
"""
import os
from picasso_workflow import WorkflowRunner, standard_singledataset_workflows


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
    "gpufit_installed": False,
}

workflow_modules_sgl = standard_singledataset_workflows.minimal(
    filename="/path/to/my/file"
)
# adapt the configuration
workflow_modules_sgl[0][1]["load_camera_info"] = True

if __name__ == "__main__":
    wr = WorkflowRunner.config_from_dicts(
        reporter_config,
        analysis_config,
        workflow_modules_sgl,
        continue_previous_runner=True,
    )
    wr.run()
