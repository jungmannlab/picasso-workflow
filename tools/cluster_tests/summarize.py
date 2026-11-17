#!/usr/bin/env python
"""Compatibility shim -- the summarizer moved into the package.

The real implementation now lives in
``picasso_workflow/cluster_tests/summarize.py`` so it ships in the wheel /
Apptainer image (and can be staged to a cluster run directory). This shim keeps
the documented host path working:

    python3 tools/cluster_tests/summarize.py <run_dir>

It execs the relocated file BY PATH (a repo-relative sibling) rather than
importing ``picasso_workflow``, so it stays standard-library only and does not
pull the heavy analysis import chain -- exactly like running the staged copy
under a bare ``python3`` on a cluster node.
"""

import os
import runpy

_REAL = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "picasso_workflow",
        "cluster_tests",
        "summarize.py",
    )
)

if __name__ == "__main__":
    runpy.run_path(_REAL, run_name="__main__")
