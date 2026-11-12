"""Cluster-test orchestration helpers that ship inside the package.

``summarize`` condenses a cluster test run's artefacts into ``SUMMARY.txt`` +
``summary.json``. It lives here (rather than only under ``tools/``) so it ships
in the wheel / Apptainer image and can be staged to a cluster run directory and
run with a bare ``python3`` -- it is deliberately standard-library only, so
executing the file by path never triggers the heavy ``picasso_workflow`` import
chain.
"""
