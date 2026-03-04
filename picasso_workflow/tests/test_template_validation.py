#!/usr/bin/env python
"""
Module Name: test_template_validation.py
Author: Heinrich Grabmayr
Description: Level A template validation — structural tests that verify every
    module name referenced in a snapshotted production template exists in
    AutoPicasso.

These tests require neither picasso nor any data files.  They are part of
the normal (non-integration) test suite and run in CI without restrictions.

When tests/TestData/templates/ is empty (i.e. no snapshots have been
committed yet), pytest collects zero parametrized cases and the file is
effectively a no-op — it never causes CI failures on fresh clones.

To populate TestData/templates/ run:
    python tools/snapshot_templates.py
(requires network access to the mounted pool volumes)
"""
import importlib.util
import inspect
import logging
import os

import pytest

from picasso_workflow import util


logger = logging.getLogger(__name__)

_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "TestData",
    "templates",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _available_modules():
    """Return the set of public module names in AbstractModuleCollection."""
    return {
        name
        for name, obj in inspect.getmembers(util.AbstractModuleCollection)
        if (inspect.isfunction(obj) or inspect.ismethod(obj))
        and not name.startswith("_")
    }


def _template_files():
    """Return sorted (template_name, path_to_start_workflow.py) pairs."""
    if not os.path.isdir(_TEMPLATES_DIR):
        return []
    result = []
    for name in sorted(os.listdir(_TEMPLATES_DIR)):
        path = os.path.join(_TEMPLATES_DIR, name, "start_workflow.py")
        if os.path.isfile(path):
            result.append((name, path))
    return result


def _import_template(path):
    """Dynamically import a start_workflow.py and return the module object."""
    spec = importlib.util.spec_from_file_location("_start_workflow", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extract_module_names(mod):
    """Collect every workflow module name declared in a start_workflow module.

    Handles all three naming conventions used in the codebase:
      - workflow_modules_sgl  (single-dataset list)
      - workflow_modules_agg  (aggregation list, legacy)
      - workflow_modules_multi (dict with 'single_dataset_modules' and
                                'aggregation_modules' keys)
    """
    names = []

    def _collect(seq):
        if not seq:
            return
        for item in seq:
            if isinstance(item, (tuple, list)) and len(item) >= 1:
                names.append(item[0])

    _collect(getattr(mod, "workflow_modules_sgl", None))
    _collect(getattr(mod, "workflow_modules_agg", None))

    multi = getattr(mod, "workflow_modules_multi", None)
    if isinstance(multi, dict):
        _collect(multi.get("single_dataset_modules"))
        _collect(multi.get("aggregation_modules"))

    return names


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,path", _template_files())
def test_template_modules_exist(name, path):
    """Every module referenced in the snapshotted template must exist in
    AutoPicasso (i.e. be a public method of AbstractModuleCollection).

    Failure here means a module was renamed or removed from the codebase
    while a production template still references the old name.
    """
    try:
        mod = _import_template(path)
    except Exception as exc:
        pytest.fail(
            f"Template '{name}': could not import start_workflow.py: {exc}"
        )

    module_names = _extract_module_names(mod)
    if not module_names:
        pytest.skip(
            f"Template '{name}': no workflow module names found — "
            "check start_workflow.py uses module-level assignments"
        )

    available = _available_modules()
    unknown = [n for n in module_names if n not in available]

    assert not unknown, (
        f"Template '{name}' references unknown module(s): {unknown}\n"
        "Either the module was renamed/removed from AutoPicasso, or the "
        "template snapshot is out of date (re-run tools/snapshot_templates.py)."
    )
