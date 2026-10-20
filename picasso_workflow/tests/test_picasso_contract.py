#!/usr/bin/env python
"""Contract test for the picasso API surface picasso-workflow depends on.

This runs in the *unit* tier (picassosr is a declared runtime dependency,
importable without data or network) so it gates every PR on the required hosted
check.  It asserts that the public picasso surface recorded in
:mod:`picasso_workflow._picasso_api` still exists in the installed picassosr --
turning an upstream rename, or a function going private, into a loud and
specific CI failure here instead of a silent breakage at runtime on the cluster.

When this test fails after a picassosr bump, the fix is one of: adjust the
picassosr pin in ``pyproject.toml``, or update the boundary in
``picasso_workflow/_picasso_api.py`` (and the call sites it guards).
"""

import importlib
import inspect

import pytest

from picasso_workflow import _picasso_api as api

# picassosr is a declared runtime dependency; importorskip keeps the suite
# runnable in the (unsupported) case it is absent, preserving the unit tier's
# "runs anywhere" property.
pytest.importorskip(
    "picasso", reason="picassosr required for the picasso API contract test"
)


@pytest.mark.parametrize("submodule", api.PUBLIC_SUBMODULES)
def test_public_submodule_importable(submodule):
    """Every picasso submodule the workflow imports must still exist."""
    importlib.import_module(f"picasso.{submodule}")


def test_top_level_attrs_present():
    """Top-level picasso attributes the workflow reads must exist."""
    import picasso

    missing = [a for a in api.TOP_LEVEL_ATTRS if not hasattr(picasso, a)]
    assert not missing, f"picasso is missing top-level attribute(s): {missing}"


@pytest.mark.parametrize(
    "submodule, attrs", sorted(api.SUBMODULE_ATTRS.items())
)
def test_submodule_attrs_present(submodule, attrs):
    """Specific attributes the workflow reads off a submodule must exist."""
    mod = importlib.import_module(f"picasso.{submodule}")
    missing = [a for a in attrs if not hasattr(mod, a)]
    assert not missing, f"picasso.{submodule} is missing: {missing}"


@pytest.mark.parametrize(
    "dotted, params", sorted(api.SIGNATURE_CONTRACT.items())
)
def test_signature_params_present(dotted, params):
    """Parameter names the workflow passes must still exist upstream."""
    modname, fname = dotted.rsplit(".", 1)
    fn = getattr(importlib.import_module(f"picasso.{modname}"), fname)
    present = set(inspect.signature(fn).parameters)
    missing = [p for p in params if p not in present]
    assert not missing, (
        f"picasso.{dotted} no longer accepts {missing} "
        f"(current params: {sorted(present)})"
    )


@pytest.mark.parametrize("submodule", api.PUBLIC_SUBMODULES)
def test_boundary_reexports_submodule(submodule):
    """The boundary lazily re-exports each submodule as the real picasso one."""
    assert getattr(api, submodule) is importlib.import_module(
        f"picasso.{submodule}"
    )


def test_boundary_reexports_top_level_attrs():
    """The boundary re-exports each top-level attr as picasso's own."""
    import picasso

    for name in api.TOP_LEVEL_ATTRS:
        assert getattr(api, name) is getattr(picasso, name)


def test_fit_method_map_accessor():
    """The guarded ``_FIT_METHOD_MAP`` accessor returns a non-empty mapping."""
    fit_map = api.get_fit_method_map()
    assert isinstance(fit_map, dict) and fit_map


def test_spinna_batch_analysis_accessor():
    """The guarded ``_spinna_batch_analysis`` accessor returns a callable."""
    assert callable(api.get_spinna_batch_analysis())
