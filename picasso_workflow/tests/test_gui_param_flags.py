#!/usr/bin/env python
"""Tests for the GUI's changed-parameter flagging mechanism.

Parameters whose behaviour/units changed in a release are flagged in the GUI
with an amber marker that auto-expires a few releases later
(``gui.PARAM_CHANGE_FLAGS`` + ``gui._param_flag_active``). These tests cover
the pure expiry logic and guard against the registry drifting out of sync with
the module descriptors.

Importing ``picasso_workflow.gui`` pulls in PyQt6, so the whole module is
skipped where that (or the GUI import) is unavailable.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI tests")

try:
    from picasso_workflow import gui  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"gui import failed: {exc}", allow_module_level=True)


def test_param_flag_active_expiry():
    """``_param_flag_active`` shows/hides a 0.7.0 flag across the horizon.

    The horizon is ``PARAM_FLAG_HORIZON_MINORS`` minor releases, so a flag
    added in 0.7.0 expires at 0.10.0. An unknown/garbage running version
    fails open (stays visible).
    """
    # Fail-open for unknown / missing versions.
    assert gui._param_flag_active("0.7.0", current="unknown") is True
    assert gui._param_flag_active("0.7.0", current="") is True
    assert gui._param_flag_active("0.7.0", current=None) is True
    # Garbage still fails open.
    assert gui._param_flag_active("0.7.0", current="not-a-version") is True

    # Within the horizon -> still shown.
    assert gui._param_flag_active("0.7.0", current="0.7.0") is True
    assert gui._param_flag_active("0.7.0", current="0.7.3") is True
    assert gui._param_flag_active("0.7.0", current="0.9.9") is True

    # At or past the horizon (0.10.0) -> hidden.
    assert gui._param_flag_active("0.7.0", current="0.10.0") is False
    assert gui._param_flag_active("0.7.0", current="1.2.0") is False


def test_param_change_flags_keys_exist():
    """Every flagged (module, param) must still resolve to a rendered param.

    The key must appear as a top-level parameter of the module's descriptor
    or as a nested ``properties`` sub-key. This guards against a flag that
    points at a param that has since been renamed or removed.
    """
    descriptor = gui.ModuleDescriptor()
    for (module_name, param_name), flag in gui.PARAM_CHANGE_FLAGS.items():
        params_spec, _ = getattr(descriptor, module_name)()
        found = param_name in params_spec
        if not found:
            # Look one level deep into nested dict parameters.
            for meta in params_spec.values():
                props = (
                    meta.get("properties", {})
                    if isinstance(meta, dict)
                    else {}
                )
                if param_name in props:
                    found = True
                    break
        assert found, (
            f"PARAM_CHANGE_FLAGS key ({module_name!r}, {param_name!r}) "
            "does not resolve to a descriptor parameter"
        )
        assert "changed_in" in flag and "message" in flag


def test_fit_csr_form_matches_analyse():
    """The fit_csr form exposes exactly the params analyse.fit_csr reads."""
    descriptor = gui.ModuleDescriptor()
    params_spec, _ = descriptor.fit_csr()
    assert set(params_spec) == {
        "dimensionality",
        "kmin",
        "kmax",
        "min_dist",
        "max_dist",
        "n_simulations",
        "gof_wasserstein_warn",
    }
