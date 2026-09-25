#!/usr/bin/env python
"""GUI tests for the per-parameter default/override toggle.

A parameter with a spec ``default`` renders with a clickable label and starts
in the "use default" state: the widget is greyed/disabled and the value is
omitted from the generated workflow, so the code default applies. Clicking the
label (``_apply_param_default_state``) switches to an explicit override.

Skips where a Qt GUI cannot be constructed.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI tests")

from PyQt6 import QtWidgets  # noqa: E402

from picasso_workflow import gui  # noqa: E402

SPEC = {
    "pattern_min_samples": {"type": "int", "default": 7, "min": 2},
    "pattern_min_sites_frac": {"type": "float", "default": 0.66, "min": 0.0},
    "allow_mirror": {"type": "bool", "default": True},
    "design_file": {"type": "str", "required": True},  # no default
}


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    try:
        win = gui.Window()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not construct GUI window: {e}")
    # a lightweight stand-in for the auto-save so toggling does not drag in
    # the full workflow-item machinery
    win._on_parameter_changed = lambda: None
    yield win
    win.close()


def _build(win):
    win.parameter_widgets = {}
    for name, meta in SPEC.items():
        win.parameter_widgets[name] = win._create_parameter_row(name, meta, 0)


def _collect(win):
    vals = {}
    for name, wi in win.parameter_widgets.items():
        v = win._get_widget_value(wi.widget, wi.original_type, wi)
        if v is not None:
            vals[name] = v
    return vals


def test_new_module_defaults_greyed_and_omitted(window):
    """Defaulted params start disabled and are omitted from the workflow;
    a required no-default param is still collected."""
    _build(window)
    collected = _collect(window)
    assert "pattern_min_samples" not in collected
    assert "pattern_min_sites_frac" not in collected
    assert "allow_mirror" not in collected
    assert "design_file" in collected
    for name in ("pattern_min_samples", "pattern_min_sites_frac"):
        wi = window.parameter_widgets[name]
        assert wi.use_default is True
        assert wi.widget.isEnabled() is False


def test_toggle_to_override_writes_value(window):
    """Overriding a param (as a label click does) writes it, pre-filled with
    the default."""
    _build(window)
    wi = window.parameter_widgets["pattern_min_sites_frac"]
    window._apply_param_default_state(wi, False, persist=True)
    assert wi.widget.isEnabled() is True
    assert _collect(window).get("pattern_min_sites_frac") == 0.66


def test_reload_distinguishes_override_from_default(window):
    """A stored value equal to the default is greyed/omitted; a different
    value is an explicit override; an absent param falls back to default."""
    _build(window)
    window._populate_stored_parameters(
        {"pattern_min_samples": 15, "pattern_min_sites_frac": 0.66}
    )
    collected = _collect(window)
    assert collected.get("pattern_min_samples") == 15
    assert "pattern_min_sites_frac" not in collected  # == default -> omitted
    assert "allow_mirror" not in collected  # absent -> default
    assert window.parameter_widgets["pattern_min_samples"].use_default is False
    assert (
        window.parameter_widgets["pattern_min_sites_frac"].use_default is True
    )


def test_explicit_nondefault_zero_is_kept(window):
    """The original bug: an explicit 0.0 (!= default 0.66) is a real override
    and must not be dropped."""
    _build(window)
    window._populate_stored_parameters({"pattern_min_sites_frac": 0.0})
    assert _collect(window).get("pattern_min_sites_frac") == 0.0
