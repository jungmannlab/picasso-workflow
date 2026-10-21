#!/usr/bin/env python
"""The binding-events feature is importable at its new location.

Regression guard for the Phase 2 move of ``binding_event_analysis`` /
``binding_event_plotting`` into ``outpost/binding_events/`` and the fix to
``analyse.py``'s previously-broken
``outpost_modules.binding_event_analysis.Measurement`` reference (that submodule
was never imported into the ``outpost_modules`` package, so the call raised
``AttributeError`` at runtime -- but nothing exercised it, so no test caught it).

Needs the optional ``[binding-events]`` extra (``lmfit``, ``pandarallel``);
skipped when it is not installed.
"""

import pytest

pytest.importorskip("lmfit", reason="[binding-events] extra not installed")
pytest.importorskip(
    "pandarallel", reason="[binding-events] extra not installed"
)


def test_binding_events_classes_importable():
    """The classes ``analyse.binding_event_analysis`` imports resolve."""
    from picasso_workflow.outpost.binding_events.analysis import Measurement
    from picasso_workflow.outpost.binding_events.plotting import Plotting

    assert callable(Measurement) and callable(Plotting)


def test_measurement_constructor_accepts_call_site_kwargs():
    """``Measurement(...)`` accepts exactly the kwargs ``analyse`` passes.

    Guards the ``saving_path=`` -> ``saving_name=`` fix: the call site used
    ``saving_path``, which the constructor never accepted (``TypeError``).
    """
    import inspect

    from picasso_workflow.outpost.binding_events.analysis import Measurement

    params = inspect.signature(Measurement.__init__).parameters
    for kwarg in ("in_path", "save_path", "saving_name", "total_n_frames"):
        assert kwarg in params, kwarg
    assert "saving_path" not in params


def test_plotting_api_matches_call_site():
    """``Plotting`` exposes exactly the members ``analyse`` constructs/calls.

    Guards the wrong-case method fixes (``plot_td`` / ``plot_r`` /
    ``saveAllResults`` -> ``Plot_td`` / ``Plot_r`` / ``SaveAllResults``).
    """
    import inspect

    from picasso_workflow.outpost.binding_events.plotting import Plotting

    params = inspect.signature(Plotting.__init__).parameters
    for kwarg in (
        "table_g",
        "table_k",
        "show",
        "save",
        "saving_name",
        "total_n_frames",
    ):
        assert kwarg in params, kwarg

    for method in (
        "Plot_photons",
        "Plot_bg",
        "Plot_sbr",
        "Plot_tb",
        "Plot_td",
        "Plot_r",
        "Plot_locs",
        "SaveAllResults",
    ):
        assert callable(getattr(Plotting, method, None)), method


def test_measurement_methods_exist():
    """``Measurement`` exposes the methods ``analyse`` calls (``Begin`` /
    ``FileSaver``)."""
    from picasso_workflow.outpost.binding_events.analysis import Measurement

    for method in ("Begin", "FileSaver"):
        assert callable(getattr(Measurement, method, None)), method
