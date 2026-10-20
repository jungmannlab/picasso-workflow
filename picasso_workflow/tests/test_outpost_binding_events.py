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
