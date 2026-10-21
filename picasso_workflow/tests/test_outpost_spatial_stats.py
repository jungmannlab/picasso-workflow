#!/usr/bin/env python
"""The Ripley's-K feature is importable at its new ``spatial_stats`` location.

Regression guard for the Phase 2 move of ``ripleys`` and
``ripley_dcatlas_analysis`` out of the flat ``outpost_modules/`` into
``outpost/spatial_stats/``. ``analyse.AutoPicasso.ripleysk2`` imports ``ripleys``
from the new path and calls the functions asserted below; the unit tier mocks
``AutoPicasso``, so without this guard a broken move would surface only at
runtime on the cluster.
"""


def test_ripleys_importable_at_new_location():
    """The functions ``ripleysk2`` calls resolve at the new path."""
    from picasso_workflow.outpost.spatial_stats import ripleys

    for name in (
        "convert_picasso_to_coords",
        "typefraction_all_channels",
        "analyze_all_channels",
        "init_plot",
        "plot_ripleys",
    ):
        assert callable(getattr(ripleys, name)), name


def test_ripley_dcatlas_importable_at_new_location():
    """The DC-Atlas pipeline imports (incl. its ``ripleys`` dependency)."""
    from picasso_workflow.outpost.spatial_stats import (
        ripley_dcatlas_analysis,
    )

    assert callable(ripley_dcatlas_analysis.init_plot)


def test_ripleys_not_left_in_outpost_modules():
    """The old flat ``outpost_modules.ripleys`` attribute is gone."""
    import picasso_workflow.outpost_modules as outpost_modules

    assert not hasattr(outpost_modules, "ripleys")
