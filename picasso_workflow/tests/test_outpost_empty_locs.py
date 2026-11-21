#!/usr/bin/env python
"""Robustness guards for empty / degenerate localization sets.

Reporting histograms and the render contrast-scaler used to crash when an
upstream filter (or picker) removed every localization: ``np.ptp([])`` /
``array.max()`` on an empty field, and ``min([])`` when every rendered image is
all-zero. These surfaced as hard failures of the ``filter_locs`` and ``render``
workflow modules on sparse real data. The fixes degrade gracefully instead;
these tests pin that behaviour.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from picasso_workflow.outpost.reporting import histograms  # noqa: E402
from picasso_workflow.outpost.render import render  # noqa: E402


def test_scale_contrast_all_zero_images_does_not_raise():
    """All-zero images (nothing rendered) scale to zero, not min([])."""
    images = [np.zeros((4, 4)), np.zeros((4, 4))]
    out = render.scale_contrast(images)
    assert np.all(np.asarray(out) == 0.0)
    assert np.asarray(out).shape == (2, 4, 4)


def test_scale_contrast_mixed_still_scales():
    """A non-zero image is still contrast-scaled (regression guard)."""
    img = np.zeros((4, 4))
    img[0, 0] = 80.0  # max 80 -> upper = 80/40 = 2 -> 80/2 clipped to 1.0
    out = render.scale_contrast([img])
    assert np.isclose(np.asarray(out)[0].max(), 1.0)


def test_plot_1dhist_empty_field_does_not_raise():
    """An all-NaN field annotates instead of crashing in np.ptp([])."""
    locs = {"x": np.array([np.nan, np.nan, np.nan])}
    fig, ax = plt.subplots()
    histograms.plot_1dhist(locs, "x", fig, ax)  # must not raise
    plt.close(fig)


def test_plot_2dhist_empty_fields_does_not_raise():
    """An empty x/y pair annotates instead of crashing in hist2d/np.ptp."""
    locs = {"x": np.array([]), "y": np.array([])}
    fig, ax = plt.subplots()
    histograms.plot_2dhist(locs, "x", "y", fig, ax)  # must not raise
    plt.close(fig)


def test_plot_2dhist_partial_nan_still_plots():
    """Finite rows survive the NaN filter and are histogrammed."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    x[:5] = np.nan  # a few non-finite rows the filter must drop
    y[3:8] = np.nan
    locs = {"x": x, "y": y}
    fig, ax = plt.subplots()
    histograms.plot_2dhist(locs, "x", "y", fig, ax)  # finite pairs remain
    plt.close(fig)
