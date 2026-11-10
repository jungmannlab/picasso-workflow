#!/usr/bin/env python
"""Regression tests for the alignment shift-kernel peak finder.

Covers the ``_find_peak_center_of_mass`` fallback-tuple fix: the
``total_mass == 0`` fallback now returns an 8-tuple (matching the empty- and
normal-return arity the callers unpack), rather than the previous 6-tuple that
would raise ``ValueError: not enough values to unpack`` at the call sites in
``_calculate_pairwise_shift``.
"""

import numpy as np

from picasso_workflow.outpost.alignment.shift_kernel import (
    _find_peak_center_of_mass,
)


def _edges(n, span=5.0):
    return np.linspace(-span, span, n + 1)


def test_find_peak_center_of_mass_empty_histogram_returns_8_tuple():
    """The all-zero histogram fallback returns an 8-tuple.

    An all-zero histogram leaves no non-zero bins, so the function takes the
    first (empty-histogram) fallback. This is the sibling of the fixed
    ``total_mass == 0`` branch and must have the same 8-value arity, ending in
    ``(threshold=0.0, use_threshold=False)``.
    """
    hist = np.zeros((10, 10))
    out = _find_peak_center_of_mass(
        hist, _edges(10), _edges(10), max_shift=5.0, snr_threshold=3.0
    )
    assert len(out) == 8
    # Last two are (threshold, use_threshold).
    assert out[-2] == 0.0
    assert out[-1] is False


def test_find_peak_center_of_mass_normal_returns_8_tuple():
    """The normal (non-fallback) return path also returns an 8-tuple.

    A peaked histogram above the median threshold exercises the center-of-mass
    computation. All return paths of ``_find_peak_center_of_mass`` must agree
    on arity so the callers' fixed 8-value unpack is correct.
    """
    hist = np.ones((20, 20))
    hist[10, 10] = 100.0  # clear peak well above the median
    out = _find_peak_center_of_mass(
        hist, _edges(20), _edges(20), max_shift=5.0, snr_threshold=3.0
    )
    assert len(out) == 8
    # Center of mass of a near-symmetric peak lands near the origin.
    shift_x, shift_y = out[0], out[1]
    assert abs(shift_x) < 5.0 and abs(shift_y) < 5.0
