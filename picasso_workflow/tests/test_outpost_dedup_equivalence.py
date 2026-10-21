#!/usr/bin/env python
"""Equivalence tests gating the outpost -> picasso 0.12 de-duplication.

Before a forked ``picasso_outpost`` function is rewired to its picasso upstream,
a test here pins the *contract* (picked membership / group assignment), so a
behavioural difference between the fork and the upstream call is caught here.
The unit tier otherwise mocks ``AutoPicasso`` and would not exercise it.

These run in the unit tier (picassosr is a declared dependency, no data/network).
"""

import numpy as np
import pandas as pd
import pytest

from picasso_workflow.outpost.picks import picks as picks_mod
from picasso_workflow._picasso_api import postprocess
from picasso_workflow._picasso_api import masking as picasso_masking

pytest.importorskip(
    "picasso", reason="picassosr required for the dedup equivalence tests"
)

# Pick centers and a diameter small enough that each center's tight cluster is
# well inside the radius and the background is well outside -- so membership is
# unambiguous regardless of the exact boundary convention.
_CENTERS = [(20.0, 30.0), (60.0, 80.0), (100.0, 40.0), (33.0, 110.0)]
_PICK_DIAMETER = 1.0  # radius 0.5


def _synthetic_locs():
    rng = np.random.default_rng(7)
    rows = []
    for cx, cy in _CENTERS:  # tight clusters well inside the pick radius
        for _ in range(50):
            rows.append(
                (
                    rng.integers(0, 1000),
                    cx + rng.normal(0, 0.05),
                    cy + rng.normal(0, 0.05),
                    float(rng.poisson(1000)),
                    1.5,
                    1.5,
                    10.0,
                    0.02,
                    0.02,
                )
            )
    for _ in range(300):  # background far from every center (> radius away)
        x, y = rng.uniform(0, 128), rng.uniform(0, 128)
        if all((x - cx) ** 2 + (y - cy) ** 2 > 4.0 for cx, cy in _CENTERS):
            rows.append(
                (
                    rng.integers(0, 1000),
                    x,
                    y,
                    float(rng.poisson(1000)),
                    1.5,
                    1.5,
                    10.0,
                    0.02,
                    0.02,
                )
            )
    locs = pd.DataFrame(
        rows,
        columns=["frame", "x", "y", "photons", "sx", "sy", "bg", "lpx", "lpy"],
    )
    info = [{"Frames": 1000, "Width": 128, "Height": 128, "Pixelsize": 130}]
    return locs, info


def _membership(df):
    return set(map(tuple, df[["frame", "x", "y"]].round(6).to_numpy()))


def test_picked_locs_group_contract():
    """Each group holds exactly the locs within pick_diameter/2 of its center.

    Implementation-independent: pins what ``picked_locs`` must return regardless
    of whether the pick-finding is the in-house loop or the picasso call.
    Checked by validity (every picked loc lies within the radius) plus
    completeness by count (every loc within the radius is picked) -- which
    together prove set-equality without matching possibly-retyped coordinates.
    """
    locs, info = _synthetic_locs()
    r = _PICK_DIAMETER / 2
    result = picks_mod.picked_locs(
        locs, info, _CENTERS, _PICK_DIAMETER, add_group=True
    )
    for i, (cx, cy) in enumerate(_CENTERS):
        got = result[result["group"] == i]
        d2 = (got["x"] - cx) ** 2 + (got["y"] - cy) ** 2
        assert (d2 <= r**2 + 1e-6).all(), f"group {i} has a loc outside radius"
        n_within = int(
            ((locs["x"] - cx) ** 2 + (locs["y"] - cy) ** 2 <= r**2).sum()
        )
        assert len(got) == n_within, f"group {i}: {len(got)} != {n_within}"


def test_picked_locs_matches_upstream_membership():
    """Fork membership matches picasso.postprocess.picked_locs at pick_size=d/2.

    Guards the arg mapping the dedup relies on (``pick_diameter`` ->
    ``pick_shape="Circle", pick_size=pick_diameter/2``).
    """
    locs, info = _synthetic_locs()
    fork = picks_mod.picked_locs(locs, info, _CENTERS, _PICK_DIAMETER)
    upstream = pd.concat(
        postprocess.picked_locs(
            locs,
            info,
            _CENTERS,
            "Circle",
            pick_size=_PICK_DIAMETER / 2,
            add_group=True,
        ),
        ignore_index=True,
    )
    assert _membership(fork) == _membership(upstream)


def test_pick_similar_analysis_numba_compiles():
    """The numba-jitted pick_similar path runs end-to-end.

    Regression guard: ``pick_similar_analysis`` is ``@nb.jit(nopython=True)`` and
    calls picasso's private ``_n_block_locs_at``. That reach-in is routed through
    the boundary (INTERNAL_SYMBOLS) but must be resolved to a *module-level*
    dispatcher global -- nopython mode cannot call ``get_internal`` directly.
    The unit tier mocks ``AutoPicasso``, so nothing else actually invokes this
    jitted function; this test does, forcing numba to compile it.
    """
    locs, info = _synthetic_locs()
    x_sim, y_sim, rmsds, nlocs = picks_mod.get_pick_similar_vals(
        locs, info, diameter=_PICK_DIAMETER
    )
    assert len(x_sim) == len(y_sim) == len(rmsds) == len(nlocs)
    assert len(x_sim) >= 1


def _fork_otsu(image):
    """The pre-dedup picasso-workflow fork of Otsu's threshold.

    Verbatim copy of the ``otsu`` that lived (in triplicate) in
    ``outpost/masking/mask.py``, ``outpost/spatial_stats/ripleys.py`` and
    ``outpost/spatial_stats/ripley_dcatlas_analysis.py`` before they were
    rewired to ``picasso.masking.threshold_otsu``. Kept here as the reference
    the equivalence test pins against.
    """
    counts, bin_edges = np.histogram(image, bins=256)
    counts = counts.astype("float32", copy=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    return bin_centers[idx]


def test_otsu_threshold_matches_upstream():
    """picasso's ``masking.threshold_otsu`` reproduces the removed fork.

    The three byte-identical ``otsu`` copies in the outpost were deleted and
    their callers rewired to ``picasso.masking.threshold_otsu`` (reached via
    the _picasso_api boundary). Pin that the upstream threshold equals the
    fork's output across a range of bimodal, density-map-like grayscale images
    (what the callers actually threshold), so the dedup is behaviour-preserving.
    """
    rng = np.random.default_rng(1234)
    for _ in range(50):
        h = int(rng.integers(8, 64))
        w = int(rng.integers(8, 64))
        half = (h * w) // 2
        lo = rng.normal(2.0, 1.0, size=half)
        hi = rng.normal(12.0, 3.0, size=h * w - half)
        image = np.abs(np.concatenate([lo, hi])).reshape(h, w)
        assert picasso_masking.threshold_otsu(image) == _fork_otsu(image)
