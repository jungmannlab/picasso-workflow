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

from picasso_workflow import picasso_outpost
from picasso_workflow._picasso_api import postprocess

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
    result = picasso_outpost.picked_locs(
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
    fork = picasso_outpost.picked_locs(locs, info, _CENTERS, _PICK_DIAMETER)
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
