"""Regression tests for ``outpost/picks/picks.py``."""

import numpy as np
import pandas as pd
import pytest

from picasso_workflow.outpost.picks import picks as picks_mod

pytest.importorskip("picasso", reason="picassosr required for the picks tests")


def _synthetic_locs(centers, n_per_center=50, pixelsize=1):
    """Tight clusters at ``centers`` plus scattered background (in nm)."""
    rng = np.random.default_rng(3)
    rows = []
    for cx, cy in centers:
        for _ in range(n_per_center):
            rows.append(
                (
                    rng.integers(0, 1000),
                    cx + rng.normal(0, 0.05),
                    cy + rng.normal(0, 0.05),
                )
            )
    for _ in range(200):
        x, y = rng.uniform(0, 128), rng.uniform(0, 128)
        if all((x - cx) ** 2 + (y - cy) ** 2 > 4.0 for cx, cy in centers):
            rows.append((rng.integers(0, 1000), x, y))
    locs = pd.DataFrame(rows, columns=["frame", "x", "y"])
    locs["frame"] = locs["frame"].astype(np.int32)
    locs["x"] = locs["x"].astype(np.float32)
    locs["y"] = locs["y"].astype(np.float32)
    info = [
        {"Frames": 1000, "Width": 128, "Height": 128, "Pixelsize": pixelsize}
    ]
    return locs, info


def test_get_pick_similar_vals_returns_candidate_centers():
    """``get_pick_similar_vals`` recovers each planted cluster centre.

    Backed by ``picasso.postprocess.pick_similar_candidates`` (the public
    picasso-1.0 replacement for the former private grid scan). Returns four
    equal-length arrays ``(x, y, rmsds, nlocs)``; every planted cluster's
    centre appears with the right loc count.
    """
    centers = [(20.0, 30.0), (60.0, 80.0), (100.0, 40.0)]
    locs, info = _synthetic_locs(centers)

    x_sim, y_sim, rmsds, nlocs = picks_mod.get_pick_similar_vals(
        locs, info, diameter=1.0
    )

    assert len(x_sim) == len(y_sim) == len(rmsds) == len(nlocs)
    assert len(x_sim) >= len(centers)
    for cx, cy in centers:
        dist = np.sqrt((x_sim - cx) ** 2 + (y_sim - cy) ** 2)
        i = int(np.argmin(dist))
        assert dist[i] < 0.5
        assert nlocs[i] == 50
