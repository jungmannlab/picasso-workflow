"""`_undrift_from_picked` delegates drift estimation to picasso's memory-lean
routine and applies the result (no dense n_picks x n_frames matrix)."""

import numpy as np
import pandas as pd
import pytest

from picasso_workflow.outpost.picks import picks


def test_undrift_from_picked_applies_picasso_drift(monkeypatch):
    # picasso's undrift_from_picked returns a per-frame drift DataFrame
    drift_df = pd.DataFrame(
        {"x": np.array([0.0, 1.0, 2.0]), "y": np.array([0.0, 10.0, 20.0])}
    )
    captured = {}

    def fake(picked_locs, info):
        captured["picked"] = picked_locs
        captured["info"] = info
        return drift_df

    monkeypatch.setattr(picks.postprocess, "undrift_from_picked", fake)

    locs = pd.DataFrame(
        {
            "frame": np.array([0, 1, 2, 2]),
            "x": np.array([5.0, 5.0, 5.0, 5.0]),
            "y": np.array([7.0, 7.0, 7.0, 7.0]),
        }
    )
    picked = [locs]
    out_locs, out_info, drift = picks._undrift_from_picked(
        locs, [{"Frames": 3}], picked
    )

    # the per-frame drift is subtracted from each loc by its frame index
    assert out_locs["x"].tolist() == [5.0, 4.0, 3.0, 3.0]
    assert out_locs["y"].tolist() == [7.0, -3.0, -13.0, -13.0]
    # delegation happened with the picks + info, and drift is returned per-coord
    assert captured["picked"] is picked
    assert list(drift.keys()) == ["x", "y"]


def test_undrift_from_picked_handles_z(monkeypatch):
    drift_df = pd.DataFrame(
        {
            "x": np.array([0.0, 0.0]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 100.0]),
        }
    )
    monkeypatch.setattr(
        picks.postprocess, "undrift_from_picked", lambda p, i: drift_df
    )
    locs = pd.DataFrame(
        {
            "frame": np.array([0, 1]),
            "x": np.array([1.0, 1.0]),
            "y": np.array([2.0, 2.0]),
            "z": np.array([3.0, 3.0]),
        }
    )
    out_locs, _, drift = picks._undrift_from_picked(
        locs, [{"Frames": 2}], [locs]
    )
    assert out_locs["z"].tolist() == [3.0, -97.0]
    assert "z" in drift


def test_undrift_from_picked_aligns_with_nondefault_index(monkeypatch):
    # Drift must be applied POSITIONALLY even when locs has a filtered,
    # non-RangeIndex -- i.e. the per-frame drift must stay a plain ndarray
    # (np.asarray), not a Series that would align by label and yield NaNs.
    drift_df = pd.DataFrame(
        {"x": np.array([0.0, 1.0, 2.0]), "y": np.array([0.0, 0.0, 0.0])}
    )
    monkeypatch.setattr(
        picks.postprocess, "undrift_from_picked", lambda p, i: drift_df
    )
    locs = pd.DataFrame(
        {
            "frame": np.array([0, 1, 2, 2]),
            "x": np.array([10.0, 10.0, 10.0, 10.0]),
            "y": np.array([0.0, 0.0, 0.0, 0.0]),
        }
    )
    # drop frame 0 -> non-default index [1, 2, 3], frames [1, 2, 2]
    locs = locs[locs["frame"] > 0].copy()
    out, _, _ = picks._undrift_from_picked(locs, [{"Frames": 3}], [locs])
    # x = 10 - drift[[1,2,2]] = 10 - [1,2,2] = [9,8,8]; no NaN from misalignment
    assert out["x"].tolist() == [9.0, 8.0, 8.0]
    assert not out["x"].isna().any()


def test_undrift_from_picked_no_z_when_absent(monkeypatch):
    # drift frame has no z and locs have no z -> z left untouched / absent
    drift_df = pd.DataFrame({"x": np.array([0.0]), "y": np.array([0.0])})
    monkeypatch.setattr(
        picks.postprocess, "undrift_from_picked", lambda p, i: drift_df
    )
    locs = pd.DataFrame(
        {"frame": np.array([0]), "x": np.array([1.0]), "y": np.array([2.0])}
    )
    _, _, drift = picks._undrift_from_picked(locs, [{"Frames": 1}], [locs])
    assert "z" not in drift


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
