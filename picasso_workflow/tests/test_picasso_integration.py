#!/usr/bin/env python
"""Non-mocked picasso integration tests (the "golden pipeline" tier).

The rest of the suite mocks picasso to unit-test the workflow's own
orchestration -- fast and hermetic, but blind to *semantics*: a mock accepts
any argument, so it cannot tell nm from camera pixels or catch a changed
return shape. The picasso 1.0 migration was exactly such a semantics change
(lengths px -> nm), and the one real bug it could have shipped (a render
viewport double-conversion) was caught only because a real-picasso test
reached the render stage.

These tests close that gap for the unit-sensitive analysis modules by running
each through the *real* picasso on tiny synthetic data with ground truth, and
asserting on physical correctness (or an oracle vs. calling picasso directly).
A wrong nm<->px conversion changes the physical outcome, so these fail where a
mock would stay green. They are deliberately small (hundreds of locs, CPU-only)
so they stay CI-fast; they sit between the mocked unit tests and the Gate-5
on-cluster round-trip.

Covered here (the modules the migration touched whose config carries a length,
with the convention each uses):
  * ``dbscan``          -- AutoPicasso, radius already in nm (passed through)
  * ``picasso_dbscan``  -- picasso_set, radius in camera pixels (* pixelsize)
  * ``link_locs``       -- AutoPicasso, d_max in camera pixels (* pixelsize)
  * ``smlm_clusterer``  -- AutoPicasso, radius in nm
  * ``summarize_dataset`` (NeNA) -- precision in nm (oracle vs. postprocess.nena)
  * ``density``         -- AutoPicasso, radius in camera pixels (* pixelsize)
"""

import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from picasso_workflow import analyse
from picasso_workflow._picasso_api import postprocess

PIXELSIZE = 130.0  # nm / camera pixel


def _auto_picasso(folder):
    """An ``AutoPicasso`` whose pixel size resolves to ``PIXELSIZE`` nm."""
    analysis_config = {
        "camera_info": {
            "Gain": 1,
            "Sensitivity": 0.45,
            "Baseline": 100,
            "Qe": 0.82,
            "Pixelsize": PIXELSIZE,
        },
        "gpufit_installed": False,
    }
    return analyse.AutoPicasso(folder, analysis_config)


def _info(width_px=512, height_px=512, frames=1000):
    """A picasso info list (Width/Height in camera pixels, Pixelsize in nm)."""
    return [
        {
            "Width": width_px,
            "Height": height_px,
            "Frames": frames,
            "Pixelsize": PIXELSIZE,
        }
    ]


def _locs_frame(records, frames):
    """Build a picasso-shaped localization DataFrame from (frame, x, y) rows.

    ``x``/``y`` are in nm (picasso 1.0's unit). Constant photons/sx/sy/bg/lp so
    the table is well-formed; the geometry is what the tests assert on.
    """
    df = pd.DataFrame(records, columns=["frame", "x", "y"])
    df["frame"] = df["frame"].astype(np.uint32)
    df["x"] = df["x"].astype(np.float32)
    df["y"] = df["y"].astype(np.float32)
    n = len(df)
    df["photons"] = np.full(n, 1000.0, np.float32)
    df["sx"] = np.full(n, 130.0, np.float32)
    df["sy"] = np.full(n, 130.0, np.float32)
    df["bg"] = np.full(n, 10.0, np.float32)
    df["lpx"] = np.full(n, 8.0, np.float32)
    df["lpy"] = np.full(n, 8.0, np.float32)
    df["net_gradient"] = np.full(n, 1500.0, np.float32)
    return df


def _planted_clusters(
    n_clusters=5, pts=30, spread_nm=8.0, sep_nm=600.0, frames=1000, seed=0
):
    """Localizations forming ``n_clusters`` tight clusters ``sep_nm`` apart.

    Each cluster is ``pts`` points scattered (sigma ``spread_nm``) around a
    centre; the centres lie ``sep_nm`` apart. Because ``spread_nm`` <<
    ``sep_nm``, a clusterer that reads its radius in nm separates them, while
    one that mistook a camera-pixel radius for nm (x pixelsize too large) would
    merge them -- the handle these tests pull on.
    """
    rng = np.random.default_rng(seed)
    records = []
    for ci in range(n_clusters):
        cx = 1000.0 + ci * sep_nm
        cy = 1000.0
        for _ in range(pts):
            records.append(
                (
                    int(rng.integers(0, frames)),
                    cx + rng.normal(0.0, spread_nm),
                    cy + rng.normal(0.0, spread_nm),
                )
            )
    return _locs_frame(records, frames)


class PicassoIntegrationBase(unittest.TestCase):
    def setUp(self):
        # A per-test mkdtemp (NOT the shared repo-relative ./temp) so a failure
        # never leaves cleanup-resistant dirs that cascade into sibling tests.
        self.folder = tempfile.mkdtemp(prefix="pw_integration_")

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def _ap_with(self, locs, **info_kw):
        ap = _auto_picasso(self.folder)
        ap.info = _info(**info_kw)
        ap.locs = locs
        return ap


class TestClusteringIntegration(PicassoIntegrationBase):
    def test_dbscan_autopicasso_radius_is_nm(self):
        """AutoPicasso.dbscan: radius is nm and passed straight to picasso.

        25 nm separates the 600-nm-apart clusters (-> 5 centres); a radius that
        spans the gaps merges them (-> 1). If the radius were secretly camera
        pixels (25 px = 3250 nm > 600 nm), the first case would already merge,
        so this pins the nm interpretation.
        """
        ap = self._ap_with(_planted_clusters(n_clusters=5, seed=1))
        _, _results = ap.dbscan(
            0,
            {
                "radius": 25.0,
                "min_samples": 4,
                "min_locs": 0,
                "continue_with_centers": True,
            },
        )
        # continue_with_centers -> self.locs are the cluster centres.
        self.assertEqual(len(ap.locs), 5)

        ap_merge = self._ap_with(_planted_clusters(n_clusters=5, seed=1))
        _, _r = ap_merge.dbscan(
            0,
            {
                "radius": 3000.0,  # nm, spans the 600 nm gaps
                "min_samples": 4,
                "min_locs": 0,
                "continue_with_centers": True,
            },
        )
        self.assertEqual(len(ap_merge.locs), 1)

    def test_picasso_dbscan_radius_is_camera_pixels(self):
        """picasso_set picasso_dbscan: radius is camera pixels (x pixelsize).

        To cluster at a 25 nm physical radius the config value is 25/pixelsize
        px; the module multiplies by the pixel size to recover 25 nm -> 5
        clusters. If that conversion were dropped (0.19 taken as nm), every
        point would be isolated -> 0 clusters, so this guards the x-pixelsize.
        """
        ap = self._ap_with(_planted_clusters(n_clusters=5, seed=1))
        _, results = ap.picasso_dbscan(
            0, {"radius": 25.0 / PIXELSIZE, "density": 4}
        )
        self.assertEqual(results["nclusters"], 5)

    def test_smlm_clusterer_autopicasso_radius_is_nm(self):
        """AutoPicasso.smlm_clusterer: radius is nm -> finds the 5 clusters."""
        ap = self._ap_with(_planted_clusters(n_clusters=5, seed=1))
        _, results = ap.smlm_clusterer(
            0, {"radius": 25.0, "min_locs": 4, "basic_fa": False}
        )
        self.assertEqual(results["n_centers"], 5)


class TestLinkIntegration(PicassoIntegrationBase):
    def test_link_locs_merges_within_nm(self):
        """AutoPicasso.link_locs: d_max is camera pixels, r_max = d_max*px nm.

        Two emitters, each detected in 8 consecutive frames at a fixed spot
        (3 nm jitter). A 50 nm link distance merges each emitter's detections
        into one binding event -> 2 locs. Frames are away from the first/last
        movie frame so picasso's remove_ambiguous_lengths keeps them.
        """
        rng = np.random.default_rng(2)
        records = []
        for cx, cy in [(1000.0, 1000.0), (5000.0, 5000.0)]:
            for f in range(5, 13):  # not frame 0 or the last frame
                records.append(
                    (f, cx + rng.normal(0, 3.0), cy + rng.normal(0, 3.0))
                )
        ap = self._ap_with(_locs_frame(records, frames=1000))
        self.assertEqual(len(ap.locs), 16)
        _, _results = ap.link_locs(
            0, {"d_max": 50.0 / PIXELSIZE, "tolerance": 1}
        )
        # 50 nm (= 50/pixelsize px * pixelsize) links each emitter's run.
        self.assertEqual(len(ap.locs), 2)


class TestNenaIntegration(PicassoIntegrationBase):
    def test_nena_is_nm_and_matches_direct_picasso(self):
        """summarize_dataset NeNA: nm value equals postprocess.nena, px derived.

        picasso 1.0 nena returns the precision in nm; the module reports
        nena-nm = value and nena-px = value / pixelsize. An oracle against a
        direct nena call pins the nm handling and the px derivation.
        """
        rng = np.random.default_rng(3)
        n = 3000
        records = [
            (
                int(rng.integers(0, 1000)),
                float(rng.uniform(0, 6000)),
                float(rng.uniform(0, 6000)),
            )
            for _ in range(n)
        ]
        locs = _locs_frame(records, frames=1000)
        info = _info()

        # Oracle: the direct picasso call (nm) the module wraps.
        _res_direct, s_nm = postprocess.nena(locs.copy(), info)
        self.assertGreater(s_nm, 0)

        ap = self._ap_with(locs.copy())
        _, results = ap.summarize_dataset(0, {"methods": {"NeNa": {}}})
        self.assertIn("nena-nm", results["nena"])
        self.assertAlmostEqual(
            results["nena"]["nena-nm"], float(s_nm), places=3
        )
        self.assertAlmostEqual(
            results["nena"]["nena-px"], float(s_nm) / PIXELSIZE, places=3
        )


class TestDensityIntegration(PicassoIntegrationBase):
    def test_density_radius_is_camera_pixels(self):
        """AutoPicasso.density: radius is camera pixels (x pixelsize -> nm).

        Oracle: the per-loc density the module computes equals a direct
        compute_local_density at the physical nm radius (config px * pixelsize).
        A dropped conversion would change every density value.
        """
        locs = _planted_clusters(n_clusters=5, seed=1)
        ap = self._ap_with(locs.copy())
        radius_px = 30.0 / PIXELSIZE  # -> 30 nm physical
        before_cols = set(ap.locs.columns)
        _, _results = ap.density(0, {"radius": radius_px})

        added = set(ap.locs.columns) - before_cols
        self.assertEqual(
            len(added), 1, f"expected one density column: {added}"
        )
        density_col = added.pop()

        direct, _dinfo = postprocess.compute_local_density(
            locs.copy(), _info(), radius=radius_px * PIXELSIZE
        )
        np.testing.assert_allclose(
            ap.locs[density_col].to_numpy(),
            direct[density_col].to_numpy(),
            rtol=1e-5,
        )
        # Physical sanity: clustered points really do have neighbours.
        self.assertGreater(float(ap.locs[density_col].max()), 1.0)


if __name__ == "__main__":
    unittest.main()
