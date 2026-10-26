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

Covered here (the modules the migration touched, with the length convention /
return shape each exercises):
  * ``identify``        -- movie -> spots via IdentificationParameters/NetGradient
  * ``picasso_localize``-- movie -> nm coordinates (identify + fit; px->nm output)
  * ``dbscan``          -- AutoPicasso, radius already in nm (passed through)
  * ``picasso_dbscan``  -- picasso_set, radius in camera pixels (* pixelsize)
  * ``smlm_clusterer``  -- AutoPicasso, radius in nm
  * ``picasso_g5m``     -- return (centers, info, clustered); asynch -> parallel
  * ``link_locs``       -- AutoPicasso, d_max in camera pixels (* pixelsize)
  * ``summarize_dataset`` (NeNA) -- precision in nm (oracle vs. postprocess.nena)
  * ``density``         -- AutoPicasso, radius in camera pixels (* pixelsize)
  * ``picasso_pair_correlation`` -- bin_size/r_max camera pixels -> nm (oracle)
  * ``picasso_frc``     -- FRC viewport = FOV in nm; finite nm resolution
  * ``picasso_mask_locs``-- mask_locs 4-tuple; generate_image nm blur
  * ``summarize_dataset`` (median-loc-precision) -- lpx/lpy are nm
  * ``_picasso_set_save_picks`` -- pick file round-trips through io.load_picks
  * ``picasso_g5m`` (empty) -- (None, info, None) -> clean failure, not a crash
  * ``convert_picasso_to_coords`` (Ripley's K) -- coords already nm
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from picasso_workflow import analyse
from picasso_workflow._picasso_api import io, postprocess

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


def _grid_positions(n, size_px, margin=12):
    """``n`` well-separated (x, y) pixel positions on a near-square grid."""
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    xs = np.linspace(margin, size_px - margin, ncol)
    ys = np.linspace(margin, size_px - margin, nrow)
    return [(float(xs[k % ncol]), float(ys[k // ncol])) for k in range(n)]


def _synthetic_movie(
    n_frames=30,
    size_px=64,
    n_emitters=6,
    peak_adu=3000.0,
    bg=100.0,
    sigma_px=1.2,
    seed=0,
):
    """A ``(n_frames, size, size)`` uint16 stack of fixed Gaussian emitters.

    Bright, always-on spots laid out on a grid (deterministically separated, so
    ``identify`` detects ~one per emitter per frame and ``localize`` fits each
    to sub-pixel accuracy without neighbours merging). Returns
    ``(movie, emitters_px)`` with the emitter pixel positions as ``(x, y)``
    pairs, so a test can check that the fitted nm coordinates equal those
    pixels times the pixel size.
    """
    rng = np.random.default_rng(seed)
    positions = _grid_positions(n_emitters, size_px)
    ex = np.array([p[0] for p in positions], dtype=np.float32)
    ey = np.array([p[1] for p in positions], dtype=np.float32)
    yy, xx = np.mgrid[:size_px, :size_px].astype(np.float32)
    kernels = peak_adu * np.exp(
        -(
            (xx[None] - ex[:, None, None]) ** 2
            + (yy[None] - ey[:, None, None]) ** 2
        )
        / (2.0 * sigma_px**2)
    )
    spots = kernels.sum(axis=0)
    movie = np.zeros((n_frames, size_px, size_px), dtype=np.uint16)
    for f in range(n_frames):
        frame = rng.poisson(bg, (size_px, size_px)).astype(np.float32) + spots
        np.clip(frame, 0, 65535, out=frame)
        movie[f] = frame.astype(np.uint16)
    return movie, list(zip(ex.tolist(), ey.tolist()))


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

    def _memmap_movie(self, movie):
        """Back a movie array with an ``np.memmap``.

        picasso 1.0's ``localize.fit`` only accepts a real movie object
        (``np.memmap`` / ``AbstractPicassoMovie`` / ...), not a plain ndarray,
        so the synthetic stack is written to the test's temp dir and re-opened
        as a memmap -- the cheapest way to hand picasso a movie it accepts.
        """
        path = os.path.join(self.folder, "movie.raw")
        movie.tofile(path)
        return np.memmap(path, dtype=movie.dtype, mode="r", shape=movie.shape)


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


class TestG5mIntegration(PicassoIntegrationBase):
    def test_picasso_g5m_runs_and_returns_centers(self):
        """picasso_g5m: picasso 1.0 g5m returns ``(centers, info, clustered)``

        -- info and clustered swapped vs 0.11 -- and renamed ``asynch`` ->
        ``parallel``. Running it on grouped locs (one tight single-molecule
        blob per group) and getting a non-empty centre table proves both the
        return-tuple unpacking and the kwarg rename; a wrong unpack yields None
        -> raises, a wrong kwarg a TypeError. (The AutoPicasso
        ``gaussian_mixture_cluster`` wraps the same g5m call but carries a
        ``@profile_resource_usage`` decorator that runs the body twice, which
        is incompatible with this stateful module in-process, so the equivalent
        picasso-set module -- single-run -- is exercised here.)
        """
        # g5m fits a mixture within each pre-clustered group; plant the "group"
        # column directly (one tight single-molecule blob per group).
        n_clusters, pts = 5, 40
        locs = _planted_clusters(n_clusters=n_clusters, pts=pts, seed=4)
        locs["group"] = np.repeat(np.arange(n_clusters), pts).astype(np.uint32)
        ap = self._ap_with(locs)
        _, results = ap.picasso_g5m(0, {"min_locs": 10})
        self.assertGreater(results["n_molecules"], 0)
        self.assertEqual(len(ap.locs), results["n_molecules"])


class TestPairCorrelationIntegration(PicassoIntegrationBase):
    def test_pair_correlation_bins_are_camera_pixels_to_nm(self):
        """picasso_pair_correlation: bin_size / r_max are camera pixels -> nm.

        Oracle: the workflow's curve equals a direct postprocess.pair_correlation
        at the physical nm arguments (config px * pixelsize). A dropped
        conversion would change the bin positions and the curve.
        """
        locs = _planted_clusters(n_clusters=5, pts=60, seed=5)
        ap = self._ap_with(locs.copy())
        binsize_px, rmax_px = 0.2, 6.0
        _, results = ap.picasso_pair_correlation(
            0, {"binsize": binsize_px, "rmax": rmax_px}
        )
        data = np.loadtxt(results["filepath_pair_correlation"])
        bins_workflow, pc_workflow = data[:, 0], data[:, 1]

        bins_direct, pc_direct = postprocess.pair_correlation(
            locs.copy(),
            _info(),
            bin_size=binsize_px * PIXELSIZE,
            r_max=rmax_px * PIXELSIZE,
        )
        np.testing.assert_allclose(bins_workflow, bins_direct, rtol=1e-4)
        np.testing.assert_allclose(pc_workflow, pc_direct, rtol=1e-4)
        # The bins span the physical r_max in nm, not the camera-pixel value.
        self.assertGreater(bins_workflow.max(), rmax_px)


class TestFrcIntegration(PicassoIntegrationBase):
    def test_frc_resolution_is_finite_nm(self):
        """picasso_frc: the FRC viewport is the FOV in nm (Width * pixelsize).

        A smoke + sanity check that FRC runs end-to-end on real locs and
        reports a finite, positive resolution in nm. If the viewport were left
        in camera pixels, the locs would fall outside it and FRC would render
        empty / error.
        """
        rng = np.random.default_rng(6)
        n = 4000
        records = [
            (
                int(rng.integers(0, 1000)),
                float(rng.uniform(0, 4000)),
                float(rng.uniform(0, 4000)),
            )
            for _ in range(n)
        ]
        # Small FOV so the two rendered half-images stay well under the guard.
        ap = self._ap_with(
            _locs_frame(records, frames=1000), width_px=64, height_px=64
        )
        _, results = ap.picasso_frc(0, {})
        res = results["resolution_nm"]
        self.assertTrue(np.isfinite(res))
        self.assertGreater(res, 0.0)


class TestMaskingIntegration(PicassoIntegrationBase):
    def test_mask_locs_partitions_dense_region(self):
        """picasso_mask_locs: picasso 1.0 mask_locs returns a 4-tuple and

        generate_image takes a nm blur. Render a density image, Otsu-threshold
        it, split the locs: a dense planted blob lands inside the mask, sparse
        background outside. A non-trivial partition proves the 4-tuple unpack
        and the keyword generate_image call; a wrong unpack would crash.
        """
        rng = np.random.default_rng(7)
        # Dense blob near (2000, 2000) nm + sparse background across the FOV.
        dense = [
            (
                int(rng.integers(0, 1000)),
                2000.0 + rng.normal(0, 60.0),
                2000.0 + rng.normal(0, 60.0),
            )
            for _ in range(800)
        ]
        sparse = [
            (
                int(rng.integers(0, 1000)),
                float(rng.uniform(0, 8000)),
                float(rng.uniform(0, 8000)),
            )
            for _ in range(200)
        ]
        ap = self._ap_with(_locs_frame(dense + sparse, frames=1000))
        n_before = len(ap.locs)
        _, results = ap.picasso_mask_locs(
            0, {"disp_px_size": 50.0, "blur": 1.0, "method": "otsu"}
        )
        # self.locs continues with the inside-mask locs: a real subset.
        n_in = len(ap.locs)
        self.assertGreater(n_in, 0)
        self.assertLess(n_in, n_before)
        # The dense blob dominates the mask, so most kept locs sit near it.
        near = np.hypot(ap.locs["x"] - 2000.0, ap.locs["y"] - 2000.0) < 300.0
        self.assertGreater(near.mean(), 0.5)


class TestIdentifyIntegration(PicassoIntegrationBase):
    def test_identify_finds_the_planted_spots(self):
        """AutoPicasso.identify: the net-gradient threshold is wrapped in a 1.0

        IdentificationParameters(method=NetGradient(...)). Running identify on a
        movie of bright, always-on emitters detects about one spot per emitter
        per frame -- proving the IdentificationParameters path works end-to-end.
        """
        movie, emitters = _synthetic_movie(n_frames=20, n_emitters=6, seed=8)
        ap = _auto_picasso(self.folder)
        ap.info = _info(width_px=64, height_px=64, frames=len(movie))
        ap.movie = movie
        _, results = ap.identify(0, {"min_gradient": 500, "box_size": 7})
        n = results["num_identifications"]
        # ~one detection per emitter per frame (allow a generous band).
        self.assertGreaterEqual(n, len(emitters))
        self.assertGreater(n, 0.5 * len(emitters) * len(movie))
        self.assertLess(n, 1.5 * len(emitters) * len(movie))


class TestLocalizeIntegration(PicassoIntegrationBase):
    def test_localize_fits_spots_to_nm_coordinates(self):
        """picasso_localize: identify + fit on a movie yields nm coordinates.

        The emitters sit at known camera-pixel positions; picasso 1.0 returns
        the fitted coordinates in nm, so each planted pixel (ex, ey) must have
        localizations near (ex * pixelsize, ey * pixelsize). If localize
        returned camera pixels instead of nm, the fitted coordinates would be
        ~pixelsize-times too small and the match would fail -- so this pins the
        nm output of the whole identify->fit path.
        """
        movie, emitters = _synthetic_movie(n_frames=20, n_emitters=6, seed=9)
        ap = _auto_picasso(self.folder)
        ap.info = _info(width_px=64, height_px=64, frames=len(movie))
        ap.movie = self._memmap_movie(movie)
        _, _results = ap.picasso_localize(
            0, {"fit_method": "lq", "box_side_length": 7, "gradient": 500}
        )
        xs = ap.locs["x"].to_numpy()
        ys = ap.locs["y"].to_numpy()
        # Nearest fitted loc to each planted emitter, in nm.
        nearest_nm = [
            float(
                np.hypot(xs - ex_px * PIXELSIZE, ys - ey_px * PIXELSIZE).min()
            )
            for ex_px, ey_px in emitters
        ]
        # Most emitters localize to sub-pixel accuracy. If localize returned
        # camera pixels instead of nm, every nearest distance would be
        # ~pixelsize-times too large, so the median would blow up -- this is
        # the handle on the nm output of the whole identify->fit path.
        self.assertLess(float(np.median(nearest_nm)), PIXELSIZE)
        self.assertGreaterEqual(
            sum(d < PIXELSIZE for d in nearest_nm), 0.75 * len(emitters)
        )


class TestMedianLocPrecisionIntegration(PicassoIntegrationBase):
    def test_median_loc_precision_is_nm(self):
        """summarize_dataset median-loc-precision: lpx/lpy are nm in 1.0.

        ``_locs_frame`` sets lpx = lpy = 8 nm, so the median precision is 8 nm
        and the px value is 8 / pixelsize. Before the fix this branch multiplied
        the (already-nm) median by the pixel size (~130x) and mislabeled it px.
        """
        ap = self._ap_with(_planted_clusters(n_clusters=5, seed=1))
        _, results = ap.summarize_dataset(
            0, {"methods": {"median-loc-precision": {}}}
        )
        mlp = results["median-loc-precision"]
        self.assertAlmostEqual(mlp["median_lp-nm"], 8.0, places=3)
        self.assertAlmostEqual(mlp["median_lp-px"], 8.0 / PIXELSIZE, places=4)


class TestSavePicksRoundTrip(PicassoIntegrationBase):
    def test_save_picks_reloads_with_the_same_nm_size(self):
        """_picasso_set_save_picks -> io.load_picks round-trips in nm.

        picasso 1.0 io.load_picks only reads the nm size keys ("Diameter (nm)"
        etc.) and needs the format marker; the file is written via io.picks_dict.
        Before the fix Circle/Rectangle were saved under the legacy bare keys
        and could not be reloaded (only Square was migrated).
        """
        ap = _auto_picasso(self.folder)
        ap.info = _info()
        centers = [[1000.0, 2000.0], [3000.0, 4000.0]]
        for shape, size in [("Circle", 200.0), ("Square", 150.0)]:
            fp = ap._picasso_set_save_picks(
                {"folder": self.folder}, centers, shape, size, f"{shape}.yaml"
            )
            picks, loaded_shape, loaded_size = io.load_picks(fp)
            self.assertEqual(loaded_shape, shape)
            self.assertAlmostEqual(float(loaded_size), size, places=3)
            self.assertEqual(len(picks), len(centers))


class TestG5mEmptyIntegration(PicassoIntegrationBase):
    def test_picasso_g5m_empty_fails_cleanly(self):
        """picasso_g5m: an empty g5m result fails cleanly, not with len(None).

        With ``min_locs`` far above any group's size, g5m finds no molecules and
        returns (None, info, None); the module must report success=False rather
        than crash on ``len(None)``.
        """
        n_clusters, pts = 4, 20
        locs = _planted_clusters(n_clusters=n_clusters, pts=pts, seed=4)
        locs["group"] = np.repeat(np.arange(n_clusters), pts).astype(np.uint32)
        ap = self._ap_with(locs)
        _, results = ap.picasso_g5m(0, {"min_locs": 100000})
        self.assertFalse(results["success"])
        self.assertNotIn("n_molecules", results)


class TestRipleyCoordsIntegration(PicassoIntegrationBase):
    def test_convert_picasso_to_coords_is_nm(self):
        """Ripley's K convert_picasso_to_coords returns the nm coords as-is.

        picasso 1.0 locs are nm; before the fix this multiplied them by the
        pixel size, pushing points outside the (correct-nm) mask extent.
        """
        from picasso_workflow.outpost.spatial_stats.ripleys import (
            convert_picasso_to_coords,
        )

        mols = _planted_clusters(n_clusters=3, pts=10, seed=7)
        coords = convert_picasso_to_coords(mols, PIXELSIZE)
        np.testing.assert_allclose(coords[:, 0], mols["x"].to_numpy())
        np.testing.assert_allclose(coords[:, 1], mols["y"].to_numpy())


if __name__ == "__main__":
    unittest.main()
