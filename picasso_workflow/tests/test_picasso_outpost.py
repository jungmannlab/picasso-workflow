#!/usr/bin/env python
"""
Module Name: test_picasso_outpost.py
Author: Heinrich Grabmayr
Initial Date: March 15, 2024
Description: Test the module picasso_outpost.py
"""
import os
import logging
import unittest
from unittest.mock import patch
import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib import cm


from picasso_workflow import picasso_outpost


logger = logging.getLogger(__name__)


class TestPicassoOutpost(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_01_shift_from_rcc(self):
        locs_a = np.rec.array(
            [(1, 1), (3, 4)], dtype=[("x", "f4"), ("y", "f4")]
        )
        info_a = [{"Width": 10, "Height": 10}]
        locs_b = np.rec.array(
            [(2, 2), (4, 5)], dtype=[("x", "f4"), ("y", "f4")]
        )
        info_b = [{"Width": 10, "Height": 10}]

        picasso_outpost.shift_from_rcc([locs_a, locs_b], [info_a, info_b])

    def test_02_align_channels(self):
        locs_a = np.rec.array(
            [(1, 1), (3, 4)], dtype=[("x", "f4"), ("y", "f4")]
        )
        info_a = [{"Width": 10, "Height": 10}]
        locs_b = np.rec.array(
            [(2, 2), (4, 5)], dtype=[("x", "f4"), ("y", "f4")]
        )
        info_b = [{"Width": 10, "Height": 10}]
        locs_c = np.rec.array(
            [(3, 3), (5, 6)], dtype=[("x", "f4"), ("y", "f4")]
        )
        info_c = [{"Width": 10, "Height": 10}]

        (shift, cum_shift, use_fiducials, method, fp_figs) = (
            picasso_outpost.align_channels(
                [locs_a, locs_b, locs_c], [info_a, info_b, info_c]
            )
        )
        logger.debug(f"shift: {shift}")

    @patch("picasso_workflow.picasso_outpost.AICSImage")
    def test_03_convert_zeiss_file(self, mock_aicsi):
        temp_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"
        )
        filepath_czi = os.path.join(temp_folder, "zeissfile.czi")
        filepath_raw = os.path.join(temp_folder, "myrawfile.raw")
        info = {"Byte Order": "<", "Camera": "FusionBT"}
        picasso_outpost.convert_zeiss_file(filepath_czi, filepath_raw, info)

        # clean up
        filepath_info = os.path.splitext(filepath_raw)[0] + ".yaml"
        os.remove(filepath_raw)
        os.remove(filepath_info)

    def test_04a_nndistribution_from_csr(self):
        r = np.arange(50)
        p = picasso_outpost.nndistribution_from_csr(r, 2, 0.3)
        assert p.shape == r.shape

    def test_04b_nndist_loglikelihood_csr(self):
        rho = 0.2
        r = np.linspace(0, 20, num=30)
        pdists = [
            picasso_outpost.nndistribution_from_csr(r, k, rho)
            for k in range(1, 4)
        ]
        # test for one spot
        nnobs = np.array([max(pd) for pd in pdists])
        print(nnobs, rho)
        loglike = picasso_outpost.nndist_loglikelihood_csr(nnobs, rho)
        assert loglike <= 0

        # test for multiple spots
        nspots = 6
        nnobs = np.array(
            [
                np.random.choice(r, size=nspots, p=pd / np.sum(pd))
                for pd in pdists
            ]
        )
        loglike = picasso_outpost.nndist_loglikelihood_csr(nnobs, rho)
        assert loglike <= 0

    def test_04c_estimate_density_from_neighbordists(self):
        rho = 0.3
        r = np.linspace(0, 10, num=50)
        kmin = 1
        kmax = 5
        pdists = [
            picasso_outpost.nndistribution_from_csr(r, k, rho)
            for k in range(kmin, kmax + 1)
        ]
        nspots = 20000
        nnobs = np.array(
            [
                np.random.choice(r, size=nspots, p=pd / np.sum(pd))
                for pd in pdists
            ]
        )
        rho_init = rho * 4 / 3
        rhofit, fitres = picasso_outpost.estimate_density_from_neighbordists(
            nnobs, rho_init, kmin
        )
        # print(fitres)
        assert np.abs(rhofit - rho) < 0.1

        # colors = cm.get_cmap("viridis", kmax).colors
        # fig, ax = plt.subplots()
        # for k in range(kmin, kmax + 1):
        #     i = k - kmin
        #     color = colors[i]
        #     _ = ax.hist(nnobs[i], bins=r, color=color, alpha=.2,
        #                 label='drawn spots')
        #     # factor 4.9 because nndist_f_csr isnot normalized. returning in there
        #     # dist / np.sum(dist) leads to fitting problems (!?)
        #     ax.plot(r + (r[1] - r[0]) / 2, pdists[i] * nspots / 4.9, color=color,
        #             label='base distribution')
        #     fdist = picasso_outpost.nndistribution_from_csr(r, k, rhofit)
        #     ax.plot(r + (r[1] - r[0]) / 2, fdist * nspots / 4.9, color=color,
        #             linestyle=':', label='fitted distribution')
        # ax.set_title(f'input density {rho:.4f}; fitted density: {rhofit:.4f}')
        # ax.set_xlabel('r')
        # ax.legend()
        # results_folder = os.path.join(
        #     os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"
        # )
        # fig.savefig(os.path.join(results_folder, 'nnfit.png'))

        # test_rhos = np.linspace(rho / 4, rho * 2, num=20)
        # loglikes = np.zeros_like(test_rhos)
        # for i, trho in enumerate(test_rhos):
        #     loglikes[i] = picasso_outpost.minimization_loglike([trho], nnobs, kmin)
        # fig, ax = plt.subplots()
        # ax.plot(test_rhos, loglikes)
        # fig.savefig(os.path.join(results_folder, 'loglike_minimization.png'))

        # assert False

    def get_locs_with_gold(
        self, gold_x, gold_y, nframes=10, locs_per_frame=5, noise=0.5
    ):
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("x", "f4"),
            ("y", "f4"),
            ("sx", "f4"),
            ("sy", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
        ]
        width = 20
        height = 42
        locs = np.lib.recfunctions.stack_arrays(
            [
                np.rec.array(
                    [
                        tuple([f, p, x, y, sx, sy, lpx, lpy])
                        for f, p, x, y, sx, sy, lpx, lpy in zip(
                            [i] * locs_per_frame,
                            list(1000 * np.random.rand(locs_per_frame)),
                            list(
                                width
                                * np.random.rand(locs_per_frame - len(gold_x))
                            )
                            + [np.random.normal(x, noise) for x in gold_x],
                            list(
                                height
                                * np.random.rand(locs_per_frame - len(gold_y))
                            )
                            + [np.random.normal(y, noise) for y in gold_y],
                            list(np.random.rand(locs_per_frame)),
                            list(np.random.rand(locs_per_frame)),
                            list(np.random.rand(locs_per_frame)),
                            list(np.random.rand(locs_per_frame)),
                        )
                    ],
                    dtype=locs_dtype,
                )
                for i in range(nframes)
            ],
            asrecarray=True,
            usemask=False,
        )
        # print(locs)
        # print(locs.dtype)
        info = [
            {
                "Frames": nframes,
                "Width": width,
                "Height": height,
                "Data Type": "u4",
            }
        ]
        return locs, info

    def test_06a_pick_gold(self):
        np.random.seed(42)
        centers = [[12, 4], [4, 12], [14, 14]]
        locs, info = self.get_locs_with_gold(
            [center[0] for center in centers],
            [center[1] for center in centers],
            nframes=100,
            locs_per_frame=4,
        )
        gold_picks = picasso_outpost.pick_gold(locs, info)
        print(gold_picks)
        # round the picks for assertion
        gold_picks = [
            list(np.round(pair).astype(np.int64)) for pair in gold_picks
        ]
        print(gold_picks)
        for center in centers:
            assert center in gold_picks

    def test_06b_index_locs(self):
        locs, info = self.get_locs_with_gold([], [])
        pick_diameters = 2.3
        index_blocs = picasso_outpost.index_locs(locs, info, pick_diameters)

        assert index_blocs is not None

    def test_06c_picked_locs(self):
        centers = [[2, 4], [4, 2], [4, 4]]
        locs, info = self.get_locs_with_gold(
            [center[0] for center in centers],
            [center[1] for center in centers],
            noise=0.05,
        )
        gold_locs = picasso_outpost.picked_locs(
            locs, info, centers, pick_diameter=0.5
        )
        print(gold_locs)
        ngold_locs = len(gold_locs)

        assert ngold_locs == len(centers) * info[0]["Frames"]

    def test_07_rsso_alignment(self):
        """Test the new filtered_RCC alignment method"""
        # Create test data with known shift
        shift_x = 2.0
        shift_y = 1.5

        # Channel A (reference)
        locs_a = np.rec.array(
            [(1, 1), (3, 4), (5, 7), (8, 2)], dtype=[("x", "f4"), ("y", "f4")]
        )

        # Channel B (shifted version of A with some noise)
        locs_b = np.rec.array(
            [
                (1 + shift_x + 0.1, 1 + shift_y + 0.1),
                (3 + shift_x - 0.1, 4 + shift_y + 0.1),
                (5 + shift_x + 0.05, 7 + shift_y - 0.05),
                (8 + shift_x - 0.05, 2 + shift_y + 0.05),
            ],
            dtype=[("x", "f4"), ("y", "f4")],
        )

        info_a = [{"Width": 20, "Height": 20}]
        info_b = [{"Width": 20, "Height": 20}]

        # Test the align_channels function with filtered_RCC
        # original_locs_a = locs_a.copy()
        # original_locs_b = locs_b.copy()

        (shift, cum_shift, use_fiducials, method, fp_figs) = (
            picasso_outpost.align_channels(
                [locs_a, locs_b],
                [info_a, info_b],
                force_method="RSSO",
                max_shift=5.0,
            )
        )

        # Check that method was correctly used
        assert method == "RSSO"

        # Check that shifts are approximately correct
        # The function should return the shift needed to align channels
        # shift is a tuple (shifts_y, shifts_x)
        assert abs(shift[0][1] - shift_y) < 0.2  # y shift for channel B
        assert abs(shift[1][1] - shift_x) < 0.2  # x shift for channel B

        # Check that channel A (reference) has no shift
        assert abs(shift[0][0]) < 0.1  # y shift for channel A
        assert abs(shift[1][0]) < 0.1  # x shift for channel A

        logger.debug(f"Detected shifts: x={shift[1]}, y={shift[0]}")
        logger.debug(f"Expected shifts: x={-shift_x}, y={-shift_y}")

    def test_08_rsso_direct_function(self):
        """Test the align_by_rsso function directly"""
        # Create test data with known shift
        shift_x = 1.0
        shift_y = 0.5

        # Channel A (reference)
        locs_a = np.rec.array(
            [(2, 2), (4, 4), (6, 6)], dtype=[("x", "f4"), ("y", "f4")]
        )

        # Channel B (shifted version of A)
        locs_b = np.rec.array(
            [
                (2 + shift_x, 2 + shift_y),
                (4 + shift_x, 4 + shift_y),
                (6 + shift_x, 6 + shift_y),
            ],
            dtype=[("x", "f4"), ("y", "f4")],
        )

        # Test the direct function
        # original_locs_a = locs_a.copy()
        # original_locs_b = locs_b.copy()

        shifts, fp_figs = picasso_outpost.align_by_rsso(
            [locs_a, locs_b], max_shift=3.0
        )

        # Check that shifts are approximately correct
        # The function should return the shift needed to align channels
        assert abs(shifts[0][1] - shift_y) < 0.15  # y shift for channel B
        assert abs(shifts[1][1] - shift_x) < 0.15  # x shift for channel B

        # Check that channel A (reference) has no shift
        assert abs(shifts[0][0]) < 0.1  # y shift for channel A
        assert abs(shifts[1][0]) < 0.1  # x shift for channel A

    def test_09_rsso_three_channels(self):
        """Test rsso with 3 channels to verify redundant benefits."""
        # Create test data with known shifts
        shift_x_b = 2.0
        shift_y_b = 1.0
        shift_x_c = -1.5
        shift_y_c = 2.5

        # Channel A (reference)
        locs_a = np.rec.array(
            [(2, 2), (4, 4), (6, 6), (8, 8), (10, 10)],
            dtype=[("x", "f4"), ("y", "f4")],
        )

        # Channel B (shifted version of A with small noise)
        locs_b = np.rec.array(
            [
                (2 + shift_x_b + 0.05, 2 + shift_y_b - 0.03),
                (4 + shift_x_b - 0.02, 4 + shift_y_b + 0.04),
                (6 + shift_x_b + 0.01, 6 + shift_y_b - 0.01),
                (8 + shift_x_b - 0.03, 8 + shift_y_b + 0.02),
                (10 + shift_x_b + 0.04, 10 + shift_y_b - 0.05),
            ],
            dtype=[("x", "f4"), ("y", "f4")],
        )

        # Channel C (another shifted version of A with small noise)
        locs_c = np.rec.array(
            [
                (2 + shift_x_c - 0.02, 2 + shift_y_c + 0.06),
                (4 + shift_x_c + 0.03, 4 + shift_y_c - 0.02),
                (6 + shift_x_c - 0.01, 6 + shift_y_c + 0.03),
                (8 + shift_x_c + 0.05, 8 + shift_y_c - 0.04),
                (10 + shift_x_c - 0.04, 10 + shift_y_c + 0.01),
            ],
            dtype=[("x", "f4"), ("y", "f4")],
        )

        # Test with the improved algorithm
        shifts, fp_figs = picasso_outpost.align_by_rsso(
            [locs_a, locs_b, locs_c], max_shift=5.0
        )

        # Check that shifts are approximately correct
        # Channel A should have no shift (reference)
        assert abs(shifts[0][0]) < 0.1  # y shift for channel A
        assert abs(shifts[1][0]) < 0.1  # x shift for channel A

        # Channel B shifts
        assert abs(shifts[0][1] - shift_y_b) < 0.1  # y shift for channel B
        assert abs(shifts[1][1] - shift_x_b) < 0.1  # x shift for channel B

        # Channel C shifts
        assert abs(shifts[0][2] - shift_y_c) < 0.1  # y shift for channel C
        assert abs(shifts[1][2] - shift_x_c) < 0.1  # x shift for channel C

        logger.debug(
            f"3-channel shifts - expected: x=[0, {shift_x_b}, {shift_x_c}], "
            f"y=[0, {shift_y_b}, {shift_y_c}]"
        )
        logger.debug(
            f"3-channel shifts - detected: x={shifts[1]}, y={shifts[0]}"
        )

    def test_10_rsso_four_channels_redundancy(self):
        """Test rsso with 4 channels to demonstrate redundancy."""
        # Create test data with known shifts
        shifts_x_true = [0.0, 1.2, -0.8, 2.3]
        shifts_y_true = [0.0, 0.5, 1.8, -1.1]

        # Base localizations
        base_locs = [(3, 3), (6, 6), (9, 9), (12, 12), (15, 15)]

        # Set random seed for reproducible results
        np.random.seed(42)

        channel_locs = []
        for i in range(4):
            # Add known shift plus small random noise to each localization
            shifted_locs = [
                (
                    x + shifts_x_true[i] + np.random.normal(0, 0.02),
                    y + shifts_y_true[i] + np.random.normal(0, 0.02),
                )
                for x, y in base_locs
            ]
            locs = np.rec.array(shifted_locs, dtype=[("x", "f4"), ("y", "f4")])
            channel_locs.append(locs)

        # Test with the improved algorithm
        shifts, fp_figs = picasso_outpost.align_by_rsso(
            channel_locs, max_shift=5.0
        )

        # Check accuracy for all channels
        # The redundant calculation should provide better accuracy
        for i in range(4):
            y_error = abs(shifts[0][i] - shifts_y_true[i])
            x_error = abs(shifts[1][i] - shifts_x_true[i])
            logger.debug(
                f"Channel {i}: y_error={y_error:.3f}, x_error={x_error:.3f}"
            )
            # With redundant calculations, we expect better accuracy
            assert y_error < 0.8  # y shift
            assert x_error < 0.8  # x shift

        logger.debug(
            f"4-channel shifts - expected: x={shifts_x_true}, "
            f"y={shifts_y_true}"
        )
        logger.debug(
            f"4-channel shifts - detected: x={shifts[1]}, y={shifts[0]}"
        )

    def test_11_rsso_plotting(self):
        """Test the histogram plotting functionality."""
        import tempfile
        import os

        # Create test data
        shift_x = 1.5
        shift_y = 0.8

        locs_a = np.rec.array(
            [(2, 2), (4, 4), (6, 6)], dtype=[("x", "f4"), ("y", "f4")]
        )
        locs_b = np.rec.array(
            [
                (2 + shift_x, 2 + shift_y),
                (4 + shift_x, 4 + shift_y),
                (6 + shift_x, 6 + shift_y),
            ],
            dtype=[("x", "f4"), ("y", "f4")],
        )

        # Create temporary directory for plots
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with plotting enabled
            shifts, fp_figs = picasso_outpost.align_by_rsso(
                [locs_a, locs_b],
                max_shift=3.0,
                plot_histogram=True,
                plot_dir=temp_dir,
            )

            # Check that plot file was created via file system
            expected_filename = "shift_histogram_ch0_to_ch1.png"
            plot_path = os.path.join(temp_dir, expected_filename)
            assert os.path.exists(
                plot_path
            ), f"Plot file {plot_path} not created"

            # Check that figure paths were returned
            assert (
                len(fp_figs) == 1
            ), f"Expected 1 figure path, got {len(fp_figs)}"
            assert (
                fp_figs[0] == plot_path
            ), f"Expected {plot_path}, got {fp_figs[0]}"

            # Check file size is reasonable (not empty)
            file_size = os.path.getsize(plot_path)
            assert (
                file_size > 1000
            ), f"Plot file seems too small: {file_size} bytes"

            logger.debug(f"Plot saved successfully to {plot_path}")
            logger.debug(f"Plot file size: {file_size} bytes")
            logger.debug(f"Returned figure paths: {fp_figs}")
