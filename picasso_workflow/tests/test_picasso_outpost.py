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

        shift, cum_shift = picasso_outpost.align_channels(
            [locs_a, locs_b, locs_c], [info_a, info_b, info_c]
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
        ngold_locs = np.sum([len(lcs) for lcs in gold_locs])

        assert ngold_locs == len(centers) * info[0]["Frames"]
