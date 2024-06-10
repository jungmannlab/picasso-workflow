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
