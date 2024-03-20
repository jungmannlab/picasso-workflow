#!/usr/bin/env python
"""
Module Name: test_analyse.py
Author: Heinrich Grabmayr
Initial Date: March 14, 2024
Description: Test the module analyse.py
"""
import logging
import unittest
import os
import shutil
import numpy as np
from unittest.mock import patch

from picasso_workflow import analyse


logger = logging.getLogger(__name__)


class MockPicassoMovie:
    shape = (1000, 32, 64)
    use_dask = False
    dtype = np.uint16

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        return np.random.randint(0, 1000, size=self.shape[1:], dtype=np.uint16)


# @unittest.skip("")
class TestAnalyse(unittest.TestCase):

    locs_dtype = [
        ("frame", "u4"),
        ("x", "f4"),
        ("y", "f4"),
        ("photons", "f4"),
        ("sx", "f4"),
        ("sy", "f4"),
        ("bg", "f4"),
        ("lpx", "f4"),
        ("lpy", "f4"),
        ("ellipticity", "f4"),
        ("net_gradient", "f4"),
        ("n_id", "u4"),
    ]

    def setUp(self):
        self.results_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "TestData", "analyse"
        )
        analysis_config = {
            "camera_info": {
                "gain": 1,
                "sensitivity": 0.45,
                "baseline": 100,
                "qe": 0.82,
                "pixelsize": 130,  # nm
            },
            "gpufit_installed": False,
        }
        self.ap = analyse.AutoPicasso(self.results_folder, analysis_config)
        self.ap.movie = MockPicassoMovie()

    def tearDown(self):
        pass

    # @unittest.skip('')
    def test_01_module_decorator(self):
        class TestClass:
            results_folder = self.results_folder

            @analyse.module_decorator
            def my_method(self, i, parameters, results):
                return parameters, results

        tc = TestClass()
        pars = {}
        parameters, results = tc.my_method(0, pars)
        logger.debug(f"results: {results}")
        assert results["folder"] == os.path.join(
            self.results_folder, "00_my_method"
        )

        shutil.rmtree(os.path.join(self.results_folder, "00_my_method"))

    def test_02_AutoPicasso_create_sample_movie(self):
        self.ap.movie = np.random.randint(
            0, 1000, size=(100, 32, 48), dtype=np.uint16
        )
        results = self.ap._create_sample_movie(
            os.path.join(self.results_folder, "samplemov.mp4"),
            n_sample=10,
            min_quantile=0.05,
            max_quantile=0.95,
            fps=1,
        )
        logger.debug(f"results: {results}")

        os.remove(os.path.join(self.results_folder, "samplemov.mp4"))

    @patch("picasso_workflow.analyse.io.load_movie")
    def test_03_AutoPicasso_load_dataset(self, mock_load_movie):
        mock_load_movie.return_value = (
            MockPicassoMovie(),
            {"info": "picasso-info"},
        )

        parameters = {"filename": "a.tiff"}

        parameters, results = self.ap.load_dataset(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        # logger.debug(f'results: {results}')
        assert results["duration"] > -1

        parameters = {
            "filename": "a.tiff",
            "sample_movie": {"filename": "smplmv.mp4"},
        }
        parameters, results = self.ap.load_dataset(0, parameters)

        shutil.rmtree(os.path.join(self.results_folder, "00_load_dataset"))

    def test_04_AutoPicasso_auto_min_netgrad(self):
        fn = os.path.join(self.results_folder, "autominnet.png")
        results = self.ap._auto_min_netgrad(
            box_size=7, frame_numbers=[9], filename=fn
        )
        logger.debug(results)
        assert results["filename"] == fn

        os.remove(fn)

    def test_05_AutoPicasso_identify(self):
        parameters = {
            "box_size": 7,
            "min_gradient": 500,
            "ids_vs_frame": {"filename": "ivf.png"},
        }

        parameters, results = self.ap.identify(0, parameters)

        # logger.debug(self.ap.identifications)

        shutil.rmtree(os.path.join(self.results_folder, "00_identify"))

    @patch("picasso_workflow.analyse.gausslq.locs_from_fits")
    @patch("picasso_workflow.analyse.gausslq.fit_spot")
    @patch("picasso_workflow.analyse.localize.get_spots")
    def test_06_AutoPicasso_localize(
        self, mock_get_spots, mock_fit_spot, mock_locs_from_fits
    ):
        nspots = 5
        mock_get_spots.return_value = tuple(
            [
                np.random.randint(0, 1000, size=(7, 7), dtype=np.uint16)
                for i in range(nspots)
            ]
        )
        # fit parameters
        mock_fit_spot.return_value = [0, 0, 0, 0, 0, 0]

        mock_locs_from_fits.return_value = np.rec.array(
            [
                tuple(np.random.rand(len(self.locs_dtype)))
                for i in range(nspots)
            ],
            dtype=self.locs_dtype,
        )

        parameters = {"box_size": 7, "fit_parallel": False}

        parameters, results = self.ap.localize(0, parameters)

        shutil.rmtree(os.path.join(self.results_folder, "00_localize"))

    def test_07_AutoPicasso_plot_locs_vs_frame(self):
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("sx", "f4"),
            ("sy", "f4"),
        ]
        self.ap.locs = np.rec.array(
            [
                tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        # logger.debug(self.ap.locs)

        filepath = os.path.join(self.results_folder, "lvf.png")
        self.ap._plot_locs_vs_frame(filepath)

        os.remove(filepath)

    @patch("picasso_workflow.analyse.postprocess.undrift")
    def test_08_AutoPicasso_undrift_rcc(self, mock_undrift_rcc):
        nspots = 5
        mock_undrift_rcc.return_value = (
            np.random.rand(2, len(self.ap.movie)),
            np.rec.array(
                [
                    tuple(np.random.rand(len(self.locs_dtype)))
                    for i in range(nspots)
                ],
                dtype=self.locs_dtype,
            ),
        )
        parameters = {
            "segmentation": 5000,
        }

        self.ap.undrift_rcc(0, parameters)

        shutil.rmtree(os.path.join(self.results_folder, "00_undrift_rcc"))

    def test_09_AutoPicasso_manual(self):
        parameters = {
            "prompt": "User, please perform an action.",
            "filename": "myfile.mf",
        }

        # with self.assertRaises(analyse.ManualInputLackingError):
        #     self.ap.manual(0, parameters)
        parameters, results = self.ap.manual(0, parameters)
        assert results["success"] is False

        shutil.rmtree(os.path.join(self.results_folder, "00_manual"))
