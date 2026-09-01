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
import tempfile
import inspect
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from picasso_workflow import analyse, util, picasso_outpost

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
class TestAnalyseModules(unittest.TestCase):
    """Tests the implementation of the analysis modules defined in
    util.AbstractModuleCollection
    """

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
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"
        )
        analysis_config = {
            "camera_info": {
                "Gain": 1,
                "Sensitivity": 0.45,
                "Baseline": 100,
                "Qe": 0.82,
                "Pixelsize": 130,  # nm
            },
            "gpufit_installed": False,
        }
        self.ap = analyse.AutoPicasso(self.results_folder, analysis_config)
        self.ap.movie = MockPicassoMovie()

    def tearDown(self):
        pass

    def test_modules(self):
        """Test all modules defined in the ModuleCollection"""
        available_modules = inspect.getmembers(util.AbstractModuleCollection)
        available_modules = [
            name
            for name, _ in available_modules
            if inspect.ismethod(_) or inspect.isfunction(_)
        ]
        available_modules = [
            name for name in available_modules if name != "__init__"
        ]
        missing_modules = []
        for module in available_modules:
            # test_fun = getattr(self, module)
            # test_fun()
            try:
                test_fun = getattr(self, module)
                test_fun()
            except AttributeError as e:
                expecterr = (
                    f"'TestAnalyseModules' object has no attribute '{module}'"
                )
                if expecterr in str(e):
                    missing_modules.append(module)
                else:
                    raise e

        if missing_modules:
            all_methods = inspect.getmembers(self)
            all_methods = [
                name
                for name, _ in all_methods
                if inspect.ismethod(_) or inspect.isfunction(_)
            ]
            all_methods = [name for name in all_methods if name != "__init__"]
            errtext = (
                f"Unit tests of modules {missing_modules} not implemented!"
            )
            # errtext += f"All attributes: {all_methods}"
            raise NotImplementedError(errtext)

    @patch("picasso_workflow.analyse.io.load_movie")
    def load_dataset_movie(self, mock_load_movie):
        mock_load_movie.return_value = (
            MockPicassoMovie(),
            {"info": "picasso-info"},
        )

        parameters = {"filename": "a.tiff"}

        parameters, results = self.ap.load_dataset_movie(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        # logger.debug(f'results: {results}')
        assert results["duration"] > -1

        parameters = {
            "filename": "a.tiff",
            "sample_movie": {"filename": "smplmv.mp4"},
        }
        parameters, results = self.ap.load_dataset_movie(0, parameters)

        shutil.rmtree(
            os.path.join(self.results_folder, "00_load_dataset_movie")
        )

    def identify(self):
        parameters = {
            "box_size": 7,
            "min_gradient": 500,
            "ids_vs_frame": {"filename": "ivf.png"},
        }

        parameters, results = self.ap.identify(0, parameters)

        # logger.debug(self.ap.identifications)

        shutil.rmtree(os.path.join(self.results_folder, "00_identify"))

        # picasso 0.11 identify filters are forwarded to localize.identify
        with patch(
            "picasso_workflow.analyse.localize.identify"
        ) as mock_identify:
            mock_identify.return_value = (self.ap.identifications, {})
            parameters = {
                "box_size": 7,
                "min_gradient": 500,
                "temporal_median_window": 20,
                "gaussian_filter_sigma": 1.5,
                "identify_parallel": False,
            }
            self.ap.identify(0, parameters)
            _, kwargs = mock_identify.call_args
            assert kwargs["temporal_median_window"] == 20
            assert kwargs["gaussian_filter_sigma"] == 1.5
            assert kwargs["threaded"] is False
            # unset filters are not forwarded (picasso defaults preserved)
            assert "temporal_median_stride" not in kwargs
        shutil.rmtree(os.path.join(self.results_folder, "00_identify"))

    @patch("picasso_workflow.analyse.localize.fit")
    def localize(self, mock_fit):
        nspots = 5
        # picasso 0.11 localize.fit returns (locs DataFrame, fit info list).
        locs = pd.DataFrame(
            np.rec.array(
                [
                    tuple(np.random.rand(len(self.locs_dtype)))
                    for i in range(nspots)
                ],
                dtype=self.locs_dtype,
            )
        )
        mock_fit.return_value = (locs, [])
        self.ap.info = []

        # default: gausslq (no GPU configured), fit_parallel honored
        parameters = {"box_size": 7, "fit_parallel": False}
        parameters, results = self.ap.localize(0, parameters)
        _, kwargs = mock_fit.call_args
        assert kwargs["fitting_method"] == "gausslq"
        assert kwargs["multiprocess"] is False
        assert results["fit_method"] == "gausslq"
        shutil.rmtree(os.path.join(self.results_folder, "00_localize"))

        # explicit fitting_method + spline calibration + fitter controls
        mock_fit.reset_mock()
        spline_cal = {"coeff": [1, 2, 3]}
        parameters = {
            "box_size": 7,
            "fit_parallel": True,
            "fitting_method": "spline",
            "spline_calibration": spline_cal,
            "eps": 0.001,
            "max_it": 100,
        }
        parameters, results = self.ap.localize(0, parameters)
        _, kwargs = mock_fit.call_args
        assert kwargs["fitting_method"] == "spline"
        # a dict calibration is passed through untouched
        assert kwargs["spline_calibration"] == spline_cal
        assert kwargs["multiprocess"] is True
        assert kwargs["eps"] == 0.001
        assert kwargs["max_it"] == 100
        assert results["fit_method"] == "spline"
        shutil.rmtree(os.path.join(self.results_folder, "00_localize"))

        # sCMOS camera_calibration given as a path is resolved via the loader
        mock_fit.reset_mock()
        cam_cal = {"gain": 1.0}
        with patch(
            "picasso_workflow.analyse.io.load_camera_calibration",
            return_value=cam_cal,
        ) as mock_loader:
            parameters = {
                "box_size": 7,
                "fit_parallel": False,
                "camera_calibration": "cam.yaml",
            }
            self.ap.localize(0, parameters)
        mock_loader.assert_called_once_with("cam.yaml")
        _, kwargs = mock_fit.call_args
        assert kwargs["camera_calibration"] == cam_cal
        shutil.rmtree(os.path.join(self.results_folder, "00_localize"))

        # GPU is orthogonal to the model: with a GPU fitter configured, an
        # explicit base method is routed to its -gpu variant, and the default
        # resolves to gausslq-gpu. picasso is mocked here, so the CUDA
        # backend is presented as available to get past the fail-fast guard.
        self.ap.analysis_config["gpufit_installed"] = True
        gpu_patch = patch(
            "picasso_workflow.analyse._gpu_fitting_available",
            return_value=True,
        )
        try:
            gpu_patch.start()
            mock_fit.reset_mock()
            self.ap.localize(0, {"box_size": 7, "fitting_method": "gaussmle"})
            _, kwargs = mock_fit.call_args
            assert kwargs["fitting_method"] == "gaussmle-gpu"
            shutil.rmtree(os.path.join(self.results_folder, "00_localize"))

            mock_fit.reset_mock()
            self.ap.localize(0, {"box_size": 7})
            _, kwargs = mock_fit.call_args
            assert kwargs["fitting_method"] == "gausslq-gpu"
            shutil.rmtree(os.path.join(self.results_folder, "00_localize"))
        finally:
            gpu_patch.stop()
            self.ap.analysis_config["gpufit_installed"] = False

    def load_picassoconfig(self):
        pass

    @patch("picasso_workflow.analyse.localize.fit")
    def test_localize_gpu_guard(self, mock_fit):
        """A -gpu fitting method fails fast with an actionable error when the
        CUDA backend is unavailable, does not block CPU methods, and passes
        through when the GPU is available."""
        nspots = 3
        locs = pd.DataFrame(
            np.rec.array(
                [
                    tuple(np.random.rand(len(self.locs_dtype)))
                    for _ in range(nspots)
                ],
                dtype=self.locs_dtype,
            )
        )
        mock_fit.return_value = (locs, [])
        self.ap.info = []
        # pixelsize is a read-only property; it falls back to
        # camera_info["Pixelsize"] (130, from setUp).
        self.ap.identifications = pd.DataFrame({"frame": [0, 1, 2]})
        spline_cal = {"coeff": [1, 2, 3]}

        def _localize(method):
            return self.ap.localize(
                0,
                {
                    "box_size": 7,
                    "fit_parallel": False,
                    "fitting_method": method,
                    "spline_calibration": spline_cal,
                },
            )

        def _cleanup():
            shutil.rmtree(
                os.path.join(self.results_folder, "00_localize"),
                ignore_errors=True,
            )

        # GPU requested but unavailable -> fail fast, before calling picasso.
        with patch(
            "picasso_workflow.analyse._gpu_fitting_available",
            return_value=False,
        ):
            with self.assertRaises(analyse.AutoPicassoError) as ctx:
                _localize("spline-mle-gpu")
        assert "numba.cuda.is_available()" in str(ctx.exception)
        mock_fit.assert_not_called()
        _cleanup()

        # A CPU spline method is never blocked by the guard.
        mock_fit.reset_mock()
        with patch(
            "picasso_workflow.analyse._gpu_fitting_available",
            return_value=False,
        ):
            _localize("spline-mle")
        _, kwargs = mock_fit.call_args
        assert kwargs["fitting_method"] == "spline-mle"
        _cleanup()

        # When the GPU backend is available, the -gpu method passes through.
        mock_fit.reset_mock()
        with patch(
            "picasso_workflow.analyse._gpu_fitting_available",
            return_value=True,
        ):
            _localize("spline-mle-gpu")
        _, kwargs = mock_fit.call_args
        assert kwargs["fitting_method"] == "spline-mle-gpu"
        _cleanup()

    def zfit(self):
        pass

    def test_infer_zfit_fitting_method(self):
        # gaussmle* variants -> gaussmle; everything else -> gausslq
        self.ap.info = [{"Box Size": 7}, {"Fit method": "gaussmle-gpu"}]
        assert self.ap._infer_zfit_fitting_method() == "gaussmle"
        self.ap.info = [{"Fit method": "gausslq-rotated"}]
        assert self.ap._infer_zfit_fitting_method() == "gausslq"
        # spline localizations already carry z; fall back to gausslq label
        self.ap.info = [{"Fit method": "spline"}]
        assert self.ap._infer_zfit_fitting_method() == "gausslq"
        # no recorded method -> default gausslq
        self.ap.info = [{"Box Size": 7}]
        assert self.ap._infer_zfit_fitting_method() == "gausslq"
        self.ap.info = []
        assert self.ap._infer_zfit_fitting_method() == "gausslq"

    @patch("picasso_workflow.analyse.io.save_any_calibration")
    @patch("picasso_workflow.analyse.io.load_movie")
    def register_channels(self, mock_load_movie, mock_save_cal):
        from picasso.registration import tform

        n = 4

        def fresh_channels():
            self.ap.channel_locs = [
                pd.DataFrame(
                    {"x": np.arange(n) * 1.0, "y": np.arange(n) * 2.0}
                )
                for _ in range(2)
            ]
            self.ap.channel_info = [[], []]
            self.ap.channel_tags = ["c0", "c1"]

        fresh_channels()
        mock_load_movie.return_value = (MockPicassoMovie(), {})

        # reference channel 0 -> identity (skipped); channel 1 -> a real
        # translation so we can check it is actually warped by the inverse.
        ident = tform.identity("affine").to_dict()
        shift = {
            "model": "affine",
            "matrix": [[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]],
            "domain": None,
        }
        calibration = {
            "registration_model": "affine",
            "channel_transforms": [ident, shift],
            "rms": [1.0],
        }
        ref_before = self.ap.channel_locs[0]["x"].to_numpy().copy()
        ch1_xy = self.ap.channel_locs[1][["x", "y"]].to_numpy()
        expected_ch1 = tform.from_dict(shift).inverse().apply(ch1_xy)
        with patch(
            "picasso.registration."
            "calibrate_channel_registration_from_beads",
            return_value=calibration,
        ) as mock_calibrate:
            parameters = {
                "bead_movies": ["b0.tif", "b1.tif"],
                "box_size": 7,
                "min_gradient": 500,
            }
            parameters, results = self.ap.register_channels(0, parameters)

        mock_calibrate.assert_called_once()
        mock_save_cal.assert_called_once()
        assert results["registration_model"] == "affine"
        # reference channel untouched, and no registration record appended
        np.testing.assert_allclose(
            self.ap.channel_locs[0]["x"].to_numpy(), ref_before
        )
        assert self.ap.channel_info[0] == []
        # non-reference channel warped by the inverse transform, and recorded
        np.testing.assert_allclose(
            self.ap.channel_locs[1][["x", "y"]].to_numpy(), expected_ch1
        )
        assert len(self.ap.channel_info[1]) == 1
        shutil.rmtree(
            os.path.join(self.results_folder, "00_register_channels")
        )

        # bead-movie / channel count mismatch raises a clear error
        fresh_channels()
        with self.assertRaises(analyse.AutoPicassoError):
            self.ap.register_channels(
                0,
                {
                    "bead_movies": ["only_one.tif"],
                    "box_size": 7,
                    "min_gradient": 500,
                },
            )
        shutil.rmtree(
            os.path.join(self.results_folder, "00_register_channels")
        )

    @patch("picasso_workflow.analyse.io.load_movie")
    def export_brightfield(self, mock_load):
        frame = np.random.randint(0, 1000, size=(1, 32, 32))
        mock_load.return_value = (frame, [])

        parameters = {"filepath": "myfp.ome.tiff"}

        parameters, results = self.ap.export_brightfield(0, parameters)

        shutil.rmtree(
            os.path.join(self.results_folder, "00_export_brightfield")
        )

    @patch("picasso_workflow.analyse.postprocess.undrift")
    def undrift_rcc(self, mock_undrift_rcc):
        nspots = 5
        mock_undrift_rcc.return_value = (
            np.random.rand(2, len(self.ap.movie)),
            pd.DataFrame(
                np.rec.array(
                    [
                        tuple(np.random.rand(len(self.locs_dtype)))
                        for i in range(nspots)
                    ],
                    dtype=self.locs_dtype,
                )
            ),
        )
        parameters = {
            "segmentation": 5000,
        }

        self.ap.undrift_rcc(0, parameters)

        shutil.rmtree(os.path.join(self.results_folder, "00_undrift_rcc"))

    @patch("picasso_workflow.analyse.aim.aim")
    def undrift_aim(self, mock_undrift_aim):
        nspots = 5
        mock_undrift_aim.return_value = (
            np.random.rand(2, len(self.ap.movie)),
            [{"name": "info"}, {"Pixelsize": 130}],
            pd.DataFrame(
                np.rec.array(
                    [
                        tuple(np.random.rand(len(self.locs_dtype)))
                        for i in range(nspots)
                    ],
                    dtype=self.locs_dtype,
                )
            ),
        )
        parameters = {
            "segmentation": 50,
            "intersect_d": 20,
            "roi_r": 60,
            "dimensions": ["x", "y"],
        }

        self.ap.undrift_aim(0, parameters)

        shutil.rmtree(os.path.join(self.results_folder, "00_undrift_aim"))

    def manual(self):
        parameters = {
            "prompt": "User, please perform an action.",
            "filename": "myfile.mf",
        }

        # with self.assertRaises(analyse.ManualInputLackingError):
        #     self.ap.manual(0, parameters)
        parameters, results = self.ap.manual(0, parameters)
        assert results["success"] is False

        # clean up
        shutil.rmtree(os.path.join(self.results_folder, "00_manual"))

    @patch("picasso_workflow.analyse.postprocess.nena")
    def summarize_dataset(self, mock_nena):
        mock_nena.return_value = (1.8, [2.4, 4.1])
        parameters = {"methods": {"NeNa": {}}}
        parameters, results = self.ap.summarize_dataset(0, parameters)

        assert "nena" in results.keys()

        with self.assertRaises(NotImplementedError):
            self.ap.summarize_dataset(0, {"methods": {"NoMethod": {}}})

        # clean up
        shutil.rmtree(
            os.path.join(self.results_folder, "00_summarize_dataset")
        )

    @patch("picasso_workflow.analyse.AutoPicasso._save_locs")
    def save_single_dataset(self, mock_save):
        mock_save.return_value = {"res_a": 7}
        parameters = {"filename": "locs.hdf5"}
        parameters, results = self.ap.save_single_dataset(0, parameters)

        assert results["res_a"] == 7

        # clean up
        shutil.rmtree(
            os.path.join(self.results_folder, "00_save_single_dataset")
        )

    def save_datasets_aggregated(self):
        parameters = {"filename": "locs.hdf5"}
        self.ap.channel_locs = []
        self.ap.channel_info = []
        self.ap.channel_tags = []
        parameters, results = self.ap.save_datasets_aggregated(0, parameters)

        # clean up
        shutil.rmtree(
            os.path.join(self.results_folder, "00_save_datasets_aggregated")
        )

    @patch("picasso_workflow.analyse.io.load_locs")
    def load_dataset_localizations(self, mock_load_locs):
        mock_load_locs.return_value = ([1, 2, 3], None)
        parameters = {"filename": "locs.hdf5"}
        parameters, results = self.ap.load_dataset_localizations(0, parameters)

        assert "picasso version" in results.keys()

        # clean up
        shutil.rmtree(
            os.path.join(self.results_folder, "00_load_dataset_localizations")
        )

    @patch("picasso_workflow.analyse.io.load_locs")
    def load_datasets_to_aggregate(self, mock_load):
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("x", "f4"),
            ("y", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
        ]
        locs = np.rec.array(
            [
                tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        locs = pd.DataFrame(locs)
        mock_load.return_value = (locs, {"info": 4})
        parameters = {
            "filepaths": ["/my/path/to/locs.hdf5", "/my/path/to2/locs.hdf5"],
            "tags": ["1", "2"],
        }
        parameters, results = self.ap.load_datasets_to_aggregate(0, parameters)

        assert "filepaths" in results.keys()

        # clean up
        shutil.rmtree(
            os.path.join(self.results_folder, "00_load_datasets_to_aggregate")
        )

    @patch("picasso_workflow.outpost_modules.render.plot_scene")
    @patch("picasso_workflow.analyse.picasso_outpost.align_channels")
    def align_channels(self, mock_align_channels, mock_plot_scene):
        mock_align_channels.return_value = (
            [[3], [2]],
            np.zeros((3, 4, 5)),
            False,
            "RCC",
            [],
            {},
        )
        self.ap.channel_info = []

        parameters = {"fig_filename": "shiftplot.png"}
        parameters, results = self.ap.align_channels(0, parameters)

        assert os.path.exists(results["fig_filepath"])

        # clean up
        shutil.rmtree(os.path.join(self.results_folder, "00_align_channels"))

    def combine_channels(self):
        # create locs to be combined
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("x", "f4"),
            ("y", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
        ]
        locs1 = np.rec.array(
            [
                tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        locs1 = pd.DataFrame(locs1)
        locs2 = np.rec.array(
            [
                tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        locs2 = pd.DataFrame(locs2)
        self.ap.channel_locs = [locs1, locs2]
        self.ap.channel_info = [["info1"], ["info2"]]
        self.ap.channel_tags = ["1", "2"]
        self.ap.combine_channels(0, {})

        shutil.rmtree(os.path.join(self.results_folder, "00_combine_channels"))

    @patch(
        "picasso_workflow.analyse.picasso_outpost.convert_zeiss_file",
        MagicMock,
    )
    def convert_zeiss_movie(self):

        parameters = {"filepath": "a.tiff"}

        parameters, results = self.ap.convert_zeiss_movie(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        logger.debug(f"results: {results}")
        assert results["filename_raw"] == "a.raw"
        assert results["duration"] > -1

        shutil.rmtree(
            os.path.join(self.results_folder, "00_convert_zeiss_movie")
        )

    # @patch("picasso.postprocess.cluster_combine", MagicMock)
    # def aggregate_cluster(self):
    #     parameters = {}
    #     parameters, results = self.ap.aggregate_cluster(0, parameters)
    #     # logger.debug(f'parameters: {parameters}')
    #     logger.debug(f"results: {results}")
    #     assert results["duration"] > -1

    @patch(
        "picasso_workflow.analyse.postprocess.compute_local_density",
        MagicMock(),
    )
    def density(self):
        self.ap.info = []
        parameters = {"radius": 5}
        parameters, results = self.ap.density(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        logger.debug(f"results: {results}")
        assert results["duration"] > -1

        shutil.rmtree(os.path.join(self.results_folder, "00_density"))

    @patch("picasso_workflow.analyse.clusterer.find_cluster_centers")
    @patch("picasso_workflow.analyse.clusterer.dbscan")
    def dbscan(self, mock_dbscan, mock_fcc):
        self.ap.info = [{"Width": 1000, "Height": 1000, "Frames": 10000}]
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("x", "f4"),
            ("y", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
            ("group", "u4"),
        ]
        self.ap.locs = np.rec.array(
            [
                tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        self.ap.locs = pd.DataFrame(self.ap.locs)
        mock_dbscan.return_value = (self.ap.locs, {})
        mock_fcc.return_value = self.ap.locs

        parameters = {
            "radius": 5,
            "min_density": 0.3,
            "min_samples": 3,
            "min_locs": 8,
            "continue_with_centers": True,
        }
        parameters, results = self.ap.dbscan(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        logger.debug(f"results: {results}")
        assert results["duration"] > -1

        shutil.rmtree(os.path.join(self.results_folder, "00_dbscan"))

    @patch("picasso_workflow.analyse.clusterer.find_cluster_centers")
    @patch("picasso_workflow.analyse.clusterer.hdbscan")
    def hdbscan(self, mock_hdbscan, mock_fcc):
        self.ap.info = [{"Width": 1000, "Height": 1000, "Frames": 10000}]
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("x", "f4"),
            ("y", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
        ]
        self.ap.locs = np.rec.array(
            [
                tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        self.ap.locs = pd.DataFrame(self.ap.locs)
        mock_hdbscan.return_value = (self.ap.locs, {})
        mock_fcc.return_value = self.ap.locs

        parameters = {"min_cluster": 5, "min_samples": 3}
        parameters, results = self.ap.hdbscan(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        logger.debug(f"results: {results}")
        assert results["duration"] > -1

        shutil.rmtree(os.path.join(self.results_folder, "00_hdbscan"))

    def binding_event_analysis(self):
        # parameters, results = self.ap.binding_event_analysis(0, parameters)

        # shutil.rmtree(os.path.join(
        #     self.results_folder, "00_binding_event_analysis"))
        pass

    @patch("picasso_workflow.analyse.clusterer.find_cluster_centers")
    @patch("picasso_workflow.analyse.clusterer.cluster")
    def smlm_clusterer(self, mock_clusterer, mock_fcc):
        self.ap.info = [{"Width": 1000, "Height": 1000, "Frames": 10000}]
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("x", "f4"),
            ("y", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
        ]
        self.ap.locs = np.rec.array(
            [
                tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        self.ap.locs = pd.DataFrame(self.ap.locs)
        # clusterer.cluster adds a "group" column and drops unclustered locs;
        # emulate a few clusters so the locs-per-cluster stats/histogram work.
        clustered = self.ap.locs.copy()
        clustered["group"] = np.arange(len(clustered)) % 5
        mock_clusterer.return_value = (clustered, {})
        mock_fcc.return_value = clustered.iloc[:5].copy()

        parameters = {"radius": 5, "min_locs": 10}
        parameters, results = self.ap.smlm_clusterer(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        logger.debug(f"results: {results}")
        assert results["duration"] > -1
        # dropped-locs accounting and cluster-size histogram are populated
        assert results["n_locs_in"] == len(self.ap.movie)
        assert results["n_centers"] == 5
        assert os.path.exists(results["fp_fig_clustersizes"])

        shutil.rmtree(os.path.join(self.results_folder, "00_smlm_clusterer"))

    @patch("picasso_workflow.analyse.lib.plot_subclustering_check")
    @patch("picasso_workflow.analyse.clusterer.test_subclustering")
    @patch("picasso_workflow.analyse.g5m.g5m")
    def gaussian_mixture_cluster(
        self, mock_gmms, mock_test_subclustering, mock_plot
    ):
        self.ap.info = [{"Width": 1000, "Height": 1000, "Frames": 10000}]
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("x", "f4"),
            ("y", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
            ("n", "u4"),
            ("n_events", "u4"),
        ]
        self.ap.locs = np.rec.array(
            [
                tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        self.ap.locs = pd.DataFrame(self.ap.locs)
        mock_gmms.return_value = self.ap.locs, self.ap.locs, self.ap.info
        mock_test_subclustering.return_value = (None, None)
        mock_plot.return_value = None

        parameters = {"min_locs": 10, "min_sigma": 0.2, "max_sigma": 0.9}
        parameters, results = self.ap.gaussian_mixture_cluster(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        logger.debug(f"results: {results}")
        assert results["duration"] > -1

        shutil.rmtree(
            os.path.join(self.results_folder, "00_gaussian_mixture_cluster")
        )

    @patch("picasso_workflow.analyse.g5m.g5m")
    def test_gaussian_mixture_cluster_no_clusters_raises(self, mock_gmm):
        """No clusters found must raise a clear AutoPicassoError.

        picasso.g5m returns ``(None, None, info)`` when no molecules are
        found, and an empty centers table when postprocess filtering removes
        them all. Both previously crashed with a cryptic
        ``'NoneType' object is not subscriptable`` (or empty-quantile /
        divide-by-zero) at the histogram/statistics step.
        """
        self.ap.info = [{"Width": 1000, "Height": 1000, "Frames": 10000}]
        self.ap.locs = pd.DataFrame({"x": [1.0], "y": [2.0], "group": [0]})
        parameters = {"min_locs": 10}
        folder = os.path.join(
            self.results_folder, "00_gaussian_mixture_cluster"
        )

        no_result_returns = {
            "none": (None, None, self.ap.info),
            "empty": (
                pd.DataFrame({"n_events": []}),
                pd.DataFrame({"group": []}),
                self.ap.info,
            ),
        }
        for label, ret in no_result_returns.items():
            with self.subTest(case=label):
                mock_gmm.return_value = ret
                try:
                    with self.assertRaises(analyse.AutoPicassoError):
                        self.ap.gaussian_mixture_cluster(0, parameters)
                finally:
                    shutil.rmtree(folder, ignore_errors=True)

    @patch("picasso_workflow.analyse.g5m.g5m")
    def test_module_error_saves_current_locs(self, mock_gmm):
        """A module error dumps the current locs for post-mortem debugging.

        Exercises the full path: the module raises (g5m finds nothing), so
        module_decorator saves whatever self.locs held into an
        ``error_state`` subfolder of the module's result folder.
        """
        self.ap.info = [{"Width": 1000, "Height": 1000, "Frames": 10000}]
        self.ap.locs = pd.DataFrame(
            {
                "frame": np.arange(20, dtype="u4"),
                "x": np.random.rand(20).astype("f4"),
                "y": np.random.rand(20).astype("f4"),
                "lpx": np.full(20, 0.1, dtype="f4"),
                "lpy": np.full(20, 0.1, dtype="f4"),
            }
        )
        self.ap.channel_locs = None
        mock_gmm.return_value = (None, None, self.ap.info)

        folder = os.path.join(
            self.results_folder, "00_gaussian_mixture_cluster"
        )
        try:
            with self.assertRaises(analyse.AutoPicassoError):
                self.ap.gaussian_mixture_cluster(0, {"min_locs": 10})
            dumped = os.path.join(folder, "error_state", "locs.hdf5")
            self.assertTrue(
                os.path.exists(dumped),
                "current locs should be saved on a module error",
            )
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def test_save_state_on_error_dumps_channel_locs(self):
        """_save_state_on_error dumps aggregation channel_locs too."""
        ch_locs = pd.DataFrame(
            {
                "x": np.random.rand(5).astype("f4"),
                "y": np.random.rand(5).astype("f4"),
            }
        )
        self.ap.locs = None
        self.ap.channel_locs = [ch_locs, ch_locs.copy()]
        self.ap.channel_info = [
            [{"Width": 10, "Height": 10, "Frames": 100}],
            [{"Width": 10, "Height": 10, "Frames": 100}],
        ]
        self.ap.channel_tags = ["chan_a", "chan_b"]

        folder = os.path.join(self.results_folder, "err_dump")
        os.makedirs(folder, exist_ok=True)
        try:
            self.ap._save_state_on_error(folder)
            error_dir = os.path.join(folder, "error_state")
            self.assertTrue(
                os.path.exists(os.path.join(error_dir, "chan_a.hdf5"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(error_dir, "chan_b.hdf5"))
            )
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    @patch("picasso_workflow.analyse.AutoPicasso._save_locs")
    def test_save_state_on_error_never_raises(self, mock_save):
        """A failure while dumping must not mask the original error."""
        # force the save to blow up; the helper must swallow it (log a
        # warning) and return normally so the caller's real error survives.
        mock_save.side_effect = RuntimeError("disk full")
        self.ap.locs = pd.DataFrame({"x": [1.0], "y": [2.0]})
        self.ap.channel_locs = None
        folder = os.path.join(self.results_folder, "err_noraise")
        os.makedirs(folder, exist_ok=True)
        try:
            self.assertIsNone(self.ap._save_state_on_error(folder))
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    @patch("picasso_workflow.analyse.distance.cdist")
    def nneighbor(self, mock_cdist):
        mock_cdist.return_value = np.random.rand(len(self.ap.movie), 4)
        # def nneighbor(self):
        self.ap.info = []
        parameters = {
            "dims": ["x", "y"],
            "nth_NN": 2,
            "nth_rdf": 3,
        }
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("sx", "f4"),
            ("sy", "f4"),
            ("x", "f4"),
            ("y", "f4"),
        ]
        self.ap.locs = np.rec.array(
            [
                tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        self.ap.locs = pd.DataFrame(self.ap.locs)
        self.ap.channel_locs = [self.ap.locs]
        self.ap.tags = ["mytag"]
        parameters, results = self.ap.nneighbor(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        logger.debug(f"results: {results}")
        assert results["duration"] > -1

        # assert False

        shutil.rmtree(os.path.join(self.results_folder, "00_nneighbor"))

    def fit_csr(self):
        self.ap.info = []
        neighbors = np.array([[2, 5, 7], [3, 5, 8], [2, 4, 6], [2, 4, 7]])
        parameters = {
            "nneighbors": neighbors,
            "dimensionality": 2,
        }

        parameters, results = self.ap.fit_csr(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        logger.debug(f"results: {results}")
        assert results["duration"] > -1

        shutil.rmtree(os.path.join(self.results_folder, "00_fit_csr"))

    @patch("picasso_workflow.analyse.picasso_outpost.nndistribution_from_csr")
    def test_fit_csr_plot_max_dist_controls_display_range(self, mock_nnd):
        """plot_max_dist sets the plotting range, independent of the fit.

        The plotting loop evaluates the CSR curve on ``rvals`` spanning
        0..bin_max; capturing that range shows whether the display extent
        follows the data (default) or the explicit plot_max_dist.
        """
        captured = []

        def _fake(r, *a, **k):
            r = np.asarray(r, dtype=float)
            captured.append(float(np.max(r)) if r.size else 0.0)
            return np.zeros_like(r)

        mock_nnd.side_effect = _fake
        self.ap.info = []
        neighbors = np.array([[2, 5, 7], [3, 5, 8], [2, 4, 6], [2, 4, 7]])
        folder = os.path.join(self.results_folder, "00_fit_csr")

        # default: display range driven by the (small) data distances
        captured.clear()
        self.ap.fit_csr(0, {"nneighbors": neighbors, "dimensionality": 2})
        default_max = max(captured)
        shutil.rmtree(folder)

        # explicit plot_max_dist extends the display far beyond the data
        captured.clear()
        self.ap.fit_csr(
            0,
            {
                "nneighbors": neighbors,
                "dimensionality": 2,
                "plot_max_dist": 500,
            },
        )
        plot_max = max(captured)
        shutil.rmtree(folder)

        self.assertLess(default_max, 100)
        self.assertAlmostEqual(plot_max, 500.0, places=6)

    @patch("picasso_workflow.analyse.picasso_outpost.single_spinna_run")
    def spinna(self, mock_sptmp):
        mock_sptmp.return_value = (0, [])
        info = [{"Width": 1000, "Height": 1000, "Frames": 10000}]
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("x", "f4"),
            ("y", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
        ]
        locs = np.rec.array(
            [
                tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        locs = pd.DataFrame(locs)
        self.ap.channel_locs = [locs]
        self.ap.channel_info = [info]
        self.ap.channel_tags = ["CD86"]

        parameters = {
            "labeling_efficiency": {"CD86": 0.54},
            "labeling_uncertainty": {"CD86": 5},
            "n_simulate": 50000,
            "fp_mask_dict": None,
            "density": [0.00009],
            "random_rot_mode": "2D",
            "n_nearest_neighbors": 4,
            "sim_repeats": 5,
            "fit_NND_bin": 5,
            "fit_NND_maxdist": 300,
            # "res_factor": 10,
            "granularity": 30,
            "structures": [
                {
                    "Molecular targets": ["CD86"],
                    "Structure title": "monomer",
                    "CD86_x": [0],
                    "CD86_y": [0],
                    "CD86_z": [0],
                },
                {
                    "Molecular targets": ["CD86"],
                    "Structure title": "dimer",
                    "CD86_x": [-10, 10],
                    "CD86_y": [0, 0],
                    "CD86_z": [0, 0],
                },
            ],
        }
        parameters, results = self.ap.spinna(0, parameters)

        shutil.rmtree(os.path.join(self.results_folder, "00_spinna"))

    @patch("picasso_workflow.analyse.picasso_outpost.single_spinna_run")
    @patch("picasso_workflow.analyse.picasso_outpost.screen_label_uncertainty")
    def test_spinna_label_uncertainty_screen(self, mock_screen, mock_sptmp):
        """Screening routes to screen_label_uncertainty, feeds the
        best-fit value into single_spinna_run and prepends the scan
        figures to the reported figures."""
        best_unc = {"CD86": 6.0}
        scan = {
            "CD86": {
                "candidates": [2.0, 4.0, 6.0, 8.0],
                "scores": [0.4, 0.3, 0.1, 0.2],
            }
        }
        scan_figs = ["scanA.png"]
        mock_screen.return_value = (best_unc, scan, scan_figs)
        mock_sptmp.return_value = ("spinna-results", ["nnd.png"])

        info = [{"Width": 1000, "Height": 1000, "Frames": 10000}]
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("x", "f4"),
            ("y", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
        ]
        locs = pd.DataFrame(
            np.rec.array(
                [
                    tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                    for i in range(len(self.ap.movie))
                ],
                dtype=locs_dtype,
            )
        )
        self.ap.channel_locs = [locs]
        self.ap.channel_info = [info]
        self.ap.channel_tags = ["CD86"]

        parameters = {
            "labeling_efficiency": {"CD86": 0.54},
            "labeling_uncertainty": 5,
            "labeling_uncertainty_screen": {
                "min": 2.0,
                "max": 8.0,
                "step": 2.0,
            },
            "n_simulate": 50000,
            "fp_mask_dict": None,
            "density": [0.00009],
            "random_rot_mode": "2D",
            "n_nearest_neighbors": 4,
            "sim_repeats": 5,
            "fit_NND_bin": 5,
            "fit_NND_maxdist": 300,
            "granularity": 30,
            "structures": [
                {
                    "Molecular targets": ["CD86"],
                    "Structure title": "monomer",
                    "CD86_x": [0],
                    "CD86_y": [0],
                    "CD86_z": [0],
                },
                {
                    "Molecular targets": ["CD86"],
                    "Structure title": "dimer",
                    "CD86_x": [-10, 10],
                    "CD86_y": [0, 0],
                    "CD86_z": [0, 0],
                },
            ],
        }
        parameters, results = self.ap.spinna(0, parameters)

        # the same candidate grid is screened for every target
        mock_screen.assert_called_once()
        self.assertEqual(
            mock_screen.call_args.kwargs["label_unc"],
            {"CD86": [2.0, 4.0, 6.0, 8.0]},
        )
        # the best-fit scalar is what actually gets simulated
        self.assertEqual(mock_sptmp.call_args.kwargs["label_unc"], best_unc)
        # results carry the best value, the scan and the scan figures
        self.assertEqual(results["best_labeling_uncertainty"], best_unc)
        self.assertEqual(results["labeling_uncertainty_scan"], scan)
        self.assertEqual(results["fp_figs"], scan_figs + ["nnd.png"])

        shutil.rmtree(os.path.join(self.results_folder, "00_spinna"))

    @patch("picasso_workflow.picasso_outpost._plot_label_unc_scan")
    @patch(
        "picasso_workflow.picasso_outpost.spinna."
        "compare_models_given_label_unc"
    )
    def test_screen_label_uncertainty_selects_best(
        self, mock_compare, mock_plot
    ):
        """screen_label_uncertainty scans multi-candidate targets, keeps
        the lowest-scoring value and leaves single-candidate targets
        untouched."""
        Structure = picasso_outpost.spinna.Structure
        mono_a = Structure("mono_A")
        mono_a.define_coordinates("A", [0.0], [0.0], [0.0])
        mono_b = Structure("mono_B")
        mono_b.define_coordinates("B", [0.0], [0.0], [0.0])
        structures = [mono_a, mono_b]

        # A screened over 3 values (best is 4.0 -> lowest score), B fixed
        label_unc = {"A": [2.0, 4.0, 6.0], "B": [5.0]}
        mock_compare.side_effect = [(0.5,), (0.1,), (0.3,)]
        mock_plot.return_value = "scan_A.png"

        exp_data = {
            "A": np.random.rand(20, 2) * 100,
            "B": np.random.rand(20, 2) * 100,
        }

        best, scan, figs = picasso_outpost.screen_label_uncertainty(
            structures=structures,
            label_unc=label_unc,
            le={"A": 0.5, "B": 0.5},
            granularity=10,
            exp_data=exp_data,
            mask_dict=None,
            width=1000.0,
            height=1000.0,
            depth=None,
            random_rot_mode="2D",
            sim_repeats=1,
            asynch=False,
            result_dir="/tmp",
            save_filename="/tmp/spinna-run",
        )

        self.assertEqual(best, {"A": 4.0, "B": 5.0})
        self.assertEqual(mock_compare.call_count, 3)
        # empty savedir avoids picasso's single-candidate save crash
        self.assertEqual(mock_compare.call_args.kwargs["savedir"], "")
        self.assertEqual(scan["A"]["candidates"], [2.0, 4.0, 6.0])
        self.assertEqual(scan["A"]["scores"], [0.5, 0.1, 0.3])
        self.assertNotIn("B", scan)
        self.assertEqual(figs, ["scan_A.png"])

    @patch("picasso_workflow.analyse.picasso_outpost.single_spinna_run")
    @patch("picasso_workflow.analyse.picasso_outpost.single_spinna_fit_le_run")
    def test_spinna_pair_distance_screen(self, mock_fitle, mock_sptmp):
        """pair_distance_screen routes to fit_le (not single_spinna_run),
        passing the distance grid and per-target label_unc lists, and
        populates the fitted results."""
        mock_fitle.return_value = (
            "res",
            ["nnd.png"],
            {"CD80": 52.0, "CD86": 60.0},
            {"CD80": 5.0, "CD86": 5.0},
            12.0,
            0.08,
        )

        info = [{"Width": 1000, "Height": 1000, "Frames": 10000}]
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("x", "f4"),
            ("y", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
        ]

        def _mklocs():
            return pd.DataFrame(
                np.rec.array(
                    [
                        tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                        for i in range(len(self.ap.movie))
                    ],
                    dtype=locs_dtype,
                )
            )

        self.ap.channel_locs = [_mklocs(), _mklocs()]
        self.ap.channel_info = [info, info]
        self.ap.channel_tags = ["CD80", "CD86"]

        parameters = {
            "labeling_efficiency": {"CD80": 0.5, "CD86": 0.6},
            "labeling_uncertainty": 5,
            "pair_distance_screen": {"min": 10.0, "max": 30.0, "step": 10.0},
            "n_simulate": 50000,
            "fp_mask_dict": None,
            "density": [0.00009, 0.00009],
            "random_rot_mode": "2D",
            "n_nearest_neighbors": 4,
            "sim_repeats": 5,
            "fit_NND_bin": 5,
            "fit_NND_maxdist": 300,
            "granularity": 30,
            "structures": [
                {
                    "Molecular targets": ["CD80"],
                    "Structure title": "monoCD80",
                    "CD80_x": [0],
                    "CD80_y": [0],
                    "CD80_z": [0],
                },
                {
                    "Molecular targets": ["CD86"],
                    "Structure title": "monoCD86",
                    "CD86_x": [0],
                    "CD86_y": [0],
                    "CD86_z": [0],
                },
            ],
        }
        parameters, results = self.ap.spinna(0, parameters)

        # fit_le is used, the stoichiometry-only path is not
        mock_fitle.assert_called_once()
        mock_sptmp.assert_not_called()

        kwargs = mock_fitle.call_args.kwargs
        self.assertEqual(kwargs["distances"], [10.0, 20.0, 30.0])
        self.assertEqual(kwargs["label_unc"], {"CD80": [5], "CD86": [5]})
        self.assertEqual(kwargs["target_a"], "CD80")
        self.assertEqual(kwargs["target_b"], "CD86")

        # fitted quantities are surfaced in the results
        self.assertEqual(results["best_pair_distance"], 12.0)
        self.assertEqual(
            results["fitted_labeling_efficiency"],
            {"CD80": 52.0, "CD86": 60.0},
        )
        self.assertEqual(
            results["best_labeling_uncertainty"],
            {"CD80": 5.0, "CD86": 5.0},
        )
        self.assertEqual(results["fp_figs"], ["nnd.png"])

        shutil.rmtree(os.path.join(self.results_folder, "00_spinna"))

    @patch("picasso_workflow.picasso_outpost.plot_spinna_nnd")
    @patch("picasso_workflow.picasso_outpost.spinna.fit_le")
    def test_single_spinna_fit_le_run(self, mock_fit_le, mock_plot):
        """single_spinna_fit_le_run forwards the screen inputs to
        spinna.fit_le, plots the best-fit mixer and returns the fitted
        quantities."""
        best_mixer = MagicMock()
        best_mixer.get_structure_names.return_value = ["mA", "mB", "het"]
        mock_fit_le.return_value = (
            {"A": 52.0, "B": 60.0},  # le_values
            {"A": 5.0, "B": 5.0},  # fitted_label_unc
            12.0,  # best_distance
            0.08,  # best_score
            np.array([50.0, 30.0, 20.0]),  # best_props
            best_mixer,
        )
        mock_plot.return_value = ["nnd.png"]

        exp_data = {
            "A": np.random.rand(10, 2) * 100,
            "B": np.random.rand(10, 2) * 100,
        }

        tmpdir = tempfile.mkdtemp()
        try:
            out = picasso_outpost.single_spinna_fit_le_run(
                target_a="A",
                target_b="B",
                exp_data=exp_data,
                granularity=10,
                label_unc={"A": [2.0, 4.0], "B": [5.0]},
                distances=[10.0, 20.0, 30.0],
                mask_dict=None,
                width=1000.0,
                height=1000.0,
                depth=None,
                random_rot_mode="2D",
                sim_repeats=1,
                asynch=False,
                NND_bin=5.0,
                NND_maxdist=300.0,
                nn_plotted=4,
                n_simulated={"A": 100, "B": 100},
                result_dir=tmpdir,
                save_filename=os.path.join(tmpdir, "fitle"),
            )
        finally:
            shutil.rmtree(tmpdir)

        (
            results,
            fp_fig,
            le_values,
            fitted_label_unc,
            best_distance,
            best_score,
        ) = out

        mock_fit_le.assert_called_once()
        kw = mock_fit_le.call_args.kwargs
        self.assertEqual(kw["distances"], [10.0, 20.0, 30.0])
        self.assertEqual(kw["label_unc"], {"A": [2.0, 4.0], "B": [5.0]})
        self.assertEqual(kw["target_a"], "A")
        self.assertEqual(kw["target_b"], "B")
        # empty savedir avoids picasso's single-candidate save crash
        self.assertEqual(kw["savedir"], "")

        self.assertEqual(best_distance, 12.0)
        self.assertEqual(best_score, 0.08)
        self.assertEqual(le_values, {"A": 52.0, "B": 60.0})
        self.assertEqual(fitted_label_unc, {"A": 5.0, "B": 5.0})
        self.assertEqual(fp_fig, ["nnd.png"])
        self.assertEqual(results["Best pair distance (nm)"], 12.0)

        mock_plot.assert_called_once()
        self.assertIs(mock_plot.call_args.kwargs["mixer"], best_mixer)

    @patch("picasso_workflow.analyse.picasso_outpost.plot_spinna_nnd")
    @patch("picasso_workflow.analyse.spinna.fit_le")
    def test_labeling_efficiency_analysis_screen(self, mock_fit_le, mock_plot):
        """pair_distance_screen and labeling_uncertainty_screen expand
        into the distance list and per-tag label_unc lists that
        spinna.fit_le consumes, and the fitted values reach results."""
        best_mixer = MagicMock()
        best_mixer.get_structure_names.return_value = ["mA", "mB", "het"]
        mock_fit_le.return_value = (
            {"CD80": 52.0, "CD86": 60.0},  # le_values (percent)
            {"CD80": 6.0, "CD86": 6.0},  # fitted_label_unc
            12.0,  # best_distance
            0.08,  # best_score
            np.array([50.0, 30.0, 20.0]),  # best_props
            best_mixer,
        )
        # the module reads fp_fig_out[0..3]
        mock_plot.return_value = ["a.png", "b.png", "c.png", "d.png"]

        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("x", "f4"),
            ("y", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
        ]

        def _mklocs():
            return pd.DataFrame(
                np.rec.array(
                    [
                        tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                        for i in range(len(self.ap.movie))
                    ],
                    dtype=locs_dtype,
                )
            )

        self.ap.channel_locs = [_mklocs(), _mklocs()]
        self.ap.channel_tags = ["CD80", "CD86"]

        parameters = {
            "target_name": "CD80",
            "reference_name": "CD86",
            "pair_distance": 10,
            "pair_distance_screen": {"min": 8.0, "max": 16.0, "step": 4.0},
            "labeling_uncertainty": {"CD80": 5, "CD86": 5},
            "labeling_uncertainty_screen": {
                "min": 2.0,
                "max": 6.0,
                "step": 2.0,
            },
            "n_simulate": 50000,
            "density": {"CD80": 9e-5, "CD86": 9e-5},
            "granularity": 30,
            "sim_repeats": 5,
            "nn_nth": 2,
        }
        parameters, results = self.ap.labeling_efficiency_analysis(
            0, parameters
        )

        mock_fit_le.assert_called_once()
        kw = mock_fit_le.call_args.kwargs
        self.assertEqual(kw["distances"], [8.0, 12.0, 16.0])
        self.assertEqual(
            kw["label_unc"],
            {"CD80": [2.0, 4.0, 6.0], "CD86": [2.0, 4.0, 6.0]},
        )
        self.assertEqual(kw["target_a"], "CD80")
        self.assertEqual(kw["target_b"], "CD86")
        # empty savedir avoids picasso's single-candidate save crash
        self.assertEqual(kw["savedir"], "")

        self.assertEqual(results["best_pair_distance"], 12.0)
        self.assertEqual(
            results["best_labeling_uncertainty"],
            {"CD80": 6.0, "CD86": 6.0},
        )
        self.assertAlmostEqual(results["labeling_efficiency"]["CD80"], 0.52)
        self.assertAlmostEqual(results["labeling_efficiency"]["CD86"], 0.60)

        shutil.rmtree(
            os.path.join(
                self.results_folder, "00_labeling_efficiency_analysis"
            )
        )

    @patch("picasso_workflow.analyse.picasso_outpost.spinna_batch")
    @patch("picasso_workflow.analyse.io.save_locs")
    def spinna_batch(self, mock_save_locs, mock_spinna_batch):
        # a minimal spinna batch config with two analysis rows; the
        # module must broadcast the locs filepath across all rows.
        cfg_fp = os.path.join(self.results_folder, "spinna_batch_config.csv")
        pd.DataFrame(
            {
                "structures_filename": ["structures.yaml"] * 2,
                "granularity": [30, 30],
                "save_filename": ["run1", "run2"],
                "NND_bin": [5, 5],
                "NND_maxdist": [300, 300],
                "sim_repeats": [5, 5],
            }
        ).to_csv(cfg_fp, index=False)
        cfg_original = pd.read_csv(cfg_fp)

        result_dir = os.path.join(
            self.results_folder, "spinna_batch_config_fitting_results"
        )
        fp_summary = os.path.join(result_dir, "summary_results.csv")
        fp_figs = [os.path.join(result_dir, "run1_NND_CD86_CD86.png")]
        mock_spinna_batch.return_value = (result_dir, fp_summary, fp_figs)

        info = [{"Width": 1000, "Height": 1000, "Frames": 10000}]
        locs_dtype = [("frame", "u4"), ("x", "f4"), ("y", "f4")]
        locs = pd.DataFrame(
            np.rec.array([(i, 0.0, 0.0) for i in range(5)], dtype=locs_dtype)
        )
        self.ap.channel_locs = [locs]
        self.ap.channel_info = [info]
        self.ap.channel_tags = ["CD86"]

        parameters = {
            "fp_spinna_batch_config": cfg_fp,
            "use_workflow_locs": True,
        }
        parameters, results = self.ap.spinna_batch(0, parameters)

        # the modified config is written to a copy in the results
        # folder, and that copy is the one passed to picasso.
        cfg_fp_used = os.path.join(
            results["folder"], "spinna_batch_config.csv"
        )
        mock_spinna_batch.assert_called_once_with(cfg_fp_used)
        assert results["fp_spinna_batch_config"] == cfg_fp_used
        assert results["success"] is True
        assert results["result_dir"] == result_dir
        assert results["fp_summary"] == fp_summary
        assert results["fp_figs"] == fp_figs

        # the current locs are saved once per channel
        mock_save_locs.assert_called_once()

        # the locs filepath is written into the config copy for every
        # row, under the target-specific column.
        written_cfg = pd.read_csv(cfg_fp_used)
        assert "exp_data_CD86" in written_cfg.columns
        assert len(written_cfg) == 2
        expected_locs_fp = os.path.join(results["folder"], "CD86.hdf5")
        assert list(written_cfg["exp_data_CD86"]) == [expected_locs_fp] * 2

        # the user's original config file is left untouched
        pd.testing.assert_frame_equal(pd.read_csv(cfg_fp), cfg_original)

        os.remove(cfg_fp)
        shutil.rmtree(os.path.join(self.results_folder, "00_spinna_batch"))

    @patch("picasso_workflow.analyse.picasso_outpost.spinna_batch")
    @patch("picasso_workflow.analyse.io.save_locs")
    def test_spinna_batch_single_dataset(
        self, mock_save_locs, mock_spinna_batch
    ):
        """spinna_batch falls back to self.locs when no channels are set."""
        cfg_fp = os.path.join(
            self.results_folder, "spinna_batch_config_sgl.csv"
        )
        pd.DataFrame(
            {
                "structures_filename": ["structures.yaml"],
                "granularity": [30],
                "save_filename": ["run1"],
                "NND_bin": [5],
                "NND_maxdist": [300],
                "sim_repeats": [5],
            }
        ).to_csv(cfg_fp, index=False)
        mock_spinna_batch.return_value = ("res_dir", "summary.csv", [])

        self.ap.channel_tags = None
        self.ap.locs = pd.DataFrame(
            np.rec.array(
                [(i, 0.0, 0.0) for i in range(3)],
                dtype=[("frame", "u4"), ("x", "f4"), ("y", "f4")],
            )
        )
        self.ap.info = [{"Width": 1000, "Height": 1000, "Frames": 10000}]

        parameters = {
            "fp_spinna_batch_config": cfg_fp,
            "use_workflow_locs": True,
        }
        parameters, results = self.ap.spinna_batch(0, parameters)

        assert results["success"] is True
        mock_save_locs.assert_called_once()
        cfg_fp_used = os.path.join(
            results["folder"], "spinna_batch_config_sgl.csv"
        )
        assert results["fp_spinna_batch_config"] == cfg_fp_used
        written_cfg = pd.read_csv(cfg_fp_used)
        assert "exp_data_locs" in written_cfg.columns

        os.remove(cfg_fp)
        shutil.rmtree(os.path.join(self.results_folder, "00_spinna_batch"))

    @patch("picasso_workflow.metaworkflow.platform.node")
    @patch(
        "picasso_workflow.metaworkflow.CONFIG",
        {
            "Drivepaths": {
                "srcmachineXXX": ["/src/pool-a", "/src/pool-b"],
                "dstmachineXXX": ["/dst/pool-a", "/dst/pool-b"],
            }
        },
    )
    @patch("picasso_workflow.analyse.picasso_outpost.spinna_batch")
    @patch("picasso_workflow.analyse.io.save_locs")
    def test_spinna_batch_converts_paths(
        self, mock_save_locs, mock_spinna_batch, mock_node
    ):
        """Cross-machine file paths in the config are converted, while
        the workflow's own channel keeps its local locs filepath."""
        mock_node.return_value = "dstmachine001"
        mock_spinna_batch.return_value = ("res_dir", "summary.csv", [])

        cfg_fp = os.path.join(
            self.results_folder, "spinna_batch_config_conv.csv"
        )
        pd.DataFrame(
            {
                "structures_filename": ["/src/pool-a/structs.yaml"],
                "exp_data_OTHER": ["/src/pool-b/other.hdf5"],
                "mask_filename_OTHER": ["/src/pool-a/mask.npy"],
                "granularity": [30],
                "save_filename": ["run1"],
                "NND_bin": [5],
                "NND_maxdist": [300],
                "sim_repeats": [5],
            }
        ).to_csv(cfg_fp, index=False)

        info = [{"Width": 1000, "Height": 1000, "Frames": 10000}]
        locs = pd.DataFrame(
            np.rec.array(
                [(i, 0.0, 0.0) for i in range(5)],
                dtype=[("frame", "u4"), ("x", "f4"), ("y", "f4")],
            )
        )
        self.ap.channel_locs = [locs]
        self.ap.channel_info = [info]
        self.ap.channel_tags = ["CD86"]

        parameters = {
            "fp_spinna_batch_config": cfg_fp,
            "use_workflow_locs": True,
        }
        parameters, results = self.ap.spinna_batch(0, parameters)

        written_cfg = pd.read_csv(results["fp_spinna_batch_config"])
        # foreign-machine paths converted to the current machine's roots
        assert (
            written_cfg["structures_filename"].iloc[0]
            == "/dst/pool-a/structs.yaml"
        )
        assert (
            written_cfg["exp_data_OTHER"].iloc[0] == "/dst/pool-b/other.hdf5"
        )
        assert (
            written_cfg["mask_filename_OTHER"].iloc[0]
            == "/dst/pool-a/mask.npy"
        )
        # this workflow's own channel keeps its freshly saved local locs
        expected_locs_fp = os.path.join(results["folder"], "CD86.hdf5")
        assert written_cfg["exp_data_CD86"].iloc[0] == expected_locs_fp

        os.remove(cfg_fp)
        shutil.rmtree(os.path.join(self.results_folder, "00_spinna_batch"))

    # @patch("picasso_workflow.analyse.picasso_outpost.spinna_temp", MagicMock)
    def spinna_manual(self):
        info = [{"Width": 1000, "Height": 1000, "Frames": 10000}]
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("x", "f4"),
            ("y", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
        ]
        locs = np.rec.array(
            [
                tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        locs = pd.DataFrame(locs)
        self.ap.channel_locs = [locs]
        self.ap.channel_info = [info]
        self.ap.channel_tags = ["CD86"]

        parameters = {
            "proposed_labeling_efficiency": 50,
            "proposed_labeling_uncertainty": 6,
            "proposed_n_simulate": 50000,
            "proposed_density": 0.56,
            "proposed_nn_plotted": 4,
            "structures_d": 10,
        }
        # test preparatory stage
        parameters, results = self.ap.spinna_manual(0, parameters)

        # test calling spinna
        parameters, results = self.ap.spinna_manual(0, parameters)

        shutil.rmtree(os.path.join(self.results_folder, "00_spinna_manual"))

    def analysis_documentation(self):
        return
        shutil.rmtree(
            os.path.join(self.results_folder, "00_analysis_documentation")
        )

    def dummy_module(self):
        return
        shutil.rmtree(os.path.join(self.results_folder, "00_dummy_module"))

    def random_val(self):
        return
        shutil.rmtree(os.path.join(self.results_folder, "00_random_val"))

    def render(self):
        return
        shutil.rmtree(os.path.join(self.results_folder, "00_render"))

    def find_structures(self):
        return
        shutil.rmtree(os.path.join(self.results_folder, "00_find_structures"))

    def pairwise_module_executor(self):
        return
        shutil.rmtree(
            os.path.join(self.results_folder, "00_pairwise_module_executor")
        )

    def ripleysk(self):
        return
        shutil.rmtree(os.path.join(self.results_folder, "00_ripleysk"))

    def ripleysk2(self):
        return
        shutil.rmtree(os.path.join(self.results_folder, "00_ripleysk2"))

    def ripleysk_average(self):
        return
        shutil.rmtree(os.path.join(self.results_folder, "00_ripleysk"))

    def ripleysk_average2(self):
        return
        shutil.rmtree(
            os.path.join(self.results_folder, "00_ripleysk_average2")
        )

    def protein_interactions(self):
        return
        shutil.rmtree(
            os.path.join(self.results_folder, "00_protein_interactions")
        )

    def create_mask(self):
        """Create a density mask"""
        return
        shutil.rmtree(os.path.join(self.results_folder, "00_create_mask"))

    def create_mask2(self):
        """Create a density mask"""
        return
        shutil.rmtree(os.path.join(self.results_folder, "00_create_mask2"))

    def refine_mask_by_density(self):
        """Create a density mask"""
        return
        shutil.rmtree(
            os.path.join(self.results_folder, "00_refine_mask_by_density")
        )

    def dbscan_molint(self):
        """TO BE CLEANED UP
        dbscan implementation for molecular interactions workflow
        """
        return
        shutil.rmtree(os.path.join(self.results_folder, "00_dbscan_molint"))

    def CSR_sim_in_mask(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask
        """
        return
        shutil.rmtree(os.path.join(self.results_folder, "00_CSR_sim_in_mask"))

    def find_cluster_motifs(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask
        """
        return
        shutil.rmtree(
            os.path.join(self.results_folder, "00_find_cluster_motifs")
        )

    def interaction_graph(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask
        """
        return
        shutil.rmtree(
            os.path.join(self.results_folder, "00_interaction_graph")
        )

    def plot_densities(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask
        """
        return
        shutil.rmtree(os.path.join(self.results_folder, "00_plot_densities"))

    def protein_interactions_average(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask
        """
        return
        shutil.rmtree(
            os.path.join(
                self.results_folder, "00_protein_interactions_average"
            )
        )

    @patch("picasso_workflow.analyse.picasso_outpost.pick_gold")
    @patch("picasso_workflow.analyse.picasso_outpost.picked_locs")
    @patch("picasso_workflow.analyse.io.save_locs")
    def find_gold(self, mock_save_locs, mock_picked_locs, mock_pick_gold):
        parameters = {}
        mock_pick_gold.return_value = [[2, 4], [4, 2], [4, 4]]
        mock_picked_locs.return_value = (self.ap.locs, self.ap.locs)
        parameters, results = self.ap.find_gold(0, parameters)

        shutil.rmtree(os.path.join(self.results_folder, "00_find_gold"))

    @patch("picasso_workflow.analyse.picasso_outpost._undrift_from_picked")
    @patch("picasso_workflow.analyse.io.save_locs")
    @patch("picasso_workflow.analyse.io.load_locs")
    def undrift_from_picked(
        self, mock_load_locs, mock_save_locs, mock_undrift
    ):
        # parameters = {"fp_picked_locs": "fp"}
        # mock_undrift.return_value = (
        #     "locs",
        #     [{"name": "info"}],
        #     ([2, 4, 3], [3, 2, 1]),
        # )
        # mock_save_locs.return_value = None
        # mock_load_locs.return_value = "locs", [{"name": "info"}]
        # parameters, results = self.ap.undrift_from_picked(0, parameters)

        # shutil.rmtree(
        #     os.path.join(self.results_folder, "00_undrift_from_picked")
        # )
        pass

    @patch("picasso_workflow.analyse.io.save_locs", MagicMock)
    def filter_locs(self):
        parameters = {"field": "photons", "minval": 800, "maxval": 1200}
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
        self.ap.locs = np.rec.array(
            [
                tuple([i] + list(1000 * np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        self.ap.locs = pd.DataFrame(self.ap.locs)

        parameters, results = self.ap.filter_locs(0, parameters)

        shutil.rmtree(os.path.join(self.results_folder, "00_filter_locs"))

    def filter_transient_binding(self):
        parameters = {}
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("x", "f4"),
            ("y", "f4"),
            ("sx", "f4"),
            ("sy", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
            ("std_frame", "u4"),
        ]
        self.ap.locs = np.rec.array(
            [
                tuple([i] + list(1000 * np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        self.ap.locs = pd.DataFrame(self.ap.locs)
        self.ap.info = [{"Frames": 1000}]

        parameters, results = self.ap.filter_transient_binding(0, parameters)

        shutil.rmtree(
            os.path.join(self.results_folder, "00_filter_transient_binding")
        )

    @patch("picasso_workflow.analyse.io.save_locs", MagicMock())
    @patch("picasso_workflow.analyse.postprocess.link", MagicMock())
    def link_locs(self):
        parameters = {"d_max": 2, "tolerance": 3}

        parameters, results = self.ap.link_locs(0, parameters)

        shutil.rmtree(os.path.join(self.results_folder, "00_link_locs"))

    @patch("picasso_workflow.analyse.picasso_outpost.plot_spinna_nnd")
    @patch("picasso_workflow.analyse.spinna.fit_le")
    def labeling_efficiency_analysis(self, mock_fit_le, mock_plot_nnd):
        parameters = {
            "target_name": "CD86",
            "reference_name": "GFP",
            "pair_distance": 10,
            "density": {"CD86": 92.4, "GFP": 83.5},
            "n_simulate": 10000,
            "granularity": 5,
            "labeling_uncertainty": {"CD86": 5, "GFP": 5},
            "sim_repeats": 2,
            # "nn_nth": 2,
        }
        # fit_le returns LE values in percent. With monomer/heterodimer
        # proportions [CD86_only, GFP_only, AB] = [0.40, 0.15, 0.35] the
        # LE (structure basis, AB halved) is AB/(monomer+AB):
        #   LE_CD86 = 0.175 / (0.15 + 0.175) = 53.85 %
        #   LE_GFP  = 0.175 / (0.40 + 0.175) = 30.43 %
        best_props = np.array([0.40, 0.15, 0.35])
        le_values = {"CD86": 53.846153846, "GFP": 30.434782609}
        mock_fit_le.return_value = (
            le_values,
            {"CD86": [5], "GFP": [5]},  # fitted_label_unc
            10.0,  # best_distance
            0.5,  # best_score
            best_props,
            MagicMock(),  # best_mixer (unused; plotting is mocked)
        )
        mock_plot_nnd.return_value = [
            "/path/to/figAA.png",
            "/path/to/figAB.png",
            "/path/to/figBA.png",
            "/path/to/figBB.png",
        ]
        self.ap.channel_tags = ["GFP", "CD86"]
        self.ap.channel_locs = [None, None]
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
        locs_a = np.rec.array(
            [
                tuple([i] + list(1000 * np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        locs_a = pd.DataFrame(locs_a)
        locs_b = np.rec.array(
            [
                tuple([i] + list(1000 * np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        locs_b = pd.DataFrame(locs_b)
        self.ap.channel_locs = [locs_a, locs_b]

        parameters, results = self.ap.labeling_efficiency_analysis(
            0, parameters
        )

        # LE stored on the 0-1 scale (fit_le returns percent)
        self.assertAlmostEqual(
            results["labeling_efficiency"]["CD86"], 0.53846153846, places=5
        )
        self.assertAlmostEqual(
            results["labeling_efficiency"]["GFP"], 0.30434782609, places=5
        )
        # no bootstrap requested -> std is zero but the key must exist
        self.assertEqual(results["labeling_efficiency_std"]["CD86"], 0.0)
        self.assertEqual(results["labeling_efficiency_std"]["GFP"], 0.0)
        self.assertEqual(len(results["fp_fig"]), 4)

        shutil.rmtree(
            os.path.join(
                self.results_folder, "00_labeling_efficiency_analysis"
            )
        )

    def conditional_branch(self):
        """Test conditional_branch module - simple test for test_modules()"""
        # Set up mock locs data
        locs_dtype = [
            ("frame", "u4"),
            ("x", "f4"),
            ("y", "f4"),
            ("photons", "f4"),
        ]
        self.ap.locs = pd.DataFrame(
            np.rec.array(
                [(0, 1.0, 1.0, 100.0), (1, 2.0, 2.0, 200.0)],
                dtype=locs_dtype,
            )
        )

        # Simple condition: 5 > 3 (always true)
        parameters = {
            "condition": {"left": 5, "operator": ">", "right": 3},
            "if_true": [],  # Empty list for simplicity in test_modules
            "if_false": [],
        }

        parameters, results = self.ap.conditional_branch(0, parameters)

        assert results["condition_result"] is True
        assert results["branch_taken"] == "if_true"
        assert "if_branch" in results
        assert "branch_modules" in results

        # Clean up
        shutil.rmtree(
            os.path.join(self.results_folder, "00_conditional_branch")
        )

    def resolution_frc_spatial(self):
        """Is tested separately in tests/outpost_modules/test_resolution_frc.py"""

    def resolution_analysis(self):
        pass

    # @unittest.skip("")
    def undrift_rsso(self):
        """Test the undrift_rsso module with synthetic drift data"""
        import numpy as np

        # Create synthetic data with known drift
        np.random.seed(42)
        n_frames = 50  # Fewer frames for faster test
        n_locs_per_frame = 200  # More locs per frame for better statistics
        true_drift_rate_x = 0.05  # Smaller drift rate (pixels per frame)
        true_drift_rate_y = 0.03  # Smaller drift rate (pixels per frame)

        # Generate synthetic localizations with drift
        all_locs = []
        true_drift_x = []
        true_drift_y = []

        for frame in range(n_frames):
            # Calculate cumulative drift
            drift_x = true_drift_rate_x * frame
            drift_y = true_drift_rate_y * frame
            true_drift_x.append(drift_x)
            true_drift_y.append(drift_y)

            # Generate random localizations with added drift
            x_base = np.random.uniform(10, 50, n_locs_per_frame)
            y_base = np.random.uniform(10, 50, n_locs_per_frame)

            frame_locs = np.rec.array(
                [
                    (
                        frame,
                        x + drift_x,
                        y + drift_y,
                        1000,
                        1.0,
                        1.0,
                        100,
                        0.1,
                        0.1,
                        0.5,
                        50,
                        1,
                    )
                    for x, y in zip(x_base, y_base)
                ],
                dtype=self.locs_dtype,
            )
            frame_locs = pd.DataFrame(frame_locs)

            all_locs.append(frame_locs)

        # Combine all localizations
        # synthetic_locs = np.lib.recfunctions.stack_arrays(
        #     all_locs, asrecarray=True, usemask=False
        # )
        synthetic_locs = pd.concat(all_locs, ignore_index=True)

        # Create AutoPicasso instance with synthetic data
        analysis_config = {
            "camera_info": {
                "Gain": 1,
                "Sensitivity": 0.45,
                "Baseline": 100,
                "Qe": 0.82,
                "Pixelsize": 130,  # nm
            },
            "gpufit_installed": False,
        }
        ap = analyse.AutoPicasso(self.results_folder, analysis_config)
        ap.locs = synthetic_locs
        ap.info = [
            {"Frames": n_frames, "Width": 64, "Height": 64, "Pixelsize": 100}
        ]  # 100 nm pixels

        # Test parameters
        parameters = {
            "ton": 3.0,  # 3 frame half-life
            "toff": 10.0,  # 10 frame reappearance time
            "max_shift": 0.5,  # 0.5 pixel max shift per frame
            "processing_chunk_size": 25,  # Processing chunk size
            "min_locs_per_frame": 20,  # Ensure sufficient data for frame-level
            "min_locs_per_block": 100,  # Ensure sufficient data for block-level
            "plot_drift": True,
            "save_locs": False,
        }

        # Run undrift_rsso
        parameters, results = ap.undrift_rsso(0, parameters)

        # Verify results - check for essential outputs
        # The function should complete and create plots

        # Check that drift plot was created
        # assert "fp_fig" in results, "Should create drift plot"
        # assert os.path.exists(
        #     results["fp_fig"]
        # ), "Drift plot file should exist"

        # # Verify that drift arrays are stored in AutoPicasso instance
        # assert hasattr(ap, "drift"), "Should store drift in ap.drift"
        # assert hasattr(ap.drift, "shape"), "Drift should be array-like"
        # assert len(ap.drift.shape) == 2, "Drift should be 2D array"
        # assert ap.drift.shape[1] == 2, "Should have x and y drift dimensions"
        # assert (
        #     ap.drift.shape[0] == n_frames
        # ), "Drift array should match number of frames"

        # # Verify drift was detected (should be non-zero since we added artificial drift)
        # drift_magnitude = np.sqrt(np.sum(ap.drift**2, axis=1)).mean()
        # assert drift_magnitude > 0, f"Should detect non-zero drift, got {drift_magnitude}"

        # Clean up plots
        import glob

        undrift_folder = os.path.join(self.results_folder, "00_undrift_rsso")
        if os.path.exists(undrift_folder):
            for pattern in [
                "drift_*.png",
                "convergence_*.png",
                "robustness_*.png",
            ]:
                for file in glob.glob(os.path.join(undrift_folder, pattern)):
                    try:
                        os.remove(file)
                    except Exception:
                        pass

    @unittest.skip("")
    def test_16_undrift_rsso_edge_cases(self):
        """Test undrift_rsso with edge cases and error conditions"""
        # Test with insufficient data
        analysis_config = {
            "camera_info": {
                "Gain": 1,
                "Sensitivity": 0.45,
                "Baseline": 100,
                "Qe": 0.82,
                "Pixelsize": 130,  # nm
            },
            "gpufit_installed": False,
        }
        ap = analyse.AutoPicasso(self.results_folder, analysis_config)

        # Create minimal dataset (too few localizations)
        minimal_locs = np.rec.array(
            [
                (0, 10.0, 10.0, 1000, 1.0, 1.0, 100, 0.1, 0.1, 0.5, 50, 1),
                (1, 10.1, 10.1, 1000, 1.0, 1.0, 100, 0.1, 0.1, 0.5, 50, 1),
            ],
            dtype=self.locs_dtype,
        )
        minimal_locs = pd.DataFrame(minimal_locs)

        ap.locs = minimal_locs
        ap.info = [{"Frames": 2, "Width": 64, "Height": 64, "Pixelsize": 100}]

        parameters = {
            "ton": 5.0,
            "toff": 20.0,
            "max_shift": 1.0,
            "min_locs_per_frame": 100,  # Impossibly high threshold
            "plot_drift": False,
            "save_locs": False,
        }

        # Should handle gracefully
        parameters, results = ap.undrift_rsso(0, parameters)

        # Should still succeed even with insufficient data
        assert results["success"]
        assert "total_drift" in results

        # Test with no localizations
        empty_locs = np.array([], dtype=self.locs_dtype).view(np.recarray)
        empty_locs = pd.DataFrame(empty_locs)
        ap.locs = empty_locs

        parameters, results = ap.undrift_rsso(0, parameters)
        assert results["success"]
        assert results["total_drift"] == 0.0

    @patch("picasso_workflow.analyse.picasso_outpost.pick_similar")
    @patch("picasso_workflow.analyse.picasso_outpost.picked_locs")
    @patch("picasso_workflow.analyse.render.plot_scene")
    @patch("picasso_workflow.analyse.io.save_locs")
    def find_similar(
        self,
        mock_save_locs,
        mock_plot_scene,
        mock_picked_locs,
        mock_pick_similar,
    ):
        """Test the find_similar module"""
        # Mock pick_similar to return test data
        mock_picks = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        mock_nlocs = np.array([50, 75, 100])
        mock_rmsds = np.array([1.5, 2.0, 2.5])
        mock_labels = np.array(
            [0, 0, -1]
        )  # Two selected picks, one not selected

        mock_pick_similar.return_value = (
            mock_picks,
            mock_nlocs,
            mock_rmsds,
            mock_labels,
        )

        # Mock picked_locs to return test localizations
        test_locs = np.rec.array(
            [
                (0, 10.0, 10.0, 1000, 1.0, 1.0, 100, 0.1, 0.1, 0.5, 50, 1, 0),
                (1, 10.1, 10.1, 1000, 1.0, 1.0, 100, 0.1, 0.1, 0.5, 50, 1, 0),
                (2, 20.0, 20.0, 1000, 1.0, 1.0, 100, 0.1, 0.1, 0.5, 50, 1, 1),
            ],
            dtype=self.locs_dtype + [("group", "<i4")],
        )
        test_locs = pd.DataFrame(test_locs)
        mock_picked_locs.return_value = test_locs

        # Set up test data
        self.ap.locs = np.rec.array(
            [
                (
                    i,
                    10 + np.random.rand(),
                    10 + np.random.rand(),
                    1000,
                    1.0,
                    1.0,
                    100,
                    0.1,
                    0.1,
                    0.5,
                    50,
                    i,
                )
                for i in range(100)
            ],
            dtype=self.locs_dtype,
        )
        self.ap.locs = pd.DataFrame(self.ap.locs)
        self.ap.info = [{"Width": 64, "Height": 64, "Pixelsize": 130}]

        # Test parameters
        parameters = {
            "diameter": 5.0,
            "min_n_locs_per_frame": 0.01,
            "max_n_locs_per_frame": 0.1,
            "min_rmsd": 1.0,
            "max_rmsd": 3.0,
            "n_plot_structures": 2,
            "display_pixelsize": 1.0,
        }

        # Run find_similar
        parameters, results = self.ap.find_similar(0, parameters)

        # Verify results
        assert results["n_picks"] == 3, "Should return correct number of picks"
        assert (
            results["n_picked_locs"] == 3
        ), "Should return correct number of picked locs"
        assert (
            results["n_locs"] == 100
        ), "Should return correct total number of locs"

        # Check that files were created
        assert "fp_phasespace" in results, "Should create phase space plot"
        assert (
            "fp_phasespace_hexbin" in results
        ), "Should create hexbin phase space plot"
        assert "fp_picked_fullfov" in results, "Should create full FOV plot"
        assert "fp_picked_locs" in results, "Should save picked locs file"

        # Verify mock calls (may be called multiple times by test framework)
        assert mock_pick_similar.called, "pick_similar should be called"
        assert mock_picked_locs.called, "picked_locs should be called"
        assert mock_save_locs.called, "save_locs should be called"

        # Verify pick_similar was called with correct arguments
        call_args = mock_pick_similar.call_args
        assert (
            call_args[0][0] is self.ap.locs
        ), "Should pass locs to pick_similar"
        assert (
            call_args[0][1] is self.ap.info
        ), "Should pass info to pick_similar"
        assert (
            call_args[1]["diameter"] == 5.0
        ), "Should pass diameter parameter"

        # Clean up
        shutil.rmtree(os.path.join(self.results_folder, "00_find_similar"))


# @unittest.skip("")
class TestAnalyse(unittest.TestCase):
    """Tests the implementation of methods in AutoPicasso other than
    the analysis modules defined in util.AbstractModuleCollection
    """

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
        self.results_folder = os.path.normpath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"
            )
        )
        analysis_config = {
            "camera_info": {
                "Gain": 1,
                "Sensitivity": 0.45,
                "Baseline": 100,
                "Qe": 0.82,
                "Pixelsize": 130,  # nm
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
            analysis_config = {}

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

    @patch("picasso_workflow.analyse.localize.get_spots")
    def test_04_AutoPicasso_auto_min_netgrad(self, mock_get_spots):
        mock_get_spots.return_value = [
            np.random.randint(0, 1000, size=(7, 7), dtype=np.uint16)
        ] * 48
        fn = os.path.join(self.results_folder, "autominnet.png")
        results = self.ap._auto_min_netgrad(
            box_size=7, frame_numbers=[9], filename=fn
        )
        logger.debug(results)
        assert results["filename"] == fn

        os.remove(fn)

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
        self.ap.locs = pd.DataFrame(self.ap.locs)
        # logger.debug(self.ap.locs)

        filepath = os.path.join(self.results_folder, "lvf.png")
        self.ap._plot_locs_vs_frame(filepath)

        os.remove(filepath)

    def test_07b_AutoPicasso_plot_locs_vs_frame_missing_frames(self):
        """Frames without localizations must not break the plot.

        Regression: when the illumination switches on mid-acquisition, the
        early frames contain no localizations. ``groupby("frame")`` drops
        them, so the per-frame aggregates were shorter than the full frame
        range and matplotlib raised "x and y must have same first
        dimension". The aggregates are now reindexed onto every frame.
        """
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("sx", "f4"),
            ("sy", "f4"),
        ]
        # Localizations only appear in the second half of the movie.
        first_lit_frame = len(self.ap.movie) // 2
        self.ap.locs = np.rec.array(
            [
                tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                for i in range(first_lit_frame, len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        self.ap.locs = pd.DataFrame(self.ap.locs)

        filepath = os.path.join(self.results_folder, "lvf_missing.png")
        self.ap._plot_locs_vs_frame(filepath)

        os.remove(filepath)

    def test_08_conditional_branch_true(self):
        """Test conditional_branch when condition evaluates to True"""
        # Set up mock locs data
        locs_dtype = [
            ("frame", "u4"),
            ("x", "f4"),
            ("y", "f4"),
            ("photons", "f4"),
            ("sx", "f4"),
            ("sy", "f4"),
            ("bg", "f4"),
        ]
        self.ap.locs = pd.DataFrame(
            np.rec.array(
                [
                    (0, 1.0, 1.0, 100.0, 1.5, 1.5, 10.0),
                    (1, 2.0, 2.0, 200.0, 1.5, 1.5, 10.0),
                ],
                dtype=locs_dtype,
            )
        )

        # Create a mock sub-module that does something simple
        def mock_sub_module(self, i, parameters, results=None, **kwargs):
            if results is None:
                results = {}
            results["test_value"] = "executed"
            results["folder"] = kwargs.get("calling_module_dir", "")
            return parameters, results

        # Temporarily add the mock module to AutoPicasso
        self.ap.mock_sub_module = mock_sub_module.__get__(
            self.ap, analyse.AutoPicasso
        )

        # Test condition: 10 > 5 (True)
        parameters = {
            "condition": {"left": 10, "operator": ">", "right": 5},
            "if_true": [("mock_sub_module", {})],
            "if_false": [],
        }

        parameters, results = self.ap.conditional_branch(0, parameters)

        # Verify the condition was evaluated correctly
        assert results["condition_result"] is True
        assert results["branch_taken"] == "if_true"
        assert "if_branch" in results
        assert len(results["if_branch"]) == 1
        assert "00_mock_sub_module" in results["if_branch"]
        assert "skipped_branch" in results
        assert results["skipped_branch"] == "if_false"

        # Clean up
        shutil.rmtree(
            os.path.join(self.results_folder, "00_conditional_branch")
        )

    def test_09_conditional_branch_false(self):
        """Test conditional_branch when condition evaluates to False"""
        # Set up mock locs data
        locs_dtype = [
            ("frame", "u4"),
            ("x", "f4"),
            ("y", "f4"),
            ("photons", "f4"),
        ]
        self.ap.locs = pd.DataFrame(
            np.rec.array(
                [(0, 1.0, 1.0, 100.0), (1, 2.0, 2.0, 200.0)],
                dtype=locs_dtype,
            )
        )

        # Create a mock sub-module
        def mock_sub_module(self, i, parameters, results=None, **kwargs):
            if results is None:
                results = {}
            results["test_value"] = "false_branch_executed"
            results["folder"] = kwargs.get("calling_module_dir", "")
            return parameters, results

        # Temporarily add the mock module
        self.ap.mock_sub_module = mock_sub_module.__get__(
            self.ap, analyse.AutoPicasso
        )

        # Test condition: 3 > 10 (False)
        parameters = {
            "condition": {"left": 3, "operator": ">", "right": 10},
            "if_true": [],
            "if_false": [("mock_sub_module", {})],
        }

        parameters, results = self.ap.conditional_branch(0, parameters)

        # Verify the condition was evaluated correctly
        assert results["condition_result"] is False
        assert results["branch_taken"] == "if_false"
        assert "if_branch" in results
        assert len(results["if_branch"]) == 1
        assert "00_mock_sub_module" in results["if_branch"]
        assert results["skipped_branch"] == "if_true"

        # Clean up
        shutil.rmtree(
            os.path.join(self.results_folder, "00_conditional_branch")
        )

    def test_10_conditional_branch_complex_condition(self):
        """Test conditional_branch with complex logical conditions"""
        # Set up mock locs data
        locs_dtype = [("frame", "u4"), ("x", "f4"), ("y", "f4")]
        self.ap.locs = pd.DataFrame(
            np.rec.array([(0, 1.0, 1.0)], dtype=locs_dtype)
        )

        # Test AND condition: (5 > 3) AND (10 < 20) = True
        parameters = {
            "condition": {
                "and": [
                    {"left": 5, "operator": ">", "right": 3},
                    {"left": 10, "operator": "<", "right": 20},
                ]
            },
            "if_true": [],
            "if_false": [],
        }

        parameters, results = self.ap.conditional_branch(0, parameters)
        assert results["condition_result"] is True

        # Clean up
        shutil.rmtree(
            os.path.join(self.results_folder, "00_conditional_branch")
        )

        # Test OR condition: (5 < 3) OR (10 < 20) = True
        parameters = {
            "condition": {
                "or": [
                    {"left": 5, "operator": "<", "right": 3},
                    {"left": 10, "operator": "<", "right": 20},
                ]
            },
            "if_true": [],
            "if_false": [],
        }

        parameters, results = self.ap.conditional_branch(1, parameters)
        assert results["condition_result"] is True

        # Clean up
        shutil.rmtree(
            os.path.join(self.results_folder, "01_conditional_branch")
        )

    def test_11_conditional_branch_equality_operators(self):
        """Test conditional_branch with equality operators"""
        # Set up minimal locs data
        self.ap.locs = pd.DataFrame({"frame": [0], "x": [1.0], "y": [1.0]})

        # Test equality
        parameters = {
            "condition": {"left": 5, "operator": "==", "right": 5},
            "if_true": [],
            "if_false": [],
        }
        parameters, results = self.ap.conditional_branch(0, parameters)
        assert results["condition_result"] is True
        shutil.rmtree(
            os.path.join(self.results_folder, "00_conditional_branch")
        )

        # Test inequality
        parameters = {
            "condition": {"left": 5, "operator": "!=", "right": 3},
            "if_true": [],
            "if_false": [],
        }
        parameters, results = self.ap.conditional_branch(1, parameters)
        assert results["condition_result"] is True
        shutil.rmtree(
            os.path.join(self.results_folder, "01_conditional_branch")
        )

        # Test >=
        parameters = {
            "condition": {"left": 5, "operator": ">=", "right": 5},
            "if_true": [],
            "if_false": [],
        }
        parameters, results = self.ap.conditional_branch(2, parameters)
        assert results["condition_result"] is True
        shutil.rmtree(
            os.path.join(self.results_folder, "02_conditional_branch")
        )

        # Test <=
        parameters = {
            "condition": {"left": 3, "operator": "<=", "right": 5},
            "if_true": [],
            "if_false": [],
        }
        parameters, results = self.ap.conditional_branch(3, parameters)
        assert results["condition_result"] is True
        shutil.rmtree(
            os.path.join(self.results_folder, "03_conditional_branch")
        )
