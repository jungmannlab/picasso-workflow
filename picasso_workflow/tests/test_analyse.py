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
import inspect
import numpy as np
from unittest.mock import patch, MagicMock

from picasso_workflow import analyse, util


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

    @patch("picasso_workflow.analyse.gausslq.locs_from_fits")
    @patch("picasso_workflow.analyse.gausslq.fit_spot")
    @patch("picasso_workflow.analyse.localize.get_spots")
    def localize(self, mock_get_spots, mock_fit_spot, mock_locs_from_fits):
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
        self.ap.info = []

        parameters = {"box_size": 7, "fit_parallel": False}

        parameters, results = self.ap.localize(0, parameters)

        shutil.rmtree(os.path.join(self.results_folder, "00_localize"))

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

    @patch("picasso_workflow.analyse.picasso_outpost.align_channels")
    def align_channels(self, mock_align_channels):
        mock_align_channels.return_value = [[3], [2]], np.zeros((3, 4, 5))
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
        locs2 = np.rec.array(
            [
                tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        self.ap.channel_locs = [locs1, locs2]
        self.ap.channel_info = [["info1"], ["info2"]]
        self.ap.channel_tags = ["1", "2"]
        self.ap.combine_channels(0, {})

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

    # @patch("picasso.postprocess.cluster_combine", MagicMock)
    # def aggregate_cluster(self):
    #     parameters = {}
    #     parameters, results = self.ap.aggregate_cluster(0, parameters)
    #     # logger.debug(f'parameters: {parameters}')
    #     logger.debug(f"results: {results}")
    #     assert results["duration"] > -1

    @patch(
        "picasso_workflow.analyse.postprocess.compute_local_density", MagicMock
    )
    def density(self):
        self.ap.info = []
        parameters = {"radius": 5}
        parameters, results = self.ap.density(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        logger.debug(f"results: {results}")
        assert results["duration"] > -1

    @patch("picasso_workflow.analyse.clusterer.find_cluster_centers")
    @patch("picasso_workflow.analyse.clusterer.dbscan")
    def dbscan(self, mock_dbscan, mock_fcc):
        self.ap.info = [{"Width": 1000, "Height": 1000}]
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
        mock_dbscan.return_value = self.ap.locs
        mock_fcc.return_value = self.ap.locs

        parameters = {"radius": 5, "min_density": 0.3}
        parameters, results = self.ap.dbscan(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        logger.debug(f"results: {results}")
        assert results["duration"] > -1

    @patch("picasso_workflow.analyse.clusterer.find_cluster_centers")
    @patch("picasso_workflow.analyse.clusterer.hdbscan")
    def hdbscan(self, mock_hdbscan, mock_fcc):
        self.ap.info = [{"Width": 1000, "Height": 1000}]
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
        mock_hdbscan.return_value = self.ap.locs
        mock_fcc.return_value = self.ap.locs

        parameters = {"min_cluster": 5, "min_samples": 3}
        parameters, results = self.ap.hdbscan(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        logger.debug(f"results: {results}")
        assert results["duration"] > -1

    @patch("picasso_workflow.analyse.clusterer.find_cluster_centers")
    @patch("picasso_workflow.analyse.clusterer.cluster")
    def smlm_clusterer(self, mock_clusterer, mock_fcc):
        self.ap.info = [{"Width": 1000, "Height": 1000}]
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
        mock_clusterer.return_value = self.ap.locs
        mock_fcc.return_value = self.ap.locs

        parameters = {"radius": 5, "min_locs": 10}
        parameters, results = self.ap.smlm_clusterer(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        logger.debug(f"results: {results}")
        assert results["duration"] > -1

    @patch("picasso_workflow.analyse.distance.cdist", MagicMock)
    def nneighbor(self):
        self.ap.info = []
        parameters = {}
        locs_dtype = [
            ("frame", "u4"),
            ("photons", "f4"),
            ("sx", "f4"),
            ("sy", "f4"),
            ("com_x", "f4"),
            ("com_y", "f4"),
        ]
        self.ap.locs = np.rec.array(
            [
                tuple([i] + list(np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        parameters, results = self.ap.nneighbor(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        logger.debug(f"results: {results}")
        assert results["duration"] > -1

    @patch("picasso_workflow.analyse.picasso_outpost.spinna_temp", MagicMock)
    def spinna_manual(self):
        info = [{"Width": 1000, "Height": 1000}]
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
        self.ap.channel_locs = [locs]
        self.ap.channel_info = [info]
        self.ap.channel_tags = ["CD86"]

        parameters = {}
        # test preparatory stage
        parameters, results = self.ap.spinna_manual(0, parameters)

        # test calling spinna
        parameters, results = self.ap.spinna_manual(0, parameters)


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
        self.results_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"
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
        # logger.debug(self.ap.locs)

        filepath = os.path.join(self.results_folder, "lvf.png")
        self.ap._plot_locs_vs_frame(filepath)

        os.remove(filepath)
