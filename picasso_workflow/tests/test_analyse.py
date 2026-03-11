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
import pandas as pd
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
        mock_locs_from_fits.return_value = pd.DataFrame(
            mock_locs_from_fits.return_value
        )
        self.ap.info = []

        parameters = {"box_size": 7, "fit_parallel": False}

        parameters, results = self.ap.localize(0, parameters)

        shutil.rmtree(os.path.join(self.results_folder, "00_localize"))

    def load_picassoconfig(self):
        pass

    def zfit(self):
        pass

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
        "picasso_workflow.analyse.postprocess.compute_local_density", MagicMock
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
        mock_dbscan.return_value = self.ap.locs
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
        mock_hdbscan.return_value = self.ap.locs
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
        mock_clusterer.return_value = self.ap.locs
        mock_fcc.return_value = self.ap.locs

        parameters = {"radius": 5, "min_locs": 10}
        parameters, results = self.ap.smlm_clusterer(0, parameters)
        # logger.debug(f'parameters: {parameters}')
        logger.debug(f"results: {results}")
        assert results["duration"] > -1

        shutil.rmtree(os.path.join(self.results_folder, "00_smlm_clusterer"))

    @patch("picasso_workflow.analyse.lib.plot_subclustering_check")
    @patch("picasso_workflow.analyse.clusterer.test_subclustering")
    @patch("picasso_workflow.analyse.g5m.g5m")
    def gaussian_mixture_cluster(self, mock_gmms, mock_test_subclustering, mock_plot):
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

    @patch("picasso_workflow.analyse.picasso_outpost.single_spinna_run")
    def spinna(self, mock_sptmp):
        mock_sptmp.return_value = (0, 1)
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

    @patch("picasso_workflow.analyse.io.save_locs", MagicMock)
    @patch("picasso_workflow.analyse.postprocess.link", MagicMock)
    def link_locs(self):
        parameters = {"d_max": 2, "tolerance": 3}

        parameters, results = self.ap.link_locs(0, parameters)

        shutil.rmtree(os.path.join(self.results_folder, "00_link_locs"))

    @patch("picasso_workflow.analyse.picasso_outpost.single_spinna_run")
    def labeling_efficiency_analysis(self, mock_spinna_sgl):
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
        spinna_result = {
            "Fitted proportions of structures": np.array([0.4, 0.15, 0.35]),
            "props": np.array([40.0, 15.0, 35.0]),
            "props_std": np.array([2.0, 1.0, 3.0]),
        }
        mock_spinna_sgl.return_value = (
            spinna_result,
            [
                "/path/to/figAA.png",
                "/path/to/figAB.png",
                "/path/to/figBA.png",
                "/path/to/figBB.png",
            ],
        )
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
        """Is tested separately in tests/outpost_modules/test_resolution_frc.py
        """
        pass

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
            for pattern in ["drift_*.png", "convergence_*.png", "robustness_*.png"]:
                for file in glob.glob(os.path.join(undrift_folder, pattern)):
                    try:
                        os.remove(file)
                    except:
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
