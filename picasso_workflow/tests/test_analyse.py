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

    @patch("picasso_workflow.analyse.aim.aim")
    def undrift_aim(self, mock_undrift_aim):
        nspots = 5
        mock_undrift_aim.return_value = (
            np.random.rand(2, len(self.ap.movie)),
            [{"name": "info"}],
            np.rec.array(
                [
                    tuple(np.random.rand(len(self.locs_dtype)))
                    for i in range(nspots)
                ],
                dtype=self.locs_dtype,
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

        shutil.rmtree(os.path.join(self.results_folder, "00_dbscan"))

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

        shutil.rmtree(os.path.join(self.results_folder, "00_hdbscan"))

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

        shutil.rmtree(os.path.join(self.results_folder, "00_smlm_clusterer"))

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

    @patch("picasso_workflow.analyse.picasso_outpost.spinna_sgl_temp")
    def spinna(self, mock_sptmp):
        mock_sptmp.return_value = (0, 1)
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
            "res_factor": 10,
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

    def ripleysk(self):
        return
        shutil.rmtree(os.path.join(self.results_folder, "00_ripleysk"))

    def ripleysk_average(self):
        return
        shutil.rmtree(os.path.join(self.results_folder, "00_ripleysk"))

    def protein_interactions(self):
        return
        shutil.rmtree(
            os.path.join(self.results_folder, "00_protein_interactions")
        )

    def create_mask(self):
        """Create a density mask"""
        return
        shutil.rmtree(os.path.join(self.results_folder, "00_create_mask"))

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

        parameters, results = self.ap.filter_locs(0, parameters)

        shutil.rmtree(os.path.join(self.results_folder, "00_filter_locs"))

    @patch("picasso_workflow.analyse.io.save_locs", MagicMock)
    @patch("picasso_workflow.analyse.postprocess.link", MagicMock)
    def link_locs(self):
        parameters = {"d_max": 2, "tolerance": 3}

        parameters, results = self.ap.link_locs(0, parameters)

        shutil.rmtree(os.path.join(self.results_folder, "00_link_locs"))

    @patch("picasso_workflow.analyse.picasso_outpost.spinna_sgl_temp")
    def labeling_efficiency_analysis(self, mock_spinna_sgl):
        parameters = {
            "target_name": "CD86",
            "reference_name": "GFP",
            "pair_distance": 10,
            "density": {"CD86": 92.4, "GFP": 83.5},
            "n_simulate": 10000,
            "res_factor": 5,
            "labeling_uncertainty": 5,
            "sim_repeats": 2,
            # "nn_nth": 2,
        }
        spinna_result = {
            "Fitted proportions of structures": np.array([0.4, 0.15, 0.35])
        }
        mock_spinna_sgl.return_value = (spinna_result, "/path/to/fig")
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
        locs_b = np.rec.array(
            [
                tuple([i] + list(1000 * np.random.rand(len(locs_dtype) - 1)))
                for i in range(len(self.ap.movie))
            ],
            dtype=locs_dtype,
        )
        self.ap.channel_locs = [locs_a, locs_b]

        parameters, results = self.ap.labeling_efficiency_analysis(
            0, parameters
        )

        shutil.rmtree(
            os.path.join(
                self.results_folder, "00_labeling_efficiency_analysis"
            )
        )


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
        # logger.debug(self.ap.locs)

        filepath = os.path.join(self.results_folder, "lvf.png")
        self.ap._plot_locs_vs_frame(filepath)

        os.remove(filepath)
