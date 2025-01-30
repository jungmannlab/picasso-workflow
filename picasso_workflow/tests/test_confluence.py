#!/usr/bin/env python
"""
Module Name: test_confluence.py
Author: Heinrich Grabmayr
Initial Date: March 8, 2024
Description: Test the module confluence.py
"""
import logging
import unittest
from unittest.mock import patch, MagicMock
import os
import inspect
import numpy as np

from picasso_workflow import confluence, util


logger = logging.getLogger(__name__)


class Test_A_ConfluenceInterface(unittest.TestCase):

    def setUp(self):
        self.confluence_url = os.getenv("TEST_CONFLUENCE_URL")
        self.confluence_token = os.getenv("TEST_CONFLUENCE_TOKEN")
        self.confluence_space = os.getenv("TEST_CONFLUENCE_SPACE")
        self.confluence_page = os.getenv("TEST_CONFLUENCE_PAGE")
        self.confluence_username = os.getenv("TEST_CONFLUENCE_USERNAME")
        self.testpgtitle = "mytestpage"
        self.bodytxt = "mybodytext"

    def tearDown(self):
        pass

    def instantiate_confluence_interface(self):
        return confluence.ConfluenceInterface(
            self.confluence_url,
            self.confluence_space,
            self.confluence_page,
            self.confluence_username,
            self.confluence_token,
        )

    # @unittest.skip("")
    def test_01_interface_01_all(self):
        logger.debug("testing all inferface")
        ci = self.instantiate_confluence_interface()
        pgid, pgtitle = ci.get_page_properties(self.confluence_page)
        assert pgtitle == self.confluence_page

        pgv = ci.get_page_version(self.confluence_page)
        logger.debug(f"page version: {pgv}")

        pgbdy = ci.create_page(self.testpgtitle, self.bodytxt)
        pgid, pgtitle = ci.get_page_properties(self.testpgtitle)

        pgbdy = ci.get_page_body(self.testpgtitle)
        assert pgbdy == self.bodytxt

        att_id = ci.upload_attachment(
            pgid,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "TestData",
                "confluence",
                "testimg.png",
            ),
        )
        ci.update_page_content_with_image_attachment(
            pgtitle, pgid, "testimg.png"
        )
        logger.debug(f"successfully uploaded attachment with id {att_id}")

        ci.update_page_content(pgtitle, pgid, "body update")

        att_id = ci.upload_attachment(
            pgid,
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "TestData",
                "confluence",
                "testmov.mp4",
            ),
        )
        ci.update_page_content_with_movie_attachment(
            pgtitle, pgid, "testmov.mp4"
        )
        logger.debug(f"successfully uploaded attachment with id {att_id}")

        ci = self.instantiate_confluence_interface()
        pgid, pgtitle = ci.get_page_properties(self.testpgtitle)
        logger.debug(f"Deleting page {pgid}, {pgtitle}")
        ci.delete_page(pgid)


# @unittest.skip('')
class Test_B_ConfluenceReporterModules(unittest.TestCase):
    """Tests the implementation of the analysis modules defined in
    util.AbstractModuleCollection
    """

    @patch("picasso_workflow.confluence.ConfluenceInterface")
    def setUp(self, mock_cfi):
        self.confluence_url = os.getenv("TEST_CONFLUENCE_URL")
        self.confluence_token = os.getenv("TEST_CONFLUENCE_TOKEN")
        self.confluence_space = os.getenv("TEST_CONFLUENCE_SPACE")
        self.confluence_page = os.getenv("TEST_CONFLUENCE_PAGE")
        self.confluence_username = os.getenv("TEST_CONFLUENCE_USERNAME")

        report_name = "my test report"

        # Mock the ConfluenceInterface to avoid Confluence interaction
        mock_instance = MagicMock()
        mock_instance.create_page.return_value = 534
        mock_instance.update_page_content.return_value = None
        mock_instance.get_page_properties.return_value = 123, "titleofhtepage"
        mock_cfi.return_value = mock_instance

        self.cr = confluence.ConfluenceReporter(
            self.confluence_url,
            self.confluence_space,
            self.confluence_page,
            report_name,
            self.confluence_username,
            self.confluence_token,
        )
        # self.cr.ci.upda

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
                    "'Test_B_ConfluenceReporterModules' object has "
                    + f"no attribute '{module}'"
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

    # @unittest.skip("")
    def load_dataset_movie(self):

        pars_load = {
            "filename": "my test file location",
            "save_directory": "my test save directory",
        }
        results_load = {
            "start time": "now",
            "picasso version": "0.0.0",
            "movie.shape": (40000, 2048, 1024),
            "duration": 10.2,
            "sample_movie": {
                "sample_frame_idx": [1, 6, 11],
                "filename": os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "TestData",
                    "confluence",
                    "testmov.mp4",
                ),
            },
        }
        self.cr.load_dataset_movie(0, pars_load, results_load)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def identify(self):
        parameters = {
            "min_gradient": 10000,
            "box_size": 7,
        }
        results = {
            "start time": "now",
            "duration": 16.4,
            "num_identifications": 23237,
        }
        self.cr.identify(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def localize(self):
        parameters = {}
        results = {
            "start time": "now",
            "duration": 16.4,
            "locs_columns": ("x", "y", "photons"),
        }
        self.cr.localize(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def export_brightfield(self):
        parameters = {}
        results = {
            "start time": "now",
            "duration": 16.4,
            "filepaths": {"GFP": "myfp.png"},
        }
        self.cr.export_brightfield(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def undrift_rcc(self):
        parameters = {
            "dimensions": ["x", "y"],
            "segmentation": 1000,
        }
        results = {
            "start time": "now",
            "message": "This is my message to you.",
            "duration": 27.4,
        }
        self.cr.undrift_rcc(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def undrift_aim(self):
        parameters = {
            "dimensions": ["x", "y"],
            "segmentation": 1000,
            "intersect_d": 20,
            "roi_r": 60,
        }
        results = {
            "start time": "now",
            "duration": 27.4,
        }
        self.cr.undrift_rcc(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def manual(self):
        parameters = {
            "prompt": "Do something.",
            "filename": "abc.png",
            "success": False,
        }
        results = {
            "start time": "now",
            "message": "This is my message to you.",
        }
        self.cr.manual(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def summarize_dataset(self):
        parameters = {"methods": {"nena": {"inputpar": "a"}}}
        results = {"nena": {"best_vals": (3, 5, 7), "res": 1.23}}
        self.cr.summarize_dataset(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def save_single_dataset(self):
        parameters = {}
        results = {
            "start time": "now",
            "filepath": "/path/to/my/file",
            "duration": 1,
        }
        self.cr.save_single_dataset(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def load_datasets_to_aggregate(self):
        parameters = {}
        results = {
            "start time": "now",
            "filepaths": ["/path/to/my/file", "/and/the/other"],
            "tags": ["a", "b"],
            "duration": 1.3,
        }
        self.cr.load_datasets_to_aggregate(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def align_channels(self):
        parameters = {}
        results = {
            "start time": "now",
            "shifts": np.array([[3, 4], [2, 3], [1, 2]]),
            "duration": 1.3,
        }
        self.cr.align_channels(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def combine_channels(self):
        parameters = {}
        results = {
            "start time": "now",
            "duration": 1.3,
            "combine_map": "placeholderforcombinemap",
        }
        self.cr.combine_channels(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def convert_zeiss_movie(self):
        parameters = {"filepath": "myfile.czi"}
        results = {
            "start time": "now",
            "duration": 4.12,
            "filepath_raw": "myfile.raw",
        }
        self.cr.convert_zeiss_movie(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def dbscan(self):
        parameters = {"filepath": "myfile.czi"}
        results = {
            "start time": "now",
            "duration": 4.12,
            "radius": 5,
            "min_samples": 3,
        }
        self.cr.dbscan(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def density(self):
        parameters = {"filepath": "myfile.czi"}
        results = {
            "start time": "now",
            "duration": 4.12,
            "radius": 5,
        }
        self.cr.density(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def hdbscan(self):
        parameters = {"filepath": "myfile.czi"}
        results = {
            "start time": "now",
            "duration": 4.12,
            "min_cluster": 7,
            "min_sample": 5,
        }
        self.cr.hdbscan(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def load_dataset_localizations(self):
        parameters = {"filename": "myfile.czi"}
        results = {
            "start time": "now",
            "duration": 4.12,
            "picasso version": "0.1.2",
            "nlocs": 12345,
        }
        self.cr.load_dataset_localizations(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def nneighbor(self):
        parameters = {
            "dims": ["x", "y"],
            "subsample_1stNN": 20,
            "nth_NN": 4,
            "nth_rdf": 10,
            "filepath": "myfile.czi",
        }
        results = {
            "start time": "now",
            "duration": 4.12,
            "nneighbors": "/path/to/file",
            "density_rdf": 43e-6,
        }
        self.cr.nneighbor(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def fit_csr(self):
        parameters = {"nneighbors": np.zeros((9, 4)), "dimensionality": 2}
        results = {
            "start time": "now",
            "duration": 4.12,
            "density": 0.52,
        }
        self.cr.fit_csr(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def save_datasets_aggregated(self):
        parameters = {"filepath": "myfile.czi"}
        results = {
            "start time": "now",
            "duration": 4.12,
            "filepaths": ["a.raw", "b.raw"],
        }
        self.cr.save_datasets_aggregated(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def smlm_clusterer(self):
        parameters = {"filepath": "myfile.czi"}
        results = {
            "start time": "now",
            "duration": 4.12,
            "radius": 8,
            "min_locs": 3,
            "basic_fa": False,
            "radius_z": 2,
        }
        self.cr.smlm_clusterer(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def gaussian_mixture_cluster(self):
        parameters = {
            "min_locs": 3,
            "min_sigma": 0.4,
            "max_sigma": 1.1,
        }
        results = {
            "start time": "now",
            "duration": 4.12,
            "n_locs_in": 2000000,
            "n_locs_clustered": 1800000,
            "n_centers": 100000,
        }
        self.cr.gaussian_mixture_cluster(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def spinna_manual(self):
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "message": "This is my message to you.",
            "success": False,
        }
        self.cr.spinna_manual(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def spinna(self):
        parameters = {
            "labeling_efficiency": {"A": 0.34, "B": 0.56},
            "labeling_uncertainty": {"A": 5, "B": 5},
            "n_simulate": 5000,
            "fp_mask_dict": None,
            "density": [8e-5],
            "height": 256,
            "depth": 4,
            "random_rot_mode": "3D",
            "n_nearest_neighbors": 4,
            "sim_repeats": 50,
            "fit_NND_bin": 0.5,
            "fit_NND_maxdist": 30,
            "res_factor": 10,
        }
        results = {
            "start time": "now",
            "duration": 4.12,
        }
        self.cr.spinna(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def ripleysk(self):
        parameters = {
            "ripleys_threshold": 1.2,
            "atype": "Ripleys",
        }
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
            "ripleys_significant": [("a", "b")],
            "fp_ripleys_meanval": "bklab",
        }
        self.cr.ripleysk(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def ripleysk2(self):
        parameters = {
            "ripleys_threshold": 1.2,
            "atype": "Ripleys",
        }
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
            "ripleys_significant": [("a", "b")],
            "fp_ripleys_meanval": "bklab",
        }
        self.cr.ripleysk2(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def ripleysk_average(self):
        parameters = {
            "ripleys_threshold": 1.2,
            "report_names": ["a", "b", "c"],
            "fp_workflows": ["/a", "/b", "/c"],
        }
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
            "output_folders": ["/d"],
            "fp_ripleys_significant": "/e",
            "ripleys_significant": [("a", "b")],
        }
        self.cr.ripleysk_average(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def protein_interactions(self):
        parameters = {"interaction_pairs": [("a", "b")]}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
        }
        self.cr.protein_interactions(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def create_mask(self):
        """Create a density mask"""
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
        }
        self.cr.create_mask(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def create_mask2(self):
        """Create a density mask"""
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
            "area": 43,
        }
        self.cr.create_mask(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def dbscan_molint(self):
        """TO BE CLEANED UP
        dbscan implementation for molecular interactions workflow
        """
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
        }
        self.cr.dbscan_molint(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def CSR_sim_in_mask(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask
        """
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
        }
        self.cr.CSR_sim_in_mask(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def analysis_documentation(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask
        """
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
        }
        self.cr.analysis_documentation(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def dummy_module(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask
        """
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
        }
        self.cr.dummy_module(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def find_cluster_motifs(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask
        """
        parameters = {
            "population_threshold": 0.01,
            "cellfraction_threshold": 0.4,
            "ttest_pvalue_max": 0.05,
        }
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
            "significant_barcodes": ["10", "11"],
        }
        self.cr.find_cluster_motifs(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def interaction_graph(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask
        """
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
        }
        self.cr.interaction_graph(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def plot_densities(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask
        """
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
        }
        self.cr.plot_densities(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def protein_interactions_average(self):
        """TO BE CLEANED UP
        simulate CSR within a density mask
        """
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
        }
        self.cr.protein_interactions_average(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def find_gold(self):
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
            "n_gold": 3,
            "fp_gold": "path/to/gold",
            "fp_nogold": "path/to/no/gold",
        }
        self.cr.find_gold(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def undrift_from_picked(self):
        parameters = {
            "fp_picked_locs": "path/to/gold",
        }
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
            "fp_locs": "path/to/locs",
        }
        self.cr.undrift_from_picked(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def filter_locs(self):
        parameters = {"field": "photons", "minval": 800, "maxval": 1200}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
            "fp_locs": "path/to/locs",
            "nlocs_before": 2000,
            "nlocs_after": 1700,
        }
        self.cr.filter_locs(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def link_locs(self):
        parameters = {"d_max": 2, "tolerance": 3}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
            "fp_locs": "path/to/locs",
        }
        self.cr.link_locs(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def labeling_efficiency_analysis(self):
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
            "labeling_efficiency": {"ref": 0.57, "tgt": 0.23},
        }
        self.cr.labeling_efficiency_analysis(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)


# @unittest.skip('')
class Test_B_ConfluenceReporter(unittest.TestCase):

    @patch("picasso_workflow.confluence.ConfluenceInterface")
    def setUp(self, mock_cfi):
        self.confluence_url = os.getenv("TEST_CONFLUENCE_URL")
        self.confluence_token = os.getenv("TEST_CONFLUENCE_TOKEN")
        self.confluence_space = os.getenv("TEST_CONFLUENCE_SPACE")
        self.confluence_page = os.getenv("TEST_CONFLUENCE_PAGE")
        self.confluence_username = os.getenv("TEST_CONFLUENCE_USERNAME")

        report_name = "my test report"

        # Mock the ConfluenceInterface to avoid Confluence interaction
        mock_instance = MagicMock()
        mock_instance.create_page.return_value = 534
        mock_instance.update_page_content.return_value = None
        mock_instance.get_page_properties.return_value = 123, "titleofhtepage"
        mock_cfi.return_value = mock_instance

        self.cr = confluence.ConfluenceReporter(
            self.confluence_url,
            self.confluence_space,
            self.confluence_page,
            report_name,
            self.confluence_username,
            self.confluence_token,
        )
        # self.cr.ci.upda

    def tearDown(self):
        pass

    def test_00(self):
        pass


# @unittest.skip('')
class Test_C_ConfluenceReporter(Test_B_ConfluenceReporter):
    """This time really use the ConfluenceInterface, no mocking."""

    def setUp(self):
        self.confluence_url = os.getenv("TEST_CONFLUENCE_URL")
        self.confluence_token = os.getenv("TEST_CONFLUENCE_TOKEN")
        self.confluence_space = os.getenv("TEST_CONFLUENCE_SPACE")
        self.confluence_page = os.getenv("TEST_CONFLUENCE_PAGE")
        self.confluence_username = os.getenv("TEST_CONFLUENCE_USERNAME")

        report_name = "my test report"

        self.cr = confluence.ConfluenceReporter(
            self.confluence_url,
            self.confluence_space,
            self.confluence_page,
            report_name,
            self.confluence_username,
            self.confluence_token,
        )

        pgid, pgtitle = self.cr.ci.get_page_properties(report_name)
        self.cr.ci.delete_page(pgid)
