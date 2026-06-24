#!/usr/bin/env python
"""
Module Name: test_confluence.py
Author: Heinrich Grabmayr
Initial Date: March 8, 2024
Description: Test the module confluence.py
"""

import logging
import os
import unittest
from unittest.mock import patch, MagicMock
import inspect
import numpy as np
import pytest

from picasso_workflow import confluence, util
from picasso_workflow.confluence import resolve_confluence_credentials

logger = logging.getLogger(__name__)

# Live-Confluence tests need the test token; non-secret fields come from the
# ConfluenceTest config section (resolver). Without a token they are skipped.
_TEST_CREDS = resolve_confluence_credentials("ConfluenceTest")
_CONFLUENCE_AVAILABLE = bool(_TEST_CREDS.get("token"))


@pytest.mark.skipif(
    not _CONFLUENCE_AVAILABLE,
    reason="Confluence test token not set (TEST_CONFLUENCE_TOKEN)",
)
class Test_A_ConfluenceInterface(unittest.TestCase):

    def setUp(self):
        self.confluence_url = _TEST_CREDS["base_url"]
        self.confluence_token = _TEST_CREDS["token"]
        self.confluence_space = _TEST_CREDS["space_key"]
        self.confluence_page = _TEST_CREDS["parent_page_title"]
        self.confluence_username = _TEST_CREDS["username"]
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
        self.confluence_url = _TEST_CREDS["base_url"]
        self.confluence_token = _TEST_CREDS["token"]
        self.confluence_space = _TEST_CREDS["space_key"]
        self.confluence_page = _TEST_CREDS["parent_page_title"]
        self.confluence_username = _TEST_CREDS["username"]

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

    def load_picassoconfig(self):
        pass

    def zfit(self):
        pass

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
        results = {
            "nena": {"best_vals": (3, 5, 7), "res": 1.23, "chisqr": 3.2}
        }
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
        parameters = {
            "filepath": "myfile.czi",
            "radius": 5,
            "min_samples": 3,
        }
        results = {
            "start time": "now",
            "duration": 4.12,
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

    def binding_event_analysis(self):
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
        }
        self.cr.binding_event_analysis(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    def smlm_clusterer(self):
        parameters = {
            "filepath": "myfile.czi",
            "radius": 8,
        }
        results = {
            "start time": "now",
            "duration": 4.12,
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

    # # @unittest.skip("")
    # def spinna_manual(self):
    #     parameters = {}
    #     results = {
    #         "start time": "now",
    #         "duration": 4.12,
    #         "message": "This is my message to you.",
    #         "success": False,
    #     }
    #     self.cr.spinna_manual(0, parameters, results)

    #     # clean up
    #     pgid, pgtitle = self.cr.ci.get_page_properties(
    #         self.cr.report_page_name
    #     )
    #     self.cr.ci.delete_page(pgid)

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
    def spinna_batch(self):
        parameters = {
            "fp_spinna_batch_config": "spinna_batch_config.csv",
        }
        result_dir = "spinna_batch_config_fitting_results"
        # with NND figures: exercises the figure-table branch
        results = {
            "start time": "now",
            "duration": 124.5,
            "message": "Successfully performed SPINNA analysis.",
            "result_dir": result_dir,
            "fp_summary": os.path.join(result_dir, "summary_results.csv"),
            "fp_figs": [
                os.path.join(result_dir, "run1_NND_A_A.png"),
                os.path.join(result_dir, "run1_NND_A_B.png"),
            ],
            "success": True,
        }
        self.cr.spinna_batch(0, parameters, results)

        # without NND figures: exercises the empty branch
        results_no_figs = {
            "start time": "now",
            "duration": 4.12,
        }
        self.cr.spinna_batch(1, parameters, results_no_figs)

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
            "metric": "RK",
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
    def ripleysk_average2(self):
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
        self.cr.ripleysk_average2(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def random_val(self):
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
        }
        self.cr.random_val(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def render(self):
        parameters = {}
        import tempfile
        import shutil
        import os

        temp_dir = tempfile.mkdtemp()
        try:
            fp_fullfov = os.path.join(temp_dir, "fullfov.png")
            with open(fp_fullfov, "wb") as f:
                f.write(b"PNG mock data")

            fp_ctrmass = os.path.join(temp_dir, "ctrmass.png")
            with open(fp_ctrmass, "wb") as f:
                f.write(b"PNG mock data")

            fp_scene_rois = []
            for i in range(5):
                fp_roi = os.path.join(temp_dir, f"roi_{i+1}.png")
                with open(fp_roi, "wb") as f:
                    f.write(b"PNG mock data")
                fp_scene_rois.append(fp_roi)

            results = {
                "start time": "now",
                "duration": 4.12,
                "success": True,
                "fp_scene_fullfov": fp_fullfov,
                "fp_scene_ctrmass": fp_ctrmass,
                "fp_scene_rois": fp_scene_rois,
            }
            self.cr.render(0, parameters, results)
        finally:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def find_structures(self):
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
            "n_clusters": 2,
            "n_picks": [412, 501],
        }
        self.cr.find_structures(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def pairwise_module_executor(self):
        parameters = {"module_name": "mymodule"}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
        }
        self.cr.pairwise_module_executor(0, parameters, results)

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
    def refine_mask_by_density(self):
        """Create a density mask"""
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
            "area_um^2": 421,
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
    def filter_transient_binding(self):
        parameters = {"field": "photons", "minval": 800, "maxval": 1200}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
            "fields_filtered": ["frame", "std_frame"],
            "nlocs_before": 4000,
            "nlocs_after": 3000,
            "fp_locs": "path/to/locs",
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
            "labeling_efficiency_std": {"ref": 0.03, "tgt": 0.01},
        }
        self.cr.labeling_efficiency_analysis(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def find_similar(self):
        parameters = {
            "diameter": 5.0,
            "min_n_locs_per_frame": 0.01,
            "max_n_locs_per_frame": 0.1,
            "min_rmsd": 1.0,
            "max_rmsd": 3.0,
            "n_plot_structures": 2,
            "display_pixelsize": 1.0,
        }
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
            "n_picks": 5,
            "n_picked_locs": 150,
            "n_locs": 2000,
            "fp_phasespace": os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "TestData",
                "confluence",
                "testimg.png",
            ),
            "fp_phasespace_hexbin": os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "TestData",
                "confluence",
                "testimg.png",
            ),
            "fp_picked_fullfov": os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "TestData",
                "confluence",
                "testimg.png",
            ),
            "fp_renderings": [
                [
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "TestData",
                        "confluence",
                        "testimg.png",
                    ),
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "TestData",
                        "confluence",
                        "testimg.png",
                    ),
                ]
            ],
        }
        self.cr.find_similar(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def conditional_branch(self):
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
        }
        self.cr.conditional_branch(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def resolution_analysis(self):
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
        }
        self.cr.resolution_analysis(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def resolution_frc_spatial(self):
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
        }
        self.cr.resolution_frc_spatial(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)

    # @unittest.skip("")
    def undrift_rsso(self):
        parameters = {}
        results = {
            "start time": "now",
            "duration": 4.12,
            "success": True,
        }
        self.cr.undrift_rsso(0, parameters, results)

        # clean up
        pgid, pgtitle = self.cr.ci.get_page_properties(
            self.cr.report_page_name
        )
        self.cr.ci.delete_page(pgid)


# @unittest.skip('')
class Test_B_ConfluenceReporter(unittest.TestCase):

    @patch("picasso_workflow.confluence.ConfluenceInterface")
    def setUp(self, mock_cfi):
        self.confluence_url = _TEST_CREDS["base_url"]
        self.confluence_token = _TEST_CREDS["token"]
        self.confluence_space = _TEST_CREDS["space_key"]
        self.confluence_page = _TEST_CREDS["parent_page_title"]
        self.confluence_username = _TEST_CREDS["username"]

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
@pytest.mark.skipif(
    not _CONFLUENCE_AVAILABLE,
    reason="Confluence test token not set (TEST_CONFLUENCE_TOKEN)",
)
class Test_C_ConfluenceReporter(Test_B_ConfluenceReporter):
    """This time really use the ConfluenceInterface, no mocking."""

    def setUp(self):
        self.confluence_url = _TEST_CREDS["base_url"]
        self.confluence_token = _TEST_CREDS["token"]
        self.confluence_space = _TEST_CREDS["space_key"]
        self.confluence_page = _TEST_CREDS["parent_page_title"]
        self.confluence_username = _TEST_CREDS["username"]

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


# ---------------------------------------------------------------------------
# Optimistic-locking (StaleState) retry handling -- no network needed.
# ---------------------------------------------------------------------------


def test_is_stale_state_conflict_detection():
    """Recognize StaleState/Conflict text and HTTP 409, reject the rest."""
    from requests.exceptions import HTTPError

    stale = HTTPError(
        "ConflictException ... StaleStateException: Batch update returned "
        "unexpected row count from update [0]"
    )
    assert confluence._is_stale_state_conflict(stale)

    resp = MagicMock(status_code=409)
    assert confluence._is_stale_state_conflict(HTTPError("x", response=resp))

    assert not confluence._is_stale_state_conflict(HTTPError("plain 500"))


def test_confluence_call_retries_on_stale_state(monkeypatch):
    """A version conflict is retried, then succeeds."""
    from requests.exceptions import HTTPError

    monkeypatch.setattr(confluence, "_STALE_STATE_BACKOFF", 0.0)
    monkeypatch.setattr(confluence.time, "sleep", lambda *_a, **_k: None)
    stale = HTTPError("StaleStateException: row count [0]")

    class Dummy:
        def __init__(self):
            self.calls = 0

        def connect(self):
            pass

        @confluence.confluence_call
        def write(self):
            self.calls += 1
            if self.calls < 3:
                raise stale
            return "ok"

    d = Dummy()
    assert d.write() == "ok"
    assert d.calls == 3


def test_confluence_call_raises_after_exhausting_retries(monkeypatch):
    """Persistent conflicts raise ConfluenceInterfaceError after retries."""
    from requests.exceptions import HTTPError

    monkeypatch.setattr(confluence, "_STALE_STATE_BACKOFF", 0.0)
    monkeypatch.setattr(confluence.time, "sleep", lambda *_a, **_k: None)
    stale = HTTPError("StaleStateException: row count [0]")

    class Dummy:
        def __init__(self):
            self.calls = 0

        def connect(self):
            pass

        @confluence.confluence_call
        def write(self):
            self.calls += 1
            raise stale

    d = Dummy()
    with pytest.raises(confluence.ConfluenceInterfaceError):
        d.write()
    assert d.calls == confluence._STALE_STATE_RETRIES + 1


def test_is_transient_error_detection():
    """Recognize HTTP 429 and 5xx as transient, reject 4xx and 409."""
    from requests.exceptions import HTTPError

    for code in (429, 500, 502, 503):
        resp = MagicMock(status_code=code)
        assert confluence._is_transient_error(HTTPError("x", response=resp))

    for code in (400, 401, 404, 409):
        resp = MagicMock(status_code=code)
        assert not confluence._is_transient_error(
            HTTPError("x", response=resp)
        )

    # No response object: fall back to message text.
    assert confluence._is_transient_error(HTTPError("Too Many Requests"))
    assert not confluence._is_transient_error(HTTPError("unauthorized"))


def test_confluence_call_retries_on_transient_error(monkeypatch):
    """A transient 5xx is retried, then succeeds."""
    from requests.exceptions import HTTPError

    monkeypatch.setattr(confluence, "_TRANSIENT_BACKOFF", 0.0)
    monkeypatch.setattr(confluence.time, "sleep", lambda *_a, **_k: None)
    transient = HTTPError("x", response=MagicMock(status_code=503))

    class Dummy:
        def __init__(self):
            self.calls = 0

        def connect(self):
            pass

        @confluence.confluence_call
        def write(self):
            self.calls += 1
            if self.calls < 3:
                raise transient
            return "ok"

    d = Dummy()
    assert d.write() == "ok"
    assert d.calls == 3


def test_confluence_call_raises_after_exhausting_transient_retries(
    monkeypatch,
):
    """Persistent transient errors raise after the retry budget is spent."""
    from requests.exceptions import HTTPError

    monkeypatch.setattr(confluence, "_TRANSIENT_BACKOFF", 0.0)
    monkeypatch.setattr(confluence.time, "sleep", lambda *_a, **_k: None)
    transient = HTTPError("x", response=MagicMock(status_code=500))

    class Dummy:
        def __init__(self):
            self.calls = 0

        def connect(self):
            pass

        @confluence.confluence_call
        def write(self):
            self.calls += 1
            raise transient

    d = Dummy()
    with pytest.raises(confluence.ConfluenceInterfaceError):
        d.write()
    assert d.calls == confluence._TRANSIENT_RETRIES + 1


def test_confluence_call_retries_on_connection_error(monkeypatch):
    """A dropped connection is reconnected and retried with backoff."""
    from requests.exceptions import ConnectionError as ReqConnectionError

    monkeypatch.setattr(confluence, "_CONNECTION_BACKOFF", 0.0)
    monkeypatch.setattr(confluence.time, "sleep", lambda *_a, **_k: None)

    class Dummy:
        def __init__(self):
            self.calls = 0
            self.connects = 0

        def connect(self):
            self.connects += 1

        @confluence.confluence_call
        def write(self):
            self.calls += 1
            if self.calls < 3:
                raise ReqConnectionError("Connection aborted.")
            return "ok"

    d = Dummy()
    assert d.write() == "ok"
    assert d.calls == 3
    assert d.connects == 2  # reconnected before each retry


def test_confluence_call_connection_retries_exhausted(monkeypatch):
    """Persistent connection drops raise after the reconnect budget."""
    from requests.exceptions import ConnectionError as ReqConnectionError

    monkeypatch.setattr(confluence, "_CONNECTION_BACKOFF", 0.0)
    monkeypatch.setattr(confluence.time, "sleep", lambda *_a, **_k: None)

    class Dummy:
        def __init__(self):
            self.calls = 0

        def connect(self):
            pass

        @confluence.confluence_call
        def write(self):
            self.calls += 1
            raise ReqConnectionError("Connection aborted.")

    d = Dummy()
    with pytest.raises(confluence.ConfluenceInterfaceError):
        d.write()
    assert d.calls == confluence._CONNECTION_RETRIES + 1


# ---------------------------------------------------------------------------
# Parent-page id passthrough -- skip the eventually-consistent title lookup.
# ---------------------------------------------------------------------------


def test_confluence_interface_uses_given_parent_page_id(monkeypatch):
    """A supplied parent_page_id skips the title-based parent lookup."""
    calls = {"lookup": 0}

    def fake_lookup(self, page_title="", page_id=""):
        calls["lookup"] += 1
        return "should-not-be-used", page_title

    monkeypatch.setattr(
        confluence.ConfluenceInterface, "connect", lambda self: None
    )
    monkeypatch.setattr(
        confluence.ConfluenceInterface, "get_page_properties", fake_lookup
    )

    ci = confluence.ConfluenceInterface(
        "http://example/wiki",
        "SPC",
        "ParentTitle",
        token="tok",
        parent_page_id="42",
    )
    assert ci.parent_page_id == "42"
    assert calls["lookup"] == 0


def test_confluence_interface_falls_back_to_title_lookup(monkeypatch):
    """Without parent_page_id, the parent is resolved by title."""
    calls = {"lookup": 0}

    def fake_lookup(self, page_title="", page_id=""):
        calls["lookup"] += 1
        return "99", page_title

    monkeypatch.setattr(
        confluence.ConfluenceInterface, "connect", lambda self: None
    )
    monkeypatch.setattr(
        confluence.ConfluenceInterface, "get_page_properties", fake_lookup
    )

    ci = confluence.ConfluenceInterface(
        "http://example/wiki",
        "SPC",
        "ParentTitle",
        token="tok",
    )
    assert ci.parent_page_id == "99"
    assert calls["lookup"] == 1


# ---------------------------------------------------------------------------
# Eventual-consistency page-lookup retry -- no network needed.
# ---------------------------------------------------------------------------


def test_get_page_properties_retries_until_page_appears(monkeypatch):
    """A newly created page found only after a few lookups still resolves."""
    monkeypatch.setattr(confluence, "_PAGE_LOOKUP_BACKOFF", 0.0)
    monkeypatch.setattr(confluence.time, "sleep", lambda *_a, **_k: None)

    calls = {"n": 0}

    def fake_get_page_by_title(space, title):
        calls["n"] += 1
        if calls["n"] < 3:
            return None
        return {"id": "123", "title": title}

    ci = MagicMock()
    ci.space_key = "SPC"
    ci.base_url = "http://example/wiki"
    ci.confluence.get_page_by_title.side_effect = fake_get_page_by_title

    result = confluence.ConfluenceInterface.get_page_properties(
        ci, page_title="newpage"
    )
    assert result == ("123", "newpage")
    assert calls["n"] == 3


def test_get_page_properties_raises_clear_error_when_absent(monkeypatch):
    """A page that never appears raises ConfluenceInterfaceError, not None."""
    monkeypatch.setattr(confluence, "_PAGE_LOOKUP_BACKOFF", 0.0)
    monkeypatch.setattr(confluence.time, "sleep", lambda *_a, **_k: None)

    ci = MagicMock()
    ci.space_key = "SPC"
    ci.base_url = "http://example/wiki"
    ci.confluence.get_page_by_title.return_value = None

    with pytest.raises(confluence.ConfluenceInterfaceError):
        confluence.ConfluenceInterface.get_page_properties(
            ci, page_title="missing"
        )
    assert (
        ci.confluence.get_page_by_title.call_count
        == confluence._PAGE_LOOKUP_RETRIES + 1
    )
