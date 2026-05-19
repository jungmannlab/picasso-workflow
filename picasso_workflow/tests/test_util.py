#!/usr/bin/env python
"""
Module Name: test_util.py
Author: Heinrich Grabmayr
Initial Date: March 14, 2024
Description: Test the module util.py
"""
import logging
import unittest
from unittest.mock import patch

from picasso_workflow import util


logger = logging.getLogger(__name__)


# @unittest.skip('')
class TestUtil(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # @unittest.skip('')
    def test_01_correct_path_separators(self):
        test_path = "\\this\\is/my/test/path"
        out_path = util.correct_path_separators(test_path)
        logger.debug(f"converted {test_path} to {out_path}")

    def test_02_get_caller_name(self):
        caller_name = util.get_caller_name(1)
        assert caller_name == "test_02_get_caller_name"
        # logger.debug(util.get_caller_name(1))
        # logger.debug(util.get_caller_name(2))
        # logger.debug(util.get_caller_name(3))

    def test_03_ParameterCommandExecutor_priorresult(self):
        pce = util.ParameterCommandExecutor(self)
        self.results = {
            "load": {"sample_movie": {"sample_frame_idx": [0, 1, 2]}}
        }
        di = [
            ("a", {"1": 1, "2": 2, "3": 3}),
            (
                "b",
                {
                    "z": 42,
                    "y": 84,
                    "x": (
                        "$get_prior_result",
                        "results, load, sample_movie, sample_frame_idx",
                    ),
                },
            ),
        ]
        di_exp = [
            ("a", {"1": 1, "2": 2, "3": 3}),
            (
                "b",
                {
                    "z": 42,
                    "y": 84,
                    "x": [0, 1, 2],
                    "x_originalnocmd": (
                        "get_prior_result",
                        "results, load, sample_movie, sample_frame_idx",
                    ),
                },
            ),
        ]
        di_out = pce.run(di)
        # logger.debug(f'dictionary expected: {di_exp}')
        # logger.debug(f'dictionary received: {di_out}')
        assert di_out == di_exp

    def test_03_ParameterCommandExecutor_previousresult(self):
        pce = util.ParameterCommandExecutor(self)
        self.results = {
            "00_load": {"sample_movie": {"sample_frame_idx": [0, 1, 2]}}
        }
        di = [
            ("load", {"1": 1, "2": 2, "3": 3}),
            (
                "identify",
                {
                    "z": 42,
                    "y": 84,
                    "x": (
                        "$get_previous_module_result",
                        "sample_movie, sample_frame_idx",
                    ),
                },
            ),
        ]
        di_exp = [
            ("load", {"1": 1, "2": 2, "3": 3}),
            (
                "identify",
                {
                    "z": 42,
                    "y": 84,
                    "x": [0, 1, 2],
                    "x_originalnocmd": (
                        "get_previous_module_result",
                        "sample_movie, sample_frame_idx",
                    ),
                },
            ),
        ]
        di_out = pce.run(di)
        # logger.debug(f'dictionary expected: {di_exp}')
        # logger.debug(f'dictionary received: {di_out}')
        assert di_out == di_exp

    def test_03_ParameterCommandExecutor_previousresult_exp(self):
        pce = util.ParameterCommandExecutor(self)
        self.results = {"00_nena": {"nena": 5.2}}
        di = [
            ("nena", {}),
            (
                "double",
                {
                    "dbl": (
                        "$get_previous_module_result *2+3",
                        "nena",
                    ),
                },
            ),
        ]
        di_exp = [
            ("nena", {}),
            (
                "double",
                {
                    "dbl": 13.4,
                    "dbl_originalnocmd": (
                        "get_previous_module_result *2+3",
                        "nena",
                    ),
                },
            ),
        ]
        di_out = pce.run(di)
        # logger.debug(f'dictionary expected: {di_exp}')
        # logger.debug(f'dictionary received: {di_out}')
        assert di_out == di_exp

    def test_04_ParameterCommandExecutor_map(self):
        mymap = {"key1": "value1", "key2": "value2"}
        pce = util.ParameterCommandExecutor(self, mymap)
        di = [
            ("a", {"1": 1, "2": 2, "3": 3}),
            ("b", {"z": 42, "y": 84, "x": ("$map", "key2")}),
        ]
        di_exp = [
            ("a", {"1": 1, "2": 2, "3": 3}),
            (
                "b",
                {
                    "z": 42,
                    "y": 84,
                    "x": "value2",
                    "x_originalnocmd": ("map", "key2"),
                },
            ),
        ]
        di_out = pce.run(di)
        # logger.debug(f'dictionary expected: {di_exp}')
        # logger.debug(f'dictionary received: {di_out}')
        assert di_out == di_exp

    def test_05_ParameterTiler(self):
        mymap = {"key1": "value1", "key2": "value2"}
        tile_entries = {
            "file_name": ["a.tiff", "b.tiff"],
            "#tags": ["RESI-1", "RESI-2"],
        }
        pce = util.ParameterTiler(self, tile_entries, mymap)
        di = [
            ("load", {"filename": ("$$map", "file_name")}),
            ("localize", {"min_ng": 20000}),
        ]
        res_exp = [
            [
                (
                    "load",
                    {
                        "filename": "a.tiff",
                        "filename_originalnocmd": ("map", "file_name"),
                    },
                ),
                ("localize", {"min_ng": 20000}),
            ],
            [
                (
                    "load",
                    {
                        "filename": "b.tiff",
                        "filename_originalnocmd": ("map", "file_name"),
                    },
                ),
                ("localize", {"min_ng": 20000}),
            ],
        ]
        res_out, tags = pce.run(di)
        logger.debug(f"result expected: {res_exp}")
        logger.debug(f"result received: {res_out}")
        assert res_out == res_exp
        logger.debug(f"tags out: {tags}")
        assert tags == ["RESI-1", "RESI-2"]

    def test_05_ParameterTiler_tags_only(self):
        """With only a '#tags' entry (no mapped files), the tiler must
        produce exactly one workflow set - this is what the single
        workflow 'no input files' mode relies on."""
        tile_entries = {"#tags": ["myrun"]}
        tiler = util.ParameterTiler(self, tile_entries)
        di = [
            ("analysis_documentation", {}),
            ("spinna_batch", {"fp_spinna_batch_config": "cfg.csv"}),
        ]
        res_out, tags = tiler.run(di)
        assert len(res_out) == 1
        assert res_out[0] == di
        assert tags == ["myrun"]

    def test_06_valid_expression(self):
        expression = "* 3.1415"
        val = util.is_valid_expression(expression)
        assert val

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
    def test_07_convert_filepath_for_machine(self, mock_node):
        mock_node.return_value = "dstmachine007"
        # a path under a known source drive root gets converted
        assert (
            util.convert_filepath_for_machine("/src/pool-a/x/data.hdf5")
            == "/dst/pool-a/x/data.hdf5"
        )
        # a path under no known drive root is returned unchanged
        assert (
            util.convert_filepath_for_machine("/home/user/data.hdf5")
            == "/home/user/data.hdf5"
        )
        # non-string / empty values pass through unchanged
        assert util.convert_filepath_for_machine(None) is None
        assert util.convert_filepath_for_machine("") == ""

    @patch("picasso_workflow.metaworkflow.platform.node")
    @patch(
        "picasso_workflow.metaworkflow.CONFIG",
        {
            "Drivepaths": {
                "srcmachineXXX": ["/src/pool-a"],
                "dstmachineXXX": ["/dst/pool-a"],
            }
        },
    )
    def test_08_pathparser_convert_path_robust(self, mock_node):
        """convert_path returns the path unchanged (no exception) when
        the machine or source drive root cannot be resolved."""
        from picasso_workflow.metaworkflow import PathParser

        # current machine not listed in Drivepaths -> unchanged
        mock_node.return_value = "unknownmachine"
        assert (
            PathParser().convert_path("/src/pool-a/data.hdf5", None)
            == "/src/pool-a/data.hdf5"
        )
        # path under no known drive root -> unchanged
        mock_node.return_value = "dstmachine001"
        assert (
            PathParser().convert_path("/elsewhere/data.hdf5", None)
            == "/elsewhere/data.hdf5"
        )
