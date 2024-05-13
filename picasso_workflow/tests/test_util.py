#!/usr/bin/env python
"""
Module Name: test_util.py
Author: Heinrich Grabmayr
Initial Date: March 14, 2024
Description: Test the module util.py
"""
import logging
import unittest

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
            ("b", {"z": 42, "y": 84, "x": [0, 1, 2]}),
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
            ("identify", {"z": 42, "y": 84, "x": [0, 1, 2]}),
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
            ("double", {"dbl": 13.4}),
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
            ("b", {"z": 42, "y": 84, "x": "value2"}),
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
                ("load", {"filename": "a.tiff"}),
                ("localize", {"min_ng": 20000}),
            ],
            [
                ("load", {"filename": "b.tiff"}),
                ("localize", {"min_ng": 20000}),
            ],
        ]
        res_out, tags = pce.run(di)
        logger.debug(f"result expected: {res_exp}")
        logger.debug(f"result received: {res_out}")
        assert res_out == res_exp
        logger.debug(f"tags out: {tags}")
        assert tags == ["RESI-1", "RESI-2"]
