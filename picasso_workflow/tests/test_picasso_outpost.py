#!/usr/bin/env python
"""
Module Name: test_picasso_outpost.py
Author: Heinrich Grabmayr
Initial Date: March 15, 2024
Description: Test the module picasso_outpost.py
"""
import logging
import unittest
import numpy as np

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
