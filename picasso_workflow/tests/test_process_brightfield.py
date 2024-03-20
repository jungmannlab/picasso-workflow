#!/usr/bin/env python
"""
Module Name: test_process_brightfield.py
Author: Heinrich Grabmayr
Initial Date: March 15, 2024
Description: Test the module process_brightfield.py
"""
import logging
import unittest
import numpy as np

from picasso_workflow import process_brightfield as pb


logger = logging.getLogger(__name__)


# @unittest.skip('')
class TestProcessBrightfield(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_01_adjust_contrast(self):
        img = np.random.randint(0, 1000, size=(32, 48))
        img_adj = pb.adjust_contrast(img, min_quantile=0.05, max_quantile=0.95)
        logger.debug(img_adj.shape)
        logger.debug(img.shape)
        assert img_adj.shape == tuple(list(img.shape) + [3])
