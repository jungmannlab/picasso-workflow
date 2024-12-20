#!/usr/bin/env python
"""
Module Name: mask.py
Author: Heinrich Grabmayr
Initial Date: Dec 13, 2024
Description: This module provides a mask class for cell masking operations.
"""
import logging
import numpy as np


logger = logging.getLogger(__name__)


class CellMask:
    _mask = np.array(np.nan)
    _area = 0
    _pixelsize = 0
    _binsize = 0

    def from_mol_coords(locs, pixelsize):
        pass
