# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 10:41:53 2022

@author: Magdalena Schneider, Janelia Research Campus
"""

import unittest
import numpy as np
import dataModule as dm
import maskModule as mm
import ripleysModule as rm


class TestRandomDistribution(unittest.TestCase):

    def test_randomCase(self):
        path = "./data/Cell3"
        filename = "MutuDC_6h_stimuli"
        pixelsize = 130
        cellMask = mm.loadMask(path, f"{filename}_mask.npy", pixelsize)

        nPoints = 10000
        points, *__ = cellMask.randomPoints(nPoints)
        radii = np.arange(10, 50, 2)
        cellMask.plotPoints(points, title="Test data")

        results = rm.RipleysAnalysis(points, radii, cellMask, nControls=200)
        results.plot(
            ci=0.95,
            normalized=True,
            showControls=True,
            title="Ripleys (normalized)",
        )
        results.plot(
            ci=0.95, normalized=False, showControls=True, title="Ripleys"
        )

    def test_cross_randomCase(self):
        path = "./data/Cell3"
        filename = "MutuDC_6h_stimuli"
        pixelsize = 130
        cellMask = mm.loadMask(path, f"{filename}_mask.npy", pixelsize)

        nPoints = 10000
        pointsA, *__ = cellMask.randomPoints(nPoints)
        pointsB, *__ = cellMask.randomPoints(nPoints)
        radii = np.arange(10, 50, 2)
        cellMask.plotPoints(pointsA, title="Test data 1")
        cellMask.plotPoints(pointsB, title="Test data 2")

        results = rm.CrossRipleysAnalysis(
            pointsA, pointsB, radii, cellMask, nControls=200
        )
        results.plot(
            ci=0.95,
            normalized=True,
            showControls=True,
            title="Cross Ripleys (normalized)",
        )
        results.plot(
            ci=0.95, normalized=False, showControls=True, title="Cross Ripleys"
        )


if __name__ == "__main__":
    unittest.main()
