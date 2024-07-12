# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 22:11:55 2022

@author: Magdalena Schneider, Janelia Research Campus
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter


class Mask:
    def __init__(self, mask, pixelsize):
        self.mask = mask
        self.shape = mask.shape
        self.pixelsize = pixelsize
        self.area = self.getArea()

    def getArea(self):
        maskArea = self.getAreaInPixel() * (self.pixelsize**2)
        return maskArea

    def getAreaInPixel(self):
        return self.mask.sum()

    def getCoveredFraction(self):
        imageArea = self.shape[0] * self.shape[1]
        maskFraction = self.mask.sum() / imageArea
        return maskFraction

    def plot(self):
        plt.figure()
        plt.imshow(self.mask, origin="lower")

    def save(self, path, filename):
        maskfile = os.path.join(path, filename)
        np.save(maskfile, self.mask)

    def randomPoints(self, nPoints):
        imageArea = self.mask.shape[0] * self.mask.shape[1]
        density = nPoints / self.getAreaInPixel()
        nSample = int(
            density * imageArea
        )  # density points in mask, scale to whole image

        points = np.random.uniform(
            0, self.mask.shape[0], size=(nSample, 2)
        )  # create points in units of pixels
        # TODO: get borders of mask in order to make random
        # number generation a bit more efficient

        # Reject points outside of mask
        (steps_x, steps_y) = (1, 1)
        x_ind = (np.floor(points[:, 0] / steps_x)).astype(int)
        y_ind = (np.floor(points[:, 1] / steps_y)).astype(int)
        index = self.mask[y_ind, x_ind].astype(bool)
        points = points[index, :]  # locs in the mask
        points *= self.pixelsize
        nFactor = (
            len(points) / nPoints
        )  # factor of actual to desired number of returned points
        return points, nFactor

    def plotPoints(self, points, title=None):
        plt.figure()
        plt.plot(points[:, 0], points[:, 1], ".", markersize=1)
        plt.xlim(0, self.shape[0] * self.pixelsize)
        plt.ylim(0, self.shape[0] * self.pixelsize)
        axes = plt.gca()
        axes.set_aspect("equal")
        if title is not None:
            axes.set_title(title)
        plt.xlabel("x")
        plt.ylabel("y")

    def crop(self, data):
        """Crop data and return only points that lie with in the mask"""
        # map x and y values of data to look up in mask
        x_lut = (data.x / self.pixelsize).astype(np.int32)
        y_lut = (data.y / self.pixelsize).astype(np.int32)
        # do inverse operation of the one in createMask
        fudr_mask = np.rot90(np.flipud(self.mask), k=-1)
        inbounds = (x_lut < fudr_mask.shape[0]) & (y_lut < fudr_mask.shape[1])
        x_lut[~inbounds] = 0
        y_lut[~inbounds] = 0
        inmask = fudr_mask[x_lut, y_lut]
        inmask = inmask & inbounds
        return data[inmask]


def loadMask(path, filename, pixelsize):
    maskfile = os.path.join(path, filename)
    maskData = np.load(maskfile)
    mask = Mask(maskData, pixelsize)
    return mask


def createMask(data, pixelsize):
    binningFactor = 1
    binEdges = np.arange(0, (512 + 1), binningFactor)  # binning
    histCounts, xedges, yedges = np.histogram2d(
        data.x / pixelsize, data.y / pixelsize, bins=[binEdges, binEdges]
    )
    histCounts = np.flipud(np.rot90(histCounts))
    histCounts = gaussian_filter(histCounts, sigma=0.3)
    histCounts = zoom(
        histCounts, binningFactor, order=0
    )  # upsample to original pixel number
    maskData = histCounts > 0  # create binary mask from histogram
    mask = Mask(maskData, pixelsize)
    return mask
