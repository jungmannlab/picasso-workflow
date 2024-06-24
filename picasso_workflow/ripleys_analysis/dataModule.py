# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 22:22:05 2022

@author: Magdalena Schneider, Janelia Research Campus
"""

import os
from tqdm import tqdm
import pandas as pd
import yaml

# import h5py
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


class LocalizationData:
    def __init__(self, path, filename, fileIDs):
        self.path = path
        self.filename = filename
        self.fileIDs = fileIDs
        self.nReceptors = len(fileIDs)
        self.nPixels = (512, 512)
        self.pixelsize = self.loadPixelSize()
        self.data = self.loadData()  # list of data
        self.forest = self.buildForest()  # list of all trees
        self.allData = self.loadAllData()  # multi-file

    def loadPixelSize(self):
        filename = self.filename + "_multi.yaml"
        fileinfo = loadYaml(self.path, filename)
        pixelsize = fileinfo["Pixelsize"]  # given in nm
        return pixelsize

    def loadAllData(self):
        filename = f"{self.filename}_multi.hdf5"
        file = os.path.join(self.path, filename)
        df = pd.read_hdf(file, key="locs")
        allData = df[
            ["x", "y"]
        ]  # load xy-coordinates of locs (input is in px)
        allData *= self.pixelsize
        return allData

    def loadData(self):
        print("Loading data...")
        data = []
        for k in tqdm(self.fileIDs):
            thisFilename = f"{self.filename}_Receptor_{k}.hdf5"
            file = os.path.join(self.path, thisFilename)
            df = pd.read_hdf(file, key="locs")
            locs = df[["x", "y"]]  # load xy-coords of locs (input is in px)
            locs *= self.pixelsize
            data.append(locs)
        return data

    def buildForest(self):
        print("Building forest...")
        forest = []  # list of all trees
        for k in tqdm(range(self.nReceptors)):
            tree = KDTree(self.data[k])
            forest.append(tree)
        return forest

    def plot(self, receptor="all", title=None):
        plt.figure()
        if receptor == "all":
            plt.plot(self.allData.x, self.allData.y, ".", markersize=1)
        elif (
            (type(receptor) == int)
            and (receptor >= 1)
            and (receptor <= self.nReceptors)
        ):
            plt.plot(
                self.data[receptor - 1].x,
                self.data[receptor - 1].y,
                ".",
                markersize=1,
            )
        else:
            raise ValueError("Invalid receptor id.")
        plt.xlim(0, self.nPixels[0] * self.pixelsize)
        plt.ylim(0, self.nPixels[0] * self.pixelsize)
        axes = plt.gca()
        axes.set_aspect("equal")
        if title is not None:
            axes.set_title(title)


class LocalizationData_fromlocs:
    def __init__(
        self, channel_locs, combined_locs, fileIDs, pixelsize, folder
    ):
        self.fileIDs = fileIDs
        self.nReceptors = len(fileIDs)
        self.nPixels = (512, 512)
        self.pixelsize = pixelsize
        self.data = self.loadData(channel_locs)  # list of data
        self.forest = self.buildForest()  # list of all trees
        self.allData = self.loadAllData(combined_locs)  # multi-file
        self.folder = folder

    def loadAllData(self, combined_locs):
        # load xy-coordinates of locs (input is in px)
        allData = pd.DataFrame(combined_locs[["x", "y"]])
        allData *= self.pixelsize
        return allData

    def loadData(self, channel_locs):
        print("Loading data...")
        data = []
        for k, locs in enumerate(channel_locs):
            locs = pd.DataFrame(locs[["x", "y"]]) * self.pixelsize
            data.append(locs)
        return data

    def applyMask(self, mask):
        for k in range(len(self.data)):
            self.data[k] = mask.crop(self.data[k])

    def buildForest(self):
        print("Building forest...")
        forest = []  # list of all trees
        for k in tqdm(range(self.nReceptors)):
            tree = KDTree(self.data[k])
            forest.append(tree)
        return forest

    def plot(self, receptor="all", title=None):
        plt.figure()
        if receptor == "all":
            plt.plot(self.allData.x, self.allData.y, ".", markersize=1)
        elif (
            (isinstance(receptor, int))
            and (receptor >= 1)
            and (receptor <= self.nReceptors)
        ):
            plt.plot(
                self.data[receptor - 1].x,
                self.data[receptor - 1].y,
                ".",
                markersize=1,
            )
        else:
            raise ValueError("Invalid receptor id.")
        plt.xlim(0, self.nPixels[0] * self.pixelsize)
        plt.ylim(0, self.nPixels[0] * self.pixelsize)
        axes = plt.gca()
        axes.set_aspect("equal")
        if title is not None:
            axes.set_title(title)
        fig = plt.gcf()
        fig.savefig(os.path.join(self.folder, "ripleys.png"))


def loadYaml(path, filename):
    filepath = os.path.join(path, filename)
    file = open(filepath)
    generator = yaml.load_all(file, Loader=yaml.FullLoader)
    fileinfo = next(generator)
    for d in generator:
        fileinfo.update(d)
    return fileinfo


def loadLocalizationData(path, filename, nReceptors):
    locData = LocalizationData(path, filename, nReceptors)
    return locData


if __name__ == "__main__":
    path = "./data/Cell3"
    filename = "MutuDC_6h_stimuli"
    fileIDs = list(range(1, 7))
    locData = loadLocalizationData(path, filename, fileIDs)
