# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 23:26:00 2022

@author: Magdalena Schneider, Janelia Research Campus
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from tqdm import tqdm


# %% Class interface


class RipleysInterface:
    def __init__(self, radii, cellMask, nControls=100):
        self.nControls = nControls
        self.mask = cellMask
        self.radii = radii

    def getRipleysRandomControlCurves(self, nPoints, cellMask, otherData=None):
        K = []
        L = []
        H = []
        print("Generating random controls...")
        for j in tqdm(range(self.nControls)):
            points, *_ = cellMask.randomPoints(nPoints)
            if j == 0:
                self.representativeData_control = points

            ripleysCurves = self.getRipleysCurves(
                points, otherData=otherData, area=cellMask.area
            )
            K.append(ripleysCurves["K"])
            L.append(ripleysCurves["H"])
            H.append(ripleysCurves["L"])

        meanControl = calculateRipleysMean(K)

        ripleysRandomControlCurves = {
            "K": np.array(K),
            "L": np.array(L),
            "H": np.array(H),
            "mean": np.array(meanControl),
        }
        return ripleysRandomControlCurves

    def getRipleysRandomControlCurves2(self, data, otherData=None, area=None):
        K = []
        L = []
        H = []
        print("Generating random controls...")
        for j in tqdm(range(self.nControls)):
            ripleysCurves, data_rnd = self.getRandomizedRipleysCurves(
                data, otherData, area=area
            )
            if j == 0:
                self.representativeData_control = data_rnd
            K.append(ripleysCurves["K"])
            L.append(ripleysCurves["H"])
            H.append(ripleysCurves["L"])

        meanControl = calculateRipleysMean(K)

        ripleysRandomControlCurves = {
            "K": np.array(K),
            "L": np.array(L),
            "H": np.array(H),
            "mean": np.array(meanControl),
        }
        return ripleysRandomControlCurves

    def getRipleysDataCurves(self, data, otherData=None, area=None):
        ripleysCurves = self.getRipleysCurves(data, otherData, area=area)
        ripleysCurves["normalized"] = self.normalizeCurve(ripleysCurves["K"])
        return ripleysCurves

    def getRipleysCurves(self, data, otherData=None, area=None):
        assert (
            area is not None
        ), "Input parameter area not specified, area is None"

        if isTree(data):
            N = data.n
        else:
            N = data.shape[0]
        density = N / area

        if otherData is None:
            tree = getTree(data)
            nNeighbors = tree.count_neighbors(tree, self.radii) - N
        else:
            tree = getTree(data)
            otherTree = getTree(otherData)
            nNeighbors = tree.count_neighbors(otherTree, self.radii)

        K = (nNeighbors / N) / density
        L = np.sqrt(K / np.pi)
        H = L - self.radii

        K = nNeighbors / N  # / density
        # now this is the radial distribution function
        n_means = nNeighbors / N
        d_n_means = n_means[1:] - n_means[:-1]
        mean_r = (self.radii[1:] + self.radii[:-1]) / 2
        d_r = self.radii[1:] - self.radii[:-1]
        d_areas = 2 * np.pi * mean_r * d_r
        rdf = d_n_means / d_areas
        K[:-1] = rdf
        K[-1] = rdf[-1]

        ripleysCurves = {"K": np.array(K), "L": np.array(L), "H": np.array(H)}
        return ripleysCurves

    def getRandomizedRipleysCurves(self, data, otherData=None, area=None):
        """Randomize the data for each radius at the respective length scale"""
        assert (
            area is not None
        ), "Input parameter area not specified, area is None"

        if isTree(data):
            N = data.n
        else:
            N = data.shape[0]
        density = N / area

        nNeighbors = np.zeros(len(self.radii))
        # print('nNeighbors shape', nNeighbors.shape)

        for i, r in enumerate(self.radii):
            # create uniform random data in a circle of radius r
            data_rnd = self.randomize_data(data, r)
            if otherData is None:
                tree = getTree(data_rnd)
                # print('tree count neighbors: ', tree.count_neighbors(tree, r))
                nNeighbors[i] = tree.count_neighbors(tree, r) - N
            else:
                tree = getTree(data)
                other_data_rnd = self.randomize_data(otherData, r)
                otherTree = getTree(other_data_rnd)
                # print('tree count neighbors: ', tree.count_neighbors(tree, r))
                nNeighbors[i] = tree.count_neighbors(otherTree, r)

        K = (nNeighbors / N) / density
        L = np.sqrt(K / np.pi)
        H = L - self.radii

        ripleysCurves = {"K": np.array(K), "L": np.array(L), "H": np.array(H)}
        return ripleysCurves, data_rnd

    def randomize_data(self, data, r):
        # create uniform random data in a circle of radius r
        N = data.shape[0]
        phase_rnd = np.exp(1j * 2 * np.pi * np.random.random(N))
        r_rnd = r * np.random.power(a=3, size=N)  # quadratic
        cart_rnd = np.stack(
            [
                r_rnd * np.real(phase_rnd),
                r_rnd * np.imag(phase_rnd),
            ]
        ).T
        return data + cart_rnd

    def normalizeCurve(self, K, ci=0.95):
        ripleysMean = self.getRipleysMean()
        Knormalized = K - ripleysMean

        quantileLow = (1 - ci) / 2
        quantileHigh = 1 - (1 - ci) / 2
        # Divide all positive values by high quantile:
        idxPos = Knormalized >= 0
        quantilesHigh = self.getRipleysQuantiles(quantileHigh)
        Knormalized[idxPos] /= abs(
            (quantilesHigh[idxPos] - ripleysMean[idxPos])
        )
        # Divide all negative values by low quantile:
        quantilesLow = self.getRipleysQuantiles(quantileLow)
        Knormalized[~idxPos] /= abs(
            (quantilesLow[~idxPos] - ripleysMean[~idxPos])
        )

        return Knormalized

    def getRipleysMean(self):
        return self.ripleysCurves_controls["mean"]

    def getRipleysQuantiles(self, quantile):
        quantilesK = [
            np.quantile(x, quantile)
            for x in np.transpose(self.ripleysCurves_controls["K"])
        ]
        return np.array(quantilesK)

    def calculateRipleysIntegral(self):
        integral = np.trapz(self.ripleysCurves_data["normalized"], self.radii)
        return integral

    def plot(
        self,
        ci=0.95,
        normalized=True,
        showControls=False,
        title=None,
        labelFontsize=14,
        axes=None,
    ):
        # Plot Ripley's K and confidence interval
        if axes is None:
            plt.figure()
            axes = plt.gca()
        if normalized:
            if showControls:
                for k in range(self.nControls):
                    axes.plot(
                        self.radii,
                        self.normalizeCurve(
                            self.ripleysCurves_controls["K"][k]
                        ),
                        c="lightgray",
                        label="Random controls",
                        linestyle="-",
                    )
            axes.plot(
                self.radii,
                np.zeros(len(self.radii)),
                c="k",
                label=f"{ci*100}% envelope",
                linestyle="--",
            )
            axes.plot(
                self.radii, np.ones(len(self.radii)), c="k", linestyle=":"
            )
            axes.plot(
                self.radii, -np.ones(len(self.radii)), c="k", linestyle=":"
            )
            axes.plot(
                self.radii,
                self.ripleysCurves_data["normalized"],
                c="k",
                label="Observed data",
                linewidth=2.0,
            )
            axes.set_xlabel("d (nm)", fontsize=labelFontsize)
            axes.set_ylabel("Normalized K(d)", fontsize=labelFontsize)
        else:
            if showControls:
                for k in range(self.nControls):
                    axes.plot(
                        self.radii,
                        self.ripleysCurves_controls["K"][k],
                        c="lightgray",
                        label="Random controls",
                        linestyle="-",
                    )
            quantileLow = (1 - ci) / 2
            quantileHigh = 1 - (1 - ci) / 2
            axes.plot(
                self.radii,
                self.ripleysCurves_controls["mean"],
                c="k",
                label="Mean of random controls",
                linestyle="--",
            )
            axes.plot(
                self.radii,
                self.getRipleysQuantiles(quantileHigh),
                c="k",
                label=f"{ci*100}% envelope",
                linestyle=":",
            )
            axes.plot(
                self.radii,
                self.getRipleysQuantiles(quantileLow),
                c="k",
                linestyle=":",
            )
            axes.plot(
                self.radii,
                self.ripleysCurves_data["K"],
                c="k",
                label="Observed data",
                linewidth=1.0,
            )
            axes.set_xlabel("d (nm)", fontsize=labelFontsize)
            axes.set_ylabel("K(d)", fontsize=labelFontsize)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=labelFontsize)

        if title is not None:
            axes.set_title(title, fontsize=labelFontsize)

    def plotRepresentativeControl(self, title="Random control", axes=None):
        if axes is None:
            plt.figure()
            axes = plt.gca()
        points = self.representativeData_control
        axes.plot(points[:, 0], points[:, 1], ".", markersize=1)
        axes.xlim(0, self.mask.shape[0] * self.mask.pixelsize)
        axes.ylim(0, self.mask.shape[1] * self.mask.pixelsize)
        axes.set_aspect("equal")
        axes.set_title(title)


# %% Subclasses


class RipleysAnalysis(RipleysInterface):
    def __init__(self, data, radii, cellMask, nControls):
        super().__init__(radii, cellMask, nControls)
        # self.ripleysCurves_controls = self.getRipleysRandomControlCurves(
        #     getNumberPoints(data), cellMask
        # )  # dictionary: K, H, L, normalized (lists)
        self.ripleysCurves_controls = self.getRipleysRandomControlCurves2(
            data, area=cellMask.area
        )  # dictionary: K, H, L, normalized (lists)
        self.ripleysCurves_data = self.getRipleysDataCurves(
            data, area=cellMask.area
        )  # dictionary: K, H, L, normalized
        self.representativeData_control  # list
        self.ripleysIntegral_data = self.calculateRipleysIntegral()


class CrossRipleysAnalysis(RipleysInterface):
    def __init__(self, data, otherData, radii, cellMask, nControls):
        super().__init__(radii, cellMask, nControls)
        # self.ripleysCurves_controls = self.getRipleysRandomControlCurves(
        #     getNumberPoints(data), cellMask, otherData
        # )  # dictionary: K, H, L, normalized (lists)
        self.ripleysCurves_controls = self.getRipleysRandomControlCurves2(
            data, area=cellMask.area
        )  # dictionary: K, H, L, normalized (lists)
        self.ripleysCurves_data = self.getRipleysDataCurves(
            data, otherData, area=cellMask.area
        )  # dictionary: K, H, L, normalized
        self.representativeData_control  # list
        self.ripleysIntegral_data = self.calculateRipleysIntegral()


# %% Helper functions


def calculateRipleysMean(Ks):
    meanK = np.mean(Ks, 0)
    return meanK


def isTree(data):
    return isinstance(data, KDTree)


def getTree(data):
    if isTree(data):
        tree = data
    else:
        tree = KDTree(data)
    return tree


def getNumberPoints(data):
    if isTree(data):
        return data.n
    else:
        return data.shape[0]


def initializeResultsMatrix(nFiles):
    return [[0] * nFiles for i in range(nFiles)]
