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
            K = (nNeighbors / N) / density
        else:
            tree = getTree(data)
            otherTree = getTree(otherData)
            # if isTree(otherData):
            #     otherN = otherData.n
            # else:
            #     otherN = otherData.shape[0]
            nNeighbors = tree.count_neighbors(otherTree, self.radii)

            # Rafal's correction:
            # lambda_inv1 = area / N
            # lambda_inv2 = area / otherN
            # const_term = lambda_inv1 * lambda_inv2 / area
            # K = const_term * nNeighbors

            # GOING back to magdalena's version
            K = (nNeighbors / N) / density

        L = np.sqrt(K / np.pi)
        H = L - self.radii

        # FOR TESTING: exchange K with RDF
        # now this is the radial distribution function
        if self.atype == "RDF":
            K = nNeighbors / N  # / density
            # mean number of other spots within a distance r of a self-spot
            n_means = nNeighbors / N
            # difference of this number from one radius to the next
            d_n_means = n_means[1:] - n_means[:-1]
            mean_r = (self.radii[1:] + self.radii[:-1]) / 2
            d_r = self.radii[1:] - self.radii[:-1]
            d_areas = 2 * np.pi * mean_r * d_r
            # density in the annulus between two radii
            rdf = d_n_means / d_areas
            K[:-1] = rdf
            K[-1] = rdf[-1]  # duplicate last entry to keep lengths

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
        K = np.zeros_like(nNeighbors)
        # print('nNeighbors shape', nNeighbors.shape)

        # always randomize to the maximum radius (otherwise, uncomment in the for loops)
        data_rnd = self.randomize_data(data, np.max(self.radii))
        if otherData is not None:
            other_data_rnd = self.randomize_data(otherData, np.max(self.radii))

        for i, r in enumerate(self.radii):
            # create uniform random data in a circle of radius r
            # data_rnd = self.randomize_data(data, r)
            tree = getTree(data_rnd)
            if otherData is None:
                # print('tree count neighbors: ', tree.count_neighbors(tree, r))
                nNeighbors[i] = tree.count_neighbors(tree, r) - N
                K[i] = (nNeighbors[i] / N) / density
            else:
                # other_data_rnd = self.randomize_data(otherData, r)
                otherTree = getTree(other_data_rnd)
                otherN = otherTree.n
                # print('tree count neighbors: ', tree.count_neighbors(tree, r))
                nNeighbors[i] = tree.count_neighbors(otherTree, r)
                lambda_inv1 = area / N
                lambda_inv2 = area / otherN
                const_term = lambda_inv1 * lambda_inv2 / area
                K = const_term * nNeighbors[i]

        L = np.sqrt(K / np.pi)
        H = L - self.radii

        # FOR TESTING: exchange K with RDF
        # now this is the radial distribution function
        if self.atype == "RDF":
            K = nNeighbors / N  # / density
            # mean number of other spots within a distance r of a self-spot
            n_means = nNeighbors / N
            # difference of this number from one radius to the next
            d_n_means = n_means[1:] - n_means[:-1]
            mean_r = (self.radii[1:] + self.radii[:-1]) / 2
            d_r = self.radii[1:] - self.radii[:-1]
            d_areas = 2 * np.pi * mean_r * d_r
            # density in the annulus between two radii
            rdf = d_n_means / d_areas
            K[:-1] = rdf
            K[-1] = rdf[-1]  # duplicate last entry to keep lengths

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

        if self.atype == "RDF":
            # no not normalize at all, but calculate the difference
            Knormalized = K - ripleysMean
            return Knormalized

        quantileLow = (1 - ci) / 2
        quantileHigh = 1 - (1 - ci) / 2
        # Divide all positive values by high quantile:
        idxPos = Knormalized >= 0
        quantilesHigh = self.getRipleysQuantiles(quantileHigh)
        # if self.atype == "RDF":
        #     # reset very low quantiles
        #     quantilesHigh[
        #         np.abs(quantilesHigh)
        #         < 0.25 * np.nanmean(np.abs(quantilesHigh))
        #     ] = 0.25 * np.nanmean(quantilesHigh)
        Knormalized[idxPos] /= abs(
            (quantilesHigh[idxPos] - ripleysMean[idxPos])
        )
        # Divide all negative values by low quantile:
        quantilesLow = self.getRipleysQuantiles(quantileLow)
        # if self.atype == "RDF":
        #     # reset very low quantiles
        #     quantilesLow[
        #         np.abs(quantilesLow) < 0.25 * np.nanmean(np.abs(quantilesLow))
        #     ] = 0.25 * np.nanmean(quantilesLow)
        Knormalized[~idxPos] /= abs(
            (quantilesLow[~idxPos] - ripleysMean[~idxPos])
        )

        return Knormalized

    def get_curve_outside_quantiles(self, K, ci=0.95):
        K_outside = np.zeros_like(K)

        quantileLow = (1 - ci) / 2
        quantileHigh = 1 - (1 - ci) / 2

        quantilesHigh = self.getRipleysQuantiles(quantileHigh)
        quantilesLow = self.getRipleysQuantiles(quantileLow)

        idxpos_outside = np.argwhere((K > quantilesHigh) | (K < quantilesLow))
        K_outside[idxpos_outside] = K[idxpos_outside]

        return K_outside

    def getRipleysMean(self):
        return self.ripleysCurves_controls["mean"]

    def getRipleysQuantiles(self, quantile):
        quantilesK = [
            np.quantile(x, quantile)
            for x in np.transpose(self.ripleysCurves_controls["K"])
        ]
        return np.array(quantilesK)

    def calculateRipleysIntegral(self, interval=None):
        if self.atype == "RDF":
            mean_r = (self.radii[1:] + self.radii[:-1]) / 2
            d_r = self.radii[1:] - self.radii[:-1]
            d_areas = 2 * np.pi * mean_r * d_r
            # for RDF, "normalized" curves are density differences.
            # calculate the number of "lacking" or "overpopulated"
            # proteins with respect to random, in the range of the radii

            # only take into account the regions outsize the confidence
            # interval
            k_significant = self.get_curve_outside_quantiles(
                self.ripleysCurves_data["normalized"], ci=0.95
            )
            n_overpop = d_areas * k_significant[:-1]
            return np.sum(n_overpop)

        if interval is None:
            integral = np.trapz(
                self.ripleysCurves_data["normalized"], self.radii
            )
        else:
            f_limits = np.interp(
                interval, self.ripleysCurves_data["normalized"], self.radii
            )
            f = [
                f_limits[0],
                self.ripleysCurves_data["normalized"],
                f_limits[1],
            ]
            x = [interval[0], self.radii, interval[1]]
            integral = np.trapz(f, x)
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
            if self.atype == "Ripleys":
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
            if self.atype == "Ripleys":
                axes.set_ylabel("Normalized K(d)", fontsize=labelFontsize)
            elif self.atype == "RDF":
                axes.set_ylabel("Normalized RDF(d)", fontsize=labelFontsize)
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
            if self.atype == "Ripleys":
                axes.set_ylabel("K(d)", fontsize=labelFontsize)
            elif self.atype == "RDF":
                axes.set_ylabel("RDF(d)", fontsize=labelFontsize)

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
    def __init__(self, data, radii, cellMask, nControls, atype="Ripleys"):
        super().__init__(radii, cellMask, nControls)
        self.atype = atype
        if atype == "Ripleys":
            self.ripleysCurves_controls = self.getRipleysRandomControlCurves(
                getNumberPoints(data), cellMask
            )  # dictionary: K, H, L, normalized (lists)
        elif atype == "RDF":
            self.ripleysCurves_controls = self.getRipleysRandomControlCurves2(
                data, area=cellMask.area
            )  # dictionary: K, H, L, normalized (lists)
        else:
            raise NotImplementedError()
        self.ripleysCurves_data = self.getRipleysDataCurves(
            data, area=cellMask.area
        )  # dictionary: K, H, L, normalized
        self.representativeData_control  # list
        self.ripleysIntegral_data = self.calculateRipleysIntegral()


class CrossRipleysAnalysis(RipleysInterface):
    def __init__(
        self, data, otherData, radii, cellMask, nControls, atype="Ripleys"
    ):
        super().__init__(radii, cellMask, nControls)
        self.atype = atype
        if atype == "Ripleys":
            self.ripleysCurves_controls = self.getRipleysRandomControlCurves(
                getNumberPoints(data), cellMask, otherData
            )  # dictionary: K, H, L, normalized (lists)
        elif atype == "RDF":
            self.ripleysCurves_controls = self.getRipleysRandomControlCurves2(
                data, area=cellMask.area
            )  # dictionary: K, H, L, normalized (lists)
        else:
            raise NotImplementedError()
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
