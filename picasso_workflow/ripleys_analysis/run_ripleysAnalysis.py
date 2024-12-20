# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 22:43:38 2022

@author: Magdalena Schneider, Janelia Research Campus

Script for Ripley's Analysis of multiplexed single molecule localization data
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import picasso_workflow.ripleys_analysis.maskModule as mm
import picasso_workflow.ripleys_analysis.dataModule as dm
import picasso_workflow.ripleys_analysis.ripleysModule as rm


def performRipleysMultiAnalysis(
    path,
    filename,
    fileIDs,
    radii,
    nRandomControls=100,
    channel_locs=None,
    combined_locs=None,
    pixelsize=None,
    atype="Ripleys",  # or RDF
):

    np.random.seed(0)

    print(f"Cell path: {path}/{filename}")

    #  Load data
    nFiles = len(fileIDs)
    if channel_locs is None:
        locData = dm.loadLocalizationData(path, filename, fileIDs)
    else:
        locData = dm.LocalizationData_fromlocs(
            channel_locs, combined_locs, fileIDs, pixelsize, path
        )

    #  Mask

    # Load mask from file
    # cellMask = loadMask(path, "Cell_Mask.npy", pixelsize)
    # cellMask.plot()

    # Create mask from all localization data
    cellMask = mm.createMask(locData.allData, locData.pixelsize)
    cellMask.plot()
    cellMask.save(path, f"{filename}_mask")

    # locData.applyMask(cellMask)

    # plt.hexbin(locData.allData['x'], locData.allData['y'])
    # plt.show()

    # Perform Ripley's analysis for all data pairs
    ripleysResults = rm.initializeResultsMatrix(nFiles)
    ripleysIntegrals = np.zeros((nFiles, nFiles))
    ripleys_mean = np.zeros((nFiles, nFiles))

    for j in range(nFiles):
        for k in range(nFiles):
            print(f"Analyzing files {fileIDs[j]} with {fileIDs[k]}...")
            if j == k:
                if atype == "Ripleys":
                    ripleysResults[j][k] = rm.RipleysAnalysis(
                        locData.forest[j],
                        radii,
                        cellMask,
                        nRandomControls,
                        atype,
                    )
                elif atype == "RDF":
                    ripleysResults[j][k] = rm.RipleysAnalysis(
                        locData.data[j],
                        radii,
                        cellMask,
                        nRandomControls,
                        atype,
                    )
                else:
                    raise NotImplementedError()
            else:
                if atype == "Ripleys":
                    ripleysResults[j][k] = rm.CrossRipleysAnalysis(
                        locData.forest[j],
                        locData.forest[k],
                        radii,
                        cellMask,
                        nRandomControls,
                        atype,
                    )
                elif atype == "RDF":
                    ripleysResults[j][k] = rm.CrossRipleysAnalysis(
                        locData.data[j],
                        locData.data[k],
                        radii,
                        cellMask,
                        nRandomControls,
                        atype,
                    )
                else:
                    raise NotImplementedError()
            ripleysIntegrals[j, k] = ripleysResults[j][k].ripleysIntegral_data
            # curve_data = ripleysResults[j][k].ripleysCurves_data["normalized"]
            # mean_val = np.nanmean(curve_data[~np.isinf(curve_data)])
            # ripleys_mean[j, k] = mean_val

            # calculate mean of a function: integral divided by
            # integration interval
            ripleys_mean[j, k] = ripleysIntegrals[j, k] / np.max(radii)

    # Normalized plot
    figsize = 30
    fig, axs = plt.subplots(nFiles, nFiles, figsize=(figsize, figsize))
    for j in range(nFiles):
        for k in range(nFiles):
            ripleysResults[j][k].plot(
                ci=0.95,
                normalized=True,
                showControls=True,
                title=f"{fileIDs[j]} -> {fileIDs[k]}",
                labelFontsize=30,
                axes=axs[j][k],
            )
    fig.savefig(os.path.join(path, f"{filename}{atype}_normalized.png"))

    # Unnormalized plot
    fig, axs = plt.subplots(nFiles, nFiles, figsize=(figsize, figsize))
    for j in range(nFiles):
        for k in range(nFiles):
            ripleysResults[j][k].plot(
                ci=0.95,
                normalized=False,
                showControls=True,
                title=f"{fileIDs[j]} -> {fileIDs[k]}",
                labelFontsize=30,
                axes=axs[j][k],
            )
    fig.savefig(os.path.join(path, f"{filename}{atype}_unnormalized.png"))

    # Print and save integral matrix
    print(f"Integral matrix:\n{ripleysIntegrals}\n")
    integralfile = os.path.join(path, f"{filename}_ripleysIntegrals")
    np.save(integralfile, ripleysIntegrals)

    # Print and save mean matrix
    print(f"Mean matrix:\n{ripleys_mean}\n")
    ripleys_mean_file = os.path.join(path, f"{filename}_ripleysIntegrals")
    np.save(ripleys_mean_file, ripleys_mean)

    if atype == "RDF":
        ripleys_mean = ripleysIntegrals

    # ripleysMeanVal = ripleysIntegrals / (2 * np.max(radii))

    return ripleysResults, ripleysIntegrals, ripleys_mean


# Set file paths and parameters

if __name__ == "__main__":
    tstart = time.time()

    cellPaths = [r"Z:\users\Cell "]

    filenames = ["Dummy"]

    fileIDs = [
        "MHC-I",
        "MHC-II",
        "CD86",
        "CD80",
        "PDL1",
        "PDL2",
    ]  # [1,2,3,4,5,6] # use list(range(1,7)) for all from 1 to 6

    nRandomControls = 100
    rmax = 200
    radii = np.concatenate((np.arange(10, 80, 2), np.arange(80, rmax, 12)))

    # %% Perform Ripleys analysis over multiple receptors for each cell

    allResults = []
    allIntegrals = []
    for path, filename in zip(cellPaths, filenames):
        ripleysResults, ripleysIntegrals = performRipleysMultiAnalysis(
            path,
            filename,
            fileIDs,
            radii=radii,
            nRandomControls=nRandomControls,
        )
        allResults.append(ripleysResults)
        allIntegrals.append(ripleysIntegrals)

    # %% Average Ripleys matrices over all cells
    meanMatrix = np.mean(np.dstack(allIntegrals), axis=2)
    np.savetxt(r"Z:\users\meanMatrix", meanMatrix)

    # %% Runtime
    elapsedTime = time.time() - tstart
    print(f"Elapsed time for whole analysis: {elapsedTime:.3f} s")
