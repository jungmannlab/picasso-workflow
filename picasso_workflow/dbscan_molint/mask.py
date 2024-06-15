# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:06:35 2022

@author: reinhardt
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import math
import pickle

from skimage import filters
import scipy.ndimage as ndi
import configparser
from datetime import datetime


from picasso_workflow.dbscan_molint import io

"""
===============================================================================
Mask Generation
===============================================================================
"""


def gen_mask(
    x,
    y,
    margin,
    binsize,
    σ_mask_blur,
    mask_resolution,
    px,
    output_path,
    filename,
    plot_figures=False,
):
    plt.close("all")
    """
    Generate a mask for the coordinate arrays x and y.
    Saves the mask and the boundaries of the bins (xedges, yedges) as npy files
    and as a dict. Optionally saves plots to monitor the mask generation.


    Parameters
    ----------
    x : array like
        x coordinates.
    y : array like
        y coordinates.
    margin : float
        Size of the added empty margin to the FOV in nm.
    binsize : float
        Size of the 2D histogram of the first step in nm.
    σ_mask_blur : float
        Parameter of the gaussian blur in "binsize" units.
    mask_resolution : float
        Controls the digital resolution of the mask in nm.
    px : float
        Pixel size in nm.
    output_path : string
        Path to output folder.
    filename : string
        Filename to save output files.
    plot_figures : bool, optional
        If True some output figures will be saved. The default is False.

    Returns
    -------
    mask_dict : dict
        Dictionary containing the mask, the x and y edges of the mask and the
        mask area.

    """

    # Translate to pixel
    margin_px = margin / px
    binsize_px = binsize / px
    mask_resolution_px = mask_resolution / px

    # remove path from filename
    filepath, filename = os.path.split(filename)

    # Filename for mask
    mask_name = filename.replace(".hdf5", "_MASK")

    # Create subfolder for mask generation files
    mask_gen_path = os.path.join(output_path, "mask")
    try:
        os.mkdir(mask_gen_path)

    except OSError:
        pass

    # Change format of x and y arrays
    pos_exp = np.array([x, y]).T

    # Create proper roi
    x0 = x.min() - margin_px
    y0 = y.min() - margin_px
    length = np.max((x.max() - x.min(), y.max() - y.min())) + 2 * margin_px

    # Scatter plot of input data
    if plot_figures:

        fig0, ax0 = plt.subplots()
        ax0.set(facecolor="black")

        ax0.scatter(
            pos_exp[:, 0],
            pos_exp[:, 1],
            facecolors="orange",
            edgecolors="none",
            s=4,
        )

        ax0.set_xlabel("x (nm)")
        ax0.set_ylabel("y (nm)")
        ax0.set_title("Scatter plot of experimental data")
        ax0.set_box_aspect(1)
        fig0.savefig(
            os.path.join(mask_gen_path, filename[:-5] + "_scatter_exp.png")
        )
        fig0.savefig(
            os.path.join(mask_gen_path, filename[:-5] + "_scatter_exp.pdf")
        )

    # Create 2d histogram of input data
    fig1, ax1 = plt.subplots()

    x0_hist = x0
    y0_hist = y0
    length_hist = length

    bins_x = np.arange(x0_hist, x0_hist + length_hist, binsize_px)
    bins_y = np.arange(y0_hist, y0_hist + length_hist, binsize_px)
    counts, xedges, yedges, *_ = ax1.hist2d(
        x, y, bins=[bins_x, bins_y], cmap="hot"
    )

    ax1.set_box_aspect(1)

    # Blur image
    image_blurred = filters.gaussian(counts, sigma=σ_mask_blur)
    image_blurred = np.rot90(
        image_blurred
    )  # TODO: check why this operation is needed

    # Plot blurred data
    if plot_figures:

        fig2, ax2 = plt.subplots()

        ax2.set_box_aspect(1)

        ax2.set_xlabel("x (nm)")
        ax2.set_ylabel("y (nm)")
        ax2.set_xlim(x0, x0 + length)
        ax2.set_ylim(y0, y0 + length)
        ax2.set_title("Blurred experimental data")

        ax2.imshow(
            image_blurred,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="hot",
        )
        fig2.savefig(
            os.path.join(
                mask_gen_path, filename[:-5] + "_blurred_exp_data.png"
            )
        )
        fig2.savefig(
            os.path.join(
                mask_gen_path, filename[:-5] + "_blurred_exp_data.pdf"
            )
        )

    # Otsu threshold. Pixels above the threshold will be "in" the mask.
    thresh = filters.threshold_otsu(image_blurred)

    mask = image_blurred >= thresh / 3

    # Plot the mask
    if plot_figures:

        fig3, ax3 = plt.subplots()

        ax3.set_box_aspect(1)
        ax3.set_title("Binary mask")
        ax3.imshow(
            mask,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="hot",
        )

        ax3.set_xlabel("x (nm)")
        ax3.set_ylabel("y (nm)")
        ax3.set_xlim(x0, x0 + length)
        ax3.set_ylim(y0, y0 + length)

    # "Zoom" (upsample) the mask to reach the final binsize ("mask_resolution") of the mask
    length_rounded = np.around(length)
    factor = int(binsize_px / mask_resolution_px)
    mask_zoomed = ndi.zoom(np.array(mask, dtype=float), factor)

    # Plot upsampled mask
    if plot_figures:

        fig4, ax4 = plt.subplots()

        ax4.set_box_aspect(1)
        ax4.set_title("Binary mask - upsampled")
        ax4.imshow(
            mask_zoomed,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="hot",
        )

        ax4.set_xlabel("x (nm)")
        ax4.set_ylabel("y (nm)")
        ax4.set_xlim(x0, x0 + length)
        ax4.set_ylim(y0, y0 + length)
        # fig4.savefig(os.path.join(mask_gen_path, filename[:-5] + '_mask_upsampled.png'))

    # Final mask
    mask_final = mask_zoomed > 0.5

    # Upsample the xedges and yedges arrays
    xedges = np.arange(
        xedges[0], xedges[-1] + mask_resolution_px / 2, step=mask_resolution_px
    )
    yedges = np.arange(
        yedges[0], yedges[-1] + mask_resolution_px / 2, step=mask_resolution_px
    )

    # Plot final mask
    if plot_figures:

        fig5, ax5 = plt.subplots()

        ax5.set_box_aspect(1)
        ax5.set_title("Binary mask - final")
        ax5.imshow(
            mask_final,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="hot",
        )

        ax5.set_xlabel("x (nm)")
        ax5.set_ylabel("y (nm)")
        ax5.set_xlim(x0, x0 + length)
        ax5.set_ylim(y0, y0 + length)
        fig5.savefig(
            os.path.join(mask_gen_path, filename[:-5] + "_mask_final.png")
        )
        fig5.savefig(
            os.path.join(mask_gen_path, filename[:-5] + "_mask_final.pdf")
        )

    # Plot overlay of final mask and exp datapoints
    fig5, ax5 = plt.subplots()

    ax5.set_box_aspect(1)
    ax5.set_title("Binary mask - final + scatter")
    ax5.imshow(
        mask_final,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="hot",
    )
    ax5.scatter(
        pos_exp[:, 0],
        pos_exp[:, 1],
        facecolors="orange",
        edgecolors="none",
        s=4,
    )

    ax5.set_xlabel("x (nm)")
    ax5.set_ylabel("y (nm)")
    ax5.set_xlim(x0, x0 + length)
    ax5.set_ylim(y0, y0 + length)
    fig5.savefig(
        os.path.join(
            mask_gen_path, filename[:-5] + "_mask_final_scatter_exp.png"
        )
    )
    fig5.savefig(
        os.path.join(
            mask_gen_path, filename[:-5] + "_mask_final_scatter_exp.pdf"
        )
    )

    # Mask area and density
    frac_of_area_with_molec = mask_final.sum() / mask_final.shape[0] ** 2
    mask_area = mask_final.sum() * (mask_resolution / 1000) ** 2

    observed_density = pos_exp.shape[0] / mask_area

    # Save the mask (format compatible with Luciano's code)
    np.save(os.path.join(mask_gen_path, mask_name + ".npy"), mask_final)
    np.save(
        os.path.join(mask_gen_path, mask_name + "_xedges" + ".npy"), xedges
    )
    np.save(
        os.path.join(mask_gen_path, mask_name + "_yedges" + ".npy"), yedges
    )

    # create config file (compatible to Luciano's code)

    config_mask = configparser.ConfigParser()

    config_mask["params"] = {
        "Date and time": str(datetime.now()),
        "px_size (nm)": px,
        "margin (nm)": margin,
        "bin size (nm)": binsize,
        "gaussian blur sigma (units of bin size)": σ_mask_blur,
        "mask digital resolution (nm)": mask_resolution,
        "Observed density (?m^-2)": observed_density,
        "mask file name": mask_name + ".npy",
        "otsu": thresh,
    }

    with open(os.path.join(mask_gen_path, "params.txt"), "w") as configfile:
        config_mask.write(configfile)

    # Save the mask as a dict for further processing
    mask_dict = {
        "mask": mask_final,
        "xedges": xedges,
        "yedges": yedges,
        "area": mask_area,
    }
    dict_filename = os.path.join(mask_gen_path, "Mask_FOV.pkl")
    with open(dict_filename, "wb+") as f:
        pickle.dump(mask_dict, f)

    return mask_dict


"""
===============================================================================
Density of Exp Points in Mask
===============================================================================
"""


def exp_data_in_mask(
    df, mask_dict, px, output_path, filename, info_input, plot_figures=False
):
    plt.close("all")
    """
    Identifies datapoints that are within a mask and saves them in a new hdf5
    file.


    Parameters
    ----------
    df : DataFrame
        Dataframe containing data from input hdf5.
    mask_dict : dict
        Dictionary containing the mask, the x and y edges of the mask and the
        mask area.
    px : float
        Pixel size in nm..
    output_path : string
        Path to output folder.
    filename : string
        Filename to save output files.
    info_input : dict
        Content of yaml file of input hdf5.
    plot_figures : bool, optional
        If True some output figures will be saved. The default is False.

    Returns
    -------
    df_mask : DataFrame
        Contains only the rows of datapoints that are within the mask.
    mask_dict : dict
        Dictionary containing the mask, the x and y edges of the mask, the
        mask area, the number of exp datapoints in the mask area and the
        resulting density.

    """

    # remove path from filename
    filepath, filename = os.path.split(filename)

    # Create subfolder for mask generation files
    mask_gen_path = os.path.join(output_path, "mask")
    try:
        os.mkdir(mask_gen_path)

    except OSError:
        pass

    # Load mask information
    mask = mask_dict["mask"]
    xedges = mask_dict["xedges"]
    yedges = mask_dict["yedges"]
    area = mask_dict["area"]

    dx = xedges[1] - xedges[0]

    # Exclude datapoints outside of x and yedges (should anyways not be the case in this scenario)
    df_mask = df[
        (df["x"] < xedges[-1])
        & (df["x"] > xedges[0])
        & (df["y"] < yedges[-1])
        & (df["y"] > yedges[0])
    ].copy()

    # Identify in which bin of the mask each datapoint is located
    # convert to an integer version of pos (in multiples of dx)
    df_mask["x_aux"] = (df_mask["x"] - xedges[0]) / dx
    df_mask["y_aux"] = (df_mask["y"] - yedges[0]) / dx

    df_mask["x_ind"] = np.array(
        (np.floor(df_mask["x_aux"]) + np.ceil(df_mask["x_aux"])) / 2, dtype=int
    )
    df_mask["y_ind"] = np.array(
        (np.floor(df_mask["y_aux"]) + np.ceil(df_mask["y_aux"])) / 2, dtype=int
    )

    # convert from (x, y) to (i, j)
    # mirror y indices of datapoints
    i_ind = (
        len(mask) - 10 - df_mask["y_ind"]
    )  # Why do I need to subtract 10???  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    j_ind = np.array(df_mask["x_ind"], dtype=int)

    # Identify if a datapoint is in or outside of the mask
    index = mask[i_ind, j_ind].astype(bool)

    # Keep only the datapoints in the mask
    df_mask = df_mask[index]

    # Drop unneccessary columns
    df_mask = df_mask.drop(columns=["x_aux", "y_aux", "x_ind", "y_ind"])

    # Save hdf5 containing only the locs in the mask
    locs = df_mask.to_records()

    N_locs_FA = len(df_mask)

    info_new = {}
    info_new["N molecules"] = N_locs_FA
    info_new["Area (um^2)"] = float(area)
    info_new["Density FA (positions/micrometer^2)"] = float(N_locs_FA / area)
    info = info_input + [info_new]

    io.save_locs(
        os.path.join(output_path, filename[:-5] + "_mask.hdf5"), locs, info
    )

    # Save the mask dict again with additional info about the density
    mask_dict["N_exp"] = N_locs_FA
    mask_dict["density_exp  (/um^2)"] = N_locs_FA / area

    dict_filename = os.path.join(mask_gen_path, "Mask_FOV.pkl")
    with open(dict_filename, "wb+") as f:
        pickle.dump(mask_dict, f)

    return df_mask, mask_dict


"""
===============================================================================
CSR Simulation inside Mask
===============================================================================
"""


def plot_pos(filename, density, width, height, pos, px):
    """
    Plots a small FOV of the generated CSR positions.

    Parameters
    ----------
    filename : string
        path and filename of the resulting image.
    density : float
        Density in /um^2.
    width : float
        Width of full FOV in pixel.
    height : float
        Height of full FOV in pixel.
    pos : (N,2) array
        contains x and y coordinates.
    px : float
        Pixel Size in nm.

    Returns
    -------
    None.

    """

    mon_color = "#FE4A49"

    # transform from pixel to nm
    width_nm = width * px
    height_nm = height * px
    pos_nm = pos * px

    # transform to molecules per um^2

    fig1, ax1 = plt.subplots()  # monomers
    fig1.suptitle("CSR")

    # monomer centers
    ax1.scatter(pos_nm[:, 0], pos_nm[:, 1], alpha=0.5, color=mon_color)

    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("y (nm)")
    ax1.set_title("Density = " + str(round(density, 2)) + "/$μm^2$")

    ax1.set_box_aspect(1)

    if width_nm > 3000 or height_nm > 3000:
        # Display only a small part of simulated positions
        length = 1000  # nm, length of the display area for the graph

        ax1.set_xlim(width_nm / 2, width_nm / 2 + length)
        ax1.set_ylim(height_nm / 2, height_nm / 2 + length)

    fig1.savefig(filename + ".pdf")
    fig1.savefig(filename + ".png")


def save_pos(output_path, filename, width, height, pos, px, info):
    """
    Creates an hdf5 file for the simulated positions with dummy values for
    frame, lpx and lpy.

    Parameters
    ----------
    output_path : string
        Path to folder where hdf5 will be saved.
    filename : string
        Filename for the hdf5 file.
    width : float
        Width of full FOV in pixel.
    height : float
        Height of full FOV in pixel.
    pos : (N,2) array
        contains x and y coordinates.
    px : float
        Pixel Size in nm.
    info : dict
        Content that will be saved in the yaml file.

    Returns
    -------
    locs : rec_array
        Array that contains frame, x, y, lpx and lpy columns.

    """
    # Save coordinates to csv or Picasso hdf5 file

    frames = np.full(len(pos), int(0))
    x = pos[:, 0]
    y = pos[:, 1]
    lpx = np.full(
        len(pos), 0.001
    )  # Dummy value required for Picasso Render to display points
    lpy = np.full(len(pos), 0.001)

    LOCS_DTYPE = [
        ("frame", "u4"),
        ("x", "f4"),
        ("y", "f4"),
        ("lpx", "f4"),
        ("lpy", "f4"),
    ]

    locs = np.rec.array(
        (frames, x, y, lpx, lpy),
        dtype=LOCS_DTYPE,
    )

    io.save_locs(os.path.join(output_path, filename + ".hdf5"), locs, info)

    return locs


def CSR_rectangle(width, height, density, depth=0, frame=6):
    """
    This function generates a CSR distribution of points with a certain density
    in a rectangular area of size width x height. The rectangle is surrounded
    by an empty frame region containing no points. If the point coordinates
    are later transfered to Picasso Simulate this frame provides enough space
    for the simulated point spread functions.

    Parameters
    ----------
    width : int
        Width (in pixels) of the rechtangle region within the simulate points
        are distributed.
    height : int
        Height (in pixels) of the rechtangle region within the simulate points
        are distributed.
    density : float
        Density of points per square pixel.
    depth : int, optional
        Depth (in nm) of the rectangle region in xy to simulate 3d point
        patterns. The default is 0.
    frame : int, optional
        Thickness (in pixel) of the empty frame region around the rechtangle.
        The default is 6.

    Returns
    -------
    pos : numpy array
        Nx2 array for 2d and Nx3 array for 3d simulations containing the point
        coordinates in units of pixel for x and y and in nm for z. N is
        calculated from the density and the rectangle area.
    total_width: float
        Width of the full FOV including frame.
    total_height
        Height of the full FOV including frame.

    """
    x_min = frame
    y_min = frame
    x_max = width + frame
    y_max = height + frame

    # if (x_max - x_min < 0) or (y_max - y_min < 0):
    #    raise Exception('FOV of size {} x {} pixel is too small for a frame of {} pixel.' .format(round(width,1), round(height,1), frame))

    if depth == 0:  # 2D
        N = int(density * width * height)
        D = 2
    else:  # 3D
        z_min = frame
        z_max = depth - frame
        N = int(density * width * height * (z_max - z_min))
        D = 3

    pos = np.zeros(
        (N, D)
    )  # initialize array of central positions for molecules
    pos[:, 0], pos[:, 1] = [
        np.random.uniform(x_min, x_max, N),
        np.random.uniform(y_min, y_max, N),
    ]
    if D == 3:  # 3D
        pos[:, 2] = np.random.uniform(z_min, z_max, N)

    total_width = int(math.ceil(width + 2 * frame))
    total_height = int(math.ceil(height + 2 * frame))
    return pos, total_width, total_height, N


def CSR_sim_in_mask(
    mask_dict, px, output_path, filename, info_input, plot_figures=False
):
    """
    Simulates a CSR protein distribution within the boundaries of the mask and
    with the experimentally measured density.
    Saves the CSR positions in a hdf5 file readable by Picasso Render.

    Parameters
    ----------
    mask_dict : dict
        Dictionary containing the mask, the x and y edges of the mask, the
        mask area, the number of exp datapoints in the mask area and the
        resulting exp density.
    px : float
        Pixel size in nm..
    output_path : string
        Path to output folder.
    filename : string
        Filename to save output files.
    info_input : dict
        Content of yaml file of input hdf5.
    plot_figures : bool, optional
        If True some output figures will be saved. The default is False.

    Returns
    -------
    df_locs : DataFrame
        Contains positions of the CSR points with dummy values for the frame,
        lpx and lpy column.
    info_unif : dict
        Contains info that is saved in the CSR yaml file.

    """
    plt.close("all")

    # remove path from filename
    filepath, filename = os.path.split(filename)

    # Create subfolder for mask generation files
    mask_gen_path = os.path.join(output_path, "mask")
    try:
        os.mkdir(mask_gen_path)

    except OSError:
        pass

    # Load mask information
    mask = mask_dict["mask"]
    xedges = mask_dict["xedges"]
    yedges = mask_dict["yedges"]
    area = mask_dict["area"]
    density_exp = mask_dict["density_exp  (/um^2)"]

    dx = xedges[1] - xedges[0]  # in pixel

    width_px = xedges[-1] - xedges[0]
    height_px = yedges[-1] - yedges[0]

    density_exp_px = density_exp * px / 1000 * px / 1000

    # Generate a rectangle of CSR positions at the experimental density
    pos_CSR, total_width_px, total_height_px, N_real = CSR_rectangle(
        width_px, height_px, density_exp_px, frame=0
    )

    pos_CSR = pos_CSR + np.array(
        [xedges[0], yedges[0]]
    )  # correct positions of the simulated molecules

    # Exclude datapoints outside of x and yedges (should anyways not be the case in this scenario)
    pos_CSR = pos_CSR[pos_CSR[:, 0] > xedges[0]]
    pos_CSR = pos_CSR[pos_CSR[:, 1] > yedges[0]]
    pos_CSR = pos_CSR[pos_CSR[:, 0] < xedges[-1]]
    pos_CSR = pos_CSR[pos_CSR[:, 1] < yedges[-1]]

    ### Apply mask
    # Identify in which bin of the mask each datapoint is located
    # convert to an integer version of pos (in multiples of dx)
    pos_aux = (pos_CSR - np.array([xedges[0], yedges[0]])) / dx
    pos_rounded = np.array(
        (np.floor(pos_aux) + np.ceil(pos_aux)) / 2, dtype=int
    )

    x_ind = np.array(pos_rounded[:, 0], dtype=int)
    y_ind = np.array(pos_rounded[:, 1], dtype=int)

    # convert from (x, y) to (i, j)
    # mirror y indices of datapoints
    i_ind = (
        len(mask) - 1 - y_ind
    )  # -1 to account for 0 indexing##################################### safer implementation than y_imd.max()!!!!!!!!!!!!!!!!!!!!!!!!!!
    j_ind = x_ind

    # Identify if a datapoint is in or outside of the mask
    index = mask[i_ind, j_ind].astype(bool)

    # Keep only the datapoints in the mask
    pos_CSR_in = pos_CSR[index]

    # Save positions as hdf5
    N_CSR = len(pos_CSR_in)
    info_unif = {}
    info_unif["Generated by"] = "Custom Simulation of positions within FA_area"
    info_unif["Distribution"] = "uniform"
    info_unif["Density exp (positions/micrometer^2)"] = float(density_exp)
    info_unif["N_positions"] = N_CSR
    info_unif["Width"] = info_input[0]["Width"]
    info_unif["Height"] = info_input[0]["Height"]
    info_unif["Frame"] = 0
    info_unif["Pixelsize"] = px

    info_unif["Area (um^2)"] = float(area)
    info_unif["Density sim FA before LE (positions/micrometer^2)"] = float(
        N_CSR / area
    )

    rec_array_locs = save_pos(
        output_path,
        filename[:-5] + "_CSR_mask",
        info_input[0]["Width"],
        info_input[0]["Height"],
        pos_CSR_in,
        px,
        [info_unif],
    )

    # Create data frame for further processing
    df_locs = pd.DataFrame.from_records(rec_array_locs)

    if plot_figures:
        plot_pos(
            os.path.join(mask_gen_path, filename[:-5] + "_CSR_mask"),
            N_CSR / area,
            total_width_px,
            total_height_px,
            pos_CSR,
            px,
        )

    return df_locs, info_unif


def CSR_sim_in_mask_multi_channel(
    channel_ID,
    mask_dict,
    px,
    output_path,
    filename,
    info_input,
    plot_figures=False,
):
    """
    Simulates a CSR protein distribution within the boundaries of the mask and
    with the experimentally measured density for each target protein lited
    in the channel_ID dict.
    Saves the CSR positions in hdf5 files readable by Picasso Render and
    merges them in a multi hdf5 file.

    Parameters
    ----------
    channel_ID : dict
        Keys are protein names, values is integer ID.
    mask_dict : dict
        Dictionary containing the mask, the x and y edges of the mask, the
        mask area. For each protein species it contains the number of exp
        datapoints in the mask area and the resulting exp density.
    px : float
        Pixel size in nm..
    output_path : string
        Path to output folder.
    filename : string
        Filename to save output files.
    info_input : dict
        Content of yaml file of input hdf5.
    plot_figures : bool, optional
        If True some output figures will be saved. The default is False.

    Returns
    -------
    df_CSR_merge : DataFrame
        Contains positions of the CSR points (of all channels togheter, "multi"
        file) with dummy values for the frame, lpx and lpy column.
    info_CSR_multi : dict
        Contains info that is saved in the multi CSR yaml file.

    """

    # create subfolder for hdf5 files of indiviual channels
    filepath, filename = os.path.split(filename)
    CSR_channels_path = os.path.join(filepath, "CSR_channels")
    try:
        os.mkdir(CSR_channels_path)

    except OSError:
        pass
    channel_filename = os.path.join(CSR_channels_path, filename)

    # merge files
    merge = []
    filename_list = []
    info_list = []

    # Create a mask dictionary containing the information for only one of the channels
    mask_dict_single_channel = {}
    mask_dict_single_channel["mask"] = mask_dict["mask"]
    mask_dict_single_channel["xedges"] = mask_dict["xedges"]
    mask_dict_single_channel["yedges"] = mask_dict["yedges"]
    mask_dict_single_channel["area"] = mask_dict["area"]

    for protein in channel_ID:
        protein_ID = channel_ID[protein]

        # Load mask information for specific channel
        mask_dict_single_channel["density_exp  (/um^2)"] = mask_dict[
            "density_exp_" + protein + " (/um^2)"
        ]
        channel_filename_i = channel_filename[:-5] + "_" + protein + ".hdf5"
        # print(channel_filename_i)
        df_CSR_i, info_CSR_i = CSR_sim_in_mask(
            mask_dict_single_channel,
            px,
            CSR_channels_path,
            channel_filename_i,
            info_input,
            plot_figures=plot_figures,
        )
        df_CSR_i["protein"] = np.full(len(df_CSR_i), protein_ID)
        filename_list.append(channel_filename_i[:-5] + "_CSR_mask.hdf5")
        info_list.append(info_CSR_i)

        merge.append(df_CSR_i)

    df_CSR_merge = pd.concat(merge, ignore_index=True)
    df_CSR_merge = df_CSR_merge.sort_values(by=["frame"])
    rec_CSR_merge = df_CSR_merge.to_records(index=False)

    info_0 = info_list[0]

    info_CSR_multi = {
        "Generated by": "Custom Combine Script",
        "Path to combined files": filename_list,
    }

    multi_CSR_filename = os.path.join(
        filepath, filename[:-5] + "_CSR_mask_multi_ID.hdf5"
    )

    io.save_locs(
        multi_CSR_filename, rec_CSR_merge, [info_0] + [info_CSR_multi]
    )

    return df_CSR_merge, [info_0] + [info_CSR_multi]
