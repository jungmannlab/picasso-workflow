#!/usr/bin/env python
"""
Module Name: picasso_outpost.py
Author: Heinrich Grabmayr
Initial Date: March 8, 2024
Description: This is a collection of exploratory DNA-PAINT analysis / picasso
    related functions which if useful should (potentially) be moved into the
    next picasso release. The reasoning to put them here is that it makes
    testing cycles faster.
"""
import logging
import numpy as np

# from numpy.lib.recfunctions import stack_arrays
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LogNorm
import yaml
import os
from aicsimageio import AICSImage

from picasso import io, localize, render, imageprocess, postprocess, lib
from picasso_workflow import util

from scipy.special import gamma as _gamma
from scipy.special import factorial as _factorial
from scipy.optimize import minimize
from scipy import stats
import itertools


logger = logging.getLogger(__name__)


def align_channels(
    channel_locs,
    channel_info,
    max_iterations=5,
    convergence=0.001,
    fiducial_locs=None,
):
    """This is taken from picasso.gui.render.View.align. As the code is not
    modular enough, it is replicated here. Potentially, this could go into
    a non-gui function in picasso.
    Args:
        channel_locs : list of recarray
            the localizations of the different channels
        channel_info : list of dict
            the infos of the different channels
        max_iterations : int
            the maximum number of iterations of alignment
        convergence : float
            convergence criterium when a shift is negligible and thus
            alignment convergence achieved. The value is in pixels.
        fiducial_locs : list of recarray
            the localizations to use as a basis for the alignment. If None,
            the channel_locs are used as fiducials.
    Returns:
        shift : list (len 2-3) of lists (len iterations)
            the shifts in x, y, (z) for each iteration, averaged over
            channels (?)
        cumulative_shift : np array (3, channels, iterations)
            the cumulative shift in the three dimensions, in all channels
            the total shift is the last value (in iterations) fo the cum shift
    """
    shift_x = []
    shift_y = []
    shift_z = []
    all_shift = np.zeros((3, len(channel_locs), max_iterations))

    logger.debug("Aligning datasets")
    if fiducial_locs is None:
        use_fiducials = False
    else:
        use_fiducials = True

    for iteration in range(max_iterations):
        completed = True

        # find shift between channels
        if fiducial_locs is None:
            # assignment by reference. Any changes to fiducial_locs will act on
            # channel_locs and vice versa.
            fiducial_locs = channel_locs
        shift = shift_from_rcc(fiducial_locs, channel_info)
        logger.debug("Shifting channels.")
        temp_shift_x = []
        temp_shift_y = []
        temp_shift_z = []
        for i, locs_ in enumerate(fiducial_locs):
            if (
                np.absolute(shift[0][i]) + np.absolute(shift[1][i])
                > convergence
            ):
                completed = False

            # shift each channel
            locs_.y -= shift[0][i]
            locs_.x -= shift[1][i]

            temp_shift_x.append(shift[1][i])
            temp_shift_y.append(shift[0][i])
            all_shift[0, i, iteration] = shift[1][i]
            all_shift[1, i, iteration] = shift[0][i]

            if len(shift) == 3:
                locs_.z -= shift[2][i]
                temp_shift_z.append(shift[2][i])
                all_shift[2, i, iteration] = shift[2][i]
        shift_x.append(np.mean(temp_shift_x))
        shift_y.append(np.mean(temp_shift_y))
        if len(shift) == 3:
            shift_z.append(np.mean(temp_shift_z))

        cumulative_shift = np.cumsum(all_shift, axis=2)

        # Skip when converged:
        if completed:
            break
    shift = [shift_x, shift_y]
    if shift_z != []:
        shift.append(shift_z)

    # logger.debug(f"calculated shifts:")
    # logger.debug(f"last shift: {str(shift)}")
    # # logger.debug(f'shift_y: {str(shift_y)}')
    # # logger.debug(f'shift_z: {str(shift_z)}')
    # logger.debug(f"all shift: {str(all_shift)}")
    # logger.debug(f"cumulative_shift: {str(cumulative_shift)}")
    # logger.debug(f"cumulative_shift shape: {cumulative_shift.shape}")
    # logger.debug(f"all_shift shape: {all_shift.shape}")

    # if fiducial_locs were separately given, shift channel_locs
    if use_fiducials:  # channel_locs != fiducial_locs:
        for i, locs_ in enumerate(channel_locs):
            # logger.debug(f"shifting x by {str(cumulative_shift[0, i, -1])}")
            locs_.x -= cumulative_shift[0, i, -1]
            # logger.debug(f"shifting y by {str(cumulative_shift[1, i, -1])}")
            locs_.y -= cumulative_shift[1, i, -1]
            if len(shift) == 3:
                locs_.z -= cumulative_shift[2, i, -1]
    return shift, cumulative_shift


def plot_shift(shifts, cum_shifts, filepath):
    """Plot the sifts generated by align_channels
    Args:
        shifts : list of 1D array
            the shifts in x, y, and potentially z dimensions
        cum_shifts : 3 D array
            cumulative shifts (dimension, channel, iteration)
        filepath : str
            the filepath to save the plot
    """
    fig, ax = plt.subplots(nrows=1 + len(shifts), sharex=True)
    # ax[0].suptitle("Shift")
    for i, (shift, dim) in enumerate(zip(shifts, ["x", "y", "z"])):
        ax[0].plot(shift, "o-", label=f"{dim} shift")
        ax[1 + i].plot(cum_shifts[i, :, :])
        ax[1 + i].set_ylabel(f"{dim}-shift (Px)")
    ax[0].set_ylabel("Mean Shift (Px)")
    ax[-1].set_xlabel("Iteration")
    fig.set_size_inches((8, 8))
    ax[0].legend(loc="best")
    fig.savefig(filepath)


def shift_from_rcc(channel_locs, channel_info):
    """
    Used by align. Estimates image shifts based on whole images'
    rcc.

    Args:
        channel_locs : list of recarray
            the localizations of the different channels
        channel_info : list of dict
            the infos of the different channels

    Returns:
        shifts : tuple
            the channel shifts shape (2,) or (3,) (if z coordinate present)
    """
    n_channels = len(channel_locs)
    images = []
    logger.debug("Rendering localizations.")
    # render each channel and save it in images
    for i, (locs_, info_) in enumerate(zip(channel_locs, channel_info)):
        _, image = render.render(locs_, info_, blur_method="smooth")
        images.append(image)
    n_pairs = int(n_channels * (n_channels - 1) / 2)
    logger.debug(f"Correlating {n_pairs} image pairs.")
    return imageprocess.rcc(images)


def convert_zeiss_file(filepath_czi, filepath_raw, info=None):
    """Convert Zeiss .czi file into a picasso-readable .raw file.
    Args:
        filepath_csi : str
            the filepath to the .czi file to load
        filepath_raw : str
            the filepath to the .raw file to write
        info : dict, default None
            the metadata to make the raw file picasso-readable.
            If None is given, dummy values are entered.
            Necesary keys:
                'Byte Order', 'Camera', 'Micro-Manager Metadata'
    """
    img = AICSImage(filepath_czi)

    with open(filepath_raw, "wb") as f:
        img.get_image_data().squeeze().tofile(f)

    if info is None:
        info = {"Byte Order": "<", "Camera": "FusionBT"}
        info["File"] = filepath_raw
        info["Height"] = img.get_image_data().shape[-2]
        info["Width"] = img.get_image_data().shape[-1]
        info["Frames"] = img.get_image_data().shape[0]
        info["Data Type"] = img.get_image_data().dtype.name
        info["Micro-Manager Metadata"] = {
            "FusionBT-ReadoutMode": 1,
            "Filter": 561,
        }

    filepath_info = os.path.splitext(filepath_raw)[0] + ".yaml"

    with open(filepath_info, "w") as f:
        yaml.dump(info, f)


#############################################################################
# for plotting single spots in analyse.AutoPicasso.
#############################################################################


def get_spots(movie, identifications, box, camera_info):
    spots = _cut_spots(movie, identifications, box)
    return localize._to_photons(spots, camera_info)


def _cut_spots(movie, ids, box):
    N = len(ids.frame)
    spots = np.zeros((N, box, box), dtype=movie.dtype)
    spots = _cut_spots_byrandomframe(
        movie, ids.frame, ids.x, ids.y, box, spots
    )
    return spots


def _cut_spots_byrandomframe(movie, ids_frame, ids_x, ids_y, box, spots):
    """Cuts the spots out of a movie by non-sorted frames.

    Args:
        movie : AbstractPicassoMovie (t, x, y)
            the image data
        ids_frame, ids_x, ids_y : 1D array (k)
            spot positions in the image data. Length: number of spots
            identified
        box : uneven int
            the cut spot box size
        spots : 3D array (k, box, box)
            the cut spots
    Returns:
        spots : as above
            the image-data filled spots
    """
    r = int(box / 2)
    for j, (fr, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        frame = movie[fr]
        spots[j] = frame[yc - r : yc + r + 1, xc - r : xc + r + 1]
    return spots


def normalize_spot(spot, maxval=255, dtype=np.uint8):
    # logger.debug('spot input: ' + str(spot))
    sp = spot - np.min(spot)
    imgmax = np.max(sp)
    imgmax = 1 if imgmax == 0 else imgmax
    sp = sp.astype(np.float32) / imgmax * maxval
    # logger.debug('spot output: ' + str(sp.astype(dtype)))
    return sp.astype(dtype)


def spinna_temp(parameters_filename):
    """While SPINNA is under development (and the paper being written)
    it is not integrated in the regular picasso package. Here, the
    corresponding module is being loaded.
    This function runs a spinna batch analysis from file.

    Returns:
        result_dir : str
            folder containing the results
        fp_summary : str
            the filepath of the summary csv file
        fp_fig : list of str
            filepaths of the NND figures
    """
    from picasso_workflow.spinna_main import _spinna_batch_analysis

    result_dir, fp_summary, fp_fig = _spinna_batch_analysis(
        parameters_filename
    )
    print("result_dir", result_dir)
    print("fp_summary", fp_summary)
    print("fp_fig", fp_fig)
    return result_dir, fp_summary, fp_fig


def spinna_sgl_temp(parameters):
    """While SPINNA is under development (and the paper being written)
    it is not integrated in the regular picasso package. Here, the
    corresponding module is being loaded.
    This function directly runs one spinna simulation.

    Args:
        parameters : dict with keys:
            structures, label_unc, le, mask_dict, width, height, depth,
            random_rot_mode, exp_data, sim_repeats, fit_NND_bin, fit_NND_maxdist,
            N_structures, save_filename, asynch, targets, apply_mask, nn_plotted,
            result_dir
    Returns:
        result_dir : str
            folder containing the results
        fp_summary : str
            the filepath of the summary csv file
        fp_fig : list of str
            filepaths of the NND figures
    """
    from picasso_workflow.spinna_main import single_spinna_run

    result_dir, fp_fig = single_spinna_run(**parameters)
    return result_dir, fp_fig


def generate_N_structures(structures, N_total, res_factor, save=""):
    from picasso_workflow.spinna import generate_N_structures

    return generate_N_structures(
        structures,
        N_total,
        res_factor,
        save="",
    )


########################################################################
# Begin Log likelihood CSR estimation
########################################################################


def estimate_density_from_neighbordists(
    nn_dists, rho_init, kmin=1, rho_bound_factor=10
):
    """For one point with k nearest neighbor distances (all assumed from
    a CSR distribution), do a maximum likelihood estimation for the
    density.
    Args:
        nn_dists : array, len k - or 2D array: (N, k)
            the k nearest neighbor distances (of N spots)
        rho_init : float
            the initial estimation of density
    Returns:
        mle_rho : float
            the maximum likelihood estimate for the local density
            based on the nearest neighbor distances
    """
    bounds = [
        (rho_init / rho_bound_factor, rho_init * rho_bound_factor)
    ]  # rho must be positive
    mle_rho = minimize(
        minimization_loglike,
        x0=[rho_init],
        args=(nn_dists, kmin),
        bounds=bounds,
        # tol=1e-8, options={'maxiter': 1e5}, method='Powell'
        # options={'maxiter': 1e5}, method='L-BFGS-B'
        # method='BFGS'#,
        # options={'maxiter': 1e5, 'gtol': 1e-6, 'eps': 1e-9},
        method="L-BFGS-B",
        options={
            "disp": None,
            "maxcor": 10,
            "ftol": 2e-15,
            "gtol": 1e-15,
            "eps": 1e-15,
            "maxfun": 150,
            "maxiter": 150,
            "iprint": -1,
            "maxls": 100,
            "finite_diff_rel_step": None,
        },
    )
    # print(mle_rho)
    return mle_rho.x[0], mle_rho


def minimization_loglike(rho, nndist_observed, kmin=1):
    """The minimization function for nndist loglikelihood fun
    based on k-th nearest neighbor CSR distributions
    Args:
        rho : list, len 1
            the estimated density
        nndist_observed : array, len k - or 2D array: (N, k)
            the k nearest neighbor distances (of N spots)
    Returns:
        loglike : float
            the log likelihood of finding the observed neighbor distances
            in the model of CSR and given rho
    """
    return -nndist_loglikelihood_csr(nndist_observed, rho[0], kmin)


def nndist_loglikelihood_csr(nndist_observed, rho, kmin=1):
    """get the Log-Likelihood of observed nearest neighbors assuming
    a CSR distribution with density rho.
    Args:
        nndist_observed : array, len k - or 2D array: (N, k)
            the k nearest neighbor distances (of one or N spots)
        rho : float
            the density
    Returns:
        log_like : float
            the log likelihood of all distances observed being drawn
            from CSR
    """
    log_like = 0
    # print("nndist_obs shape", nndist_observed.shape)
    for i, dist in enumerate(nndist_observed):
        k = i + kmin
        # print(f"evaluating csr of {len(dist)} spots at k={k}, with rho={rho}")
        # assert False
        prob = nndistribution_from_csr(dist, k, rho)
        log_like += np.sum(np.log(prob))
    return log_like


def nndistribution_from_csr(r, k, rho, d=2):
    """The CSR Nearest Neighbor distribution of finding the k-th nearest
    neighbor at r. with the spatial randomness covering d dimensions
    Args:
        r : float or array of floats
            the distance(s) to evaluate the probability density at
        k : int
            evaluation of the k-th nearest neighbor
        rho : float
            the density
        d : int
            the dimensionality of the problem
    Returns:
        p : same as r
            the probability density of k-th nearest neighbor at r
    """
    # if k != 1:
    #     print(f'evaluating CSR not at k=1 but k={k}')

    # def gaussian_pdf(x, mean, std):
    #     factor = (1 / (np.sqrt(2 * np.pi) * std))
    #     return factor * np.exp(-0.5 * ((x - mean) / std) ** 2)

    # pdf = gaussian_pdf(r, 4, k*rho*4)
    # # pdf = gaussian_pdf(r, 4+k*rho, .8)
    # return pdf #/ np.sum(pdf)
    lam = rho * np.pi ** (d / 2) / _gamma(d / 2 + 1)
    factor = d / _factorial(k - 1) * lam**k * r ** (d * k - 1)
    dist = factor * np.exp(-lam * r**d)
    return dist  # / np.sum(dist)


########################################################################
# End Log likelihood CSR estimation
########################################################################


########################################################################
# Start Molecular Interaction Patterns (Joschka)
########################################################################


def DBSCAN_analysis(clusters_csv):
    """Calculates barcodes and weights (by cluster area) for further
    DBSCAN data analysis.

    Parameters
    ----------
    clusters_csv : str or pd.DataFrame
        Path to csv file with DBSCAN results

    Returns
    -------
    barcodes : np.array
        Array of shape (N, 6) with binary barcodes for each of
        N DBSCAN clusters
    weights : np.array
        Array of shape (N,) with weights for each of N DBSCAN
        clusters
    """

    if isinstance(clusters_csv, str):
        clusters = pd.read_csv(clusters_csv)  # DBSCAN data
    elif isinstance(clusters_csv, pd.DataFrame):
        clusters = clusters_csv
    else:
        raise NotImplementedError("Type of clusters_csv not implemented.")

    columns = [
        "N_MHC-I_per_cluster",
        "N_MHC-II_per_cluster",
        "N_CD86_per_cluster",
        "N_CD80_per_cluster",
        "N_PDL1_per_cluster",
        "N_PDL2_per_cluster",
        "area (nm^2)",
    ]
    clusters = clusters[columns]

    areas = clusters.values[:, -1]  # this is for weights later

    # find the all or none (binary) barcodes
    clusters = clusters[columns[:-1]]
    idx = np.where(clusters.values > 0)
    barcodes = clusters.values.copy()
    barcodes[idx] = 1

    weights = areas

    return barcodes, weights


def DBSCAN_analysis_pd(clusters_csv, channel_tags):
    """Calculates barcodes and weights (by cluster area) for further
    DBSCAN data analysis.

    Parameters
    ----------
    clusters_csv : str or pd.DataFrame
        Path to csv file with DBSCAN results
    channel_tags : list of str
        the names of the channels, in the correct order

    Returns
    -------
    barcodes_df : pd.DataFrame, columns 'barcode', 'weight'
        Array of shape (N, 6) with binary barcodes for each of
        N DBSCAN clusters
    barcodes_agg : pd.DataFrame, columns are barcodes
        Descriptive aggregation of barcodes_df. with indexes
        count, mean, std, 25%, 50%, 75%
    barcode_map : pd.DataFrame
        index: arange
        cols: 'barcode': string with the binary barcode
            {target}_per_cluster:
    """

    if isinstance(clusters_csv, str):
        clusters = pd.read_csv(clusters_csv)  # DBSCAN data
    elif isinstance(clusters_csv, pd.DataFrame):
        clusters = clusters_csv
    else:
        raise NotImplementedError("Type of clusters_csv not implemented.")

    # per_cluster_cols = [
    #     col for col in clusters.columns
    #     if col.endswith("_per_cluster") and col.startswith('N_')
    # ]
    targets = channel_tags  # [col[: -len("_per_cluster")] for col in per_cluster_cols]
    # sort per cluster cols
    per_cluster_cols = [f"N_{target}_per_cluster" for target in targets]

    barcode_df = pd.DataFrame(
        index=clusters.index,
        columns=["barcode", "area (nm^2)"] + per_cluster_cols,
    )

    def decimal_to_binary(decimal, digits):
        binstr = bin(decimal)
        if len(binstr) - 2 < digits:
            binstr = (
                binstr[:2] + "0" * (digits - (len(binstr) - 2)) + binstr[2:]
            )
        return binstr

    def assemble_barcode(df, cols):
        barcode = np.zeros(len(df.index), dtype=np.int32)
        for i, col in enumerate(cols[::-1]):
            barcode += 2**i * (df[col] > 0)
        return barcode

    barcode_df["barcode"] = np.vectorize(decimal_to_binary)(
        assemble_barcode(clusters, per_cluster_cols), len(targets)
    )
    barcode_df["area (nm^2)"] = clusters["area (nm^2)"]
    for col in per_cluster_cols:
        barcode_df.loc[:, col] = clusters[col]

    barcodes_agg = barcode_df.groupby("barcode").describe()
    print(barcodes_agg.columns)

    barcode_map = pd.DataFrame(
        index=np.arange(2 ** len(targets)),
        columns=["barcode"] + targets,
    )

    barcode_map["barcode"] = np.vectorize(decimal_to_binary)(
        barcode_map.index, len(targets)
    )
    for i, col in enumerate(targets):
        barcode_map[col] = (barcode_map["barcode"].str[2 + i] == "1").astype(
            np.int32
        )

    return barcode_df, barcodes_agg, barcode_map


def _do_dbscan_molint(
    result_folder,
    fp_out_base,
    df_mask,
    info,
    pixelsize,
    epsilon_nm,
    minpts,
    sigma_linker,
    thresh_type,
    cell_name,
    channel_map,
    it=0,
):
    from picasso_workflow.dbscan_molint import dbscan

    filepaths = {}

    if thresh_type == "area":
        # Analysis will also be performed seperatetly for clusters
        # larger or equal than
        # area thresh and clusters smaller than thresh
        thresh = 10000  # nm^2
    elif thresh_type == "density":
        # Analysis will also be performed seperatetly for clusters
        # with densities
        # larger or equal than density thresh
        # area thresh and clusters smaller than thresh
        thresh = 100  # molecules / um^2
        # Change unit to molecules / nm^2
        thresh = thresh / 1000 / 1000

    epsilon_px = epsilon_nm / pixelsize
    sigma_linker_px = sigma_linker / pixelsize

    # DBSCAN on exp data
    new_info = {
        "Generated by": "picasso-workflow: DBSCAN-MOLECULAR INTERACTIONS",
        "epsilon": epsilon_px,
        "minpts": minpts,
        # 'Number of clusters"
    }
    info.append(new_info)
    (
        db_locs_rec,
        db_locs_rec_protein_colorcoding,
        db_cluster_props_rec,
        db_locs_df,
        db_cluster_props_df,
    ) = dbscan.dbscan_f(df_mask, epsilon_px, minpts, sigma_linker_px)

    # save locs in dbscan cluster with colorcoding = dbcluster ID
    dbscan_fp = os.path.join(
        result_folder,
        f"dbscan_{epsilon_nm:.0f}_{minpts}_{it}.hdf5",
    )
    io.save_locs(dbscan_fp, db_locs_rec, info)
    filepaths["fp_dbscan_color-cluster"] = dbscan_fp

    # save locs in dbscan cluster with colorcoding = protein ID
    dbscan_fp = os.path.join(
        result_folder,
        f"dbscan_{epsilon_nm:.0f}_{minpts}_{it}" + "_protein_colorcode.hdf5",
    )
    io.save_locs(dbscan_fp, db_locs_rec_protein_colorcoding, info)
    filepaths["fp_dbscan_color-protein"] = dbscan_fp

    # save properties of dbscan clusters
    # (analygously to DBSCAN output in Picasso)
    dbclusters_fp = os.path.join(
        result_folder,
        f"dbclusters_{epsilon_nm:.0f}_{minpts}_{it}.hdf5",
    )
    # print(db_cluster_props_rec)
    # print(db_cluster_props_rec.dtype)
    # THE FOLLOWING LINE DOES NOT WORK BECAUSE db_cluster_props_rec
    # does not have x, y, lpx, or lpy. fails picasso sanity checks.
    # io.save_locs(dbclusters_fp, db_cluster_props_rec, info)
    db_cluster_props_df.to_hdf(dbclusters_fp, key="props")
    filepaths["fp_dbclusters"] = dbclusters_fp

    from picasso_workflow.dbscan_molint import output_metrics

    """
    ===============================================================================
    Output of all clusters in one cell
    ===============================================================================
    """

    # output for each cluster = [N_per_cluster, area, circularity,
    #                            N_CD80, ... % CD80, ...]

    # Calculate and save output metrics for all clusters in the cell
    (
        cluster_filename,
        cluster_large_filename,
        cluster_small_filename,
        db_cluster_output,
    ) = output_metrics.output_cell(
        channel_map,
        db_locs_df,
        db_cluster_props_df,
        fp_out_base,
        pixelsize,
        epsilon_nm,
        minpts,
        thresh,
        thresh_type,
        cell_name,
    )

    filepaths["fpoutput_all_clusters"] = cluster_filename
    filepaths["fpoutput_large_clusters"] = cluster_large_filename
    filepaths["fpoutput_small_clusters"] = cluster_small_filename
    # stimulation_cluster_exp_dict[cell_name] = db_cluster_output
    # stimulation_cluster_exp_large_dict[cell_name] = db_cluster_output_large
    # stimulation_cluster_exp_small_dict[cell_name] = db_cluster_output_small

    # perform Rafal's analysis (binary barcode)
    barcodes, weights = DBSCAN_analysis(db_cluster_output)
    filepaths["fp_binary_barcode"] = os.path.join(
        result_folder, "binary_barcode.txt"
    )
    np.savetxt(filepaths["fp_binary_barcode"], barcodes)
    filepaths["fp_binary_barcode_weights"] = os.path.join(
        result_folder, "binary_barcode_weights.txt"
    )
    np.savetxt(filepaths["fp_binary_barcode_weights"], weights)

    # perform adapteation of Rafal's analysis
    channel_map_r = {v: k for k, v in channel_map.items()}
    targets = [channel_map_r[i] for i in sorted(channel_map_r.keys())]
    (barcode_df, barcode_agg, barcode_map) = DBSCAN_analysis_pd(
        db_cluster_output, targets
    )
    filepaths["fp_barcode"] = os.path.join(result_folder, f"barcode_{it}.xlsx")
    barcode_df.to_excel(filepaths["fp_barcode"])
    filepaths["fp_barcode_agg"] = os.path.join(
        result_folder, f"barcode_described_{it}.xlsx"
    )
    barcode_agg.to_excel(filepaths["fp_barcode_agg"])
    filepaths["fp_barcode_map"] = os.path.join(
        result_folder, f"barcode_map_{it}.xlsx"
    )
    barcode_map.to_excel(filepaths["fp_barcode_map"])

    # number of nonclustered
    cluster_info = {
        "n_input_locs": len(df_mask.index),
        "n_clustered_locs": len(db_locs_df.index),
        "n_nonclustered_locs": len(df_mask.index) - len(db_locs_df.index),
        "n_clusters": len(db_cluster_props_df.index),
    }
    fp_cluster_info = os.path.join(result_folder, f"cluster_info_{it}.yaml")
    filepaths["fp_cluster_info"] = fp_cluster_info
    with open(fp_cluster_info, "w") as f:
        yaml.dump(cluster_info, f)

    # plot results
    fig, ax = plt.subplots(nrows=3, sharex=True)
    # barplot: number of clusters
    ax[0].bar(
        np.arange(len(barcode_agg.index)),
        barcode_agg[("area (nm^2)", "count")],
    )
    ax[0].set_ylabel("# clusters found")

    fig.set_size_inches((13, 9))
    filepaths["fp_fig"] = os.path.join(result_folder, "barcodes.png")
    fig.savefig(filepaths["fp_fig"])

    # boxplot: area distribution of clusters
    # sns.boxplot(data=barcode_df, x='barcode', y='area (nm^2)', ax=ax[1])
    dflist = [
        subdf.values
        for idx, subdf in barcode_df.groupby("barcode")["area (nm^2)"]
    ]
    ax[1].boxplot(
        dflist,
        positions=np.arange(len(barcode_agg.index)),
        showfliers=False,
    )
    ax[1].set_ylabel("area per cluster (nm^2)")
    # boxplots: number of targets per cluster, for each target

    fig.set_size_inches((13, 9))
    filepaths["fp_fig"] = os.path.join(result_folder, "barcodes.png")
    fig.savefig(filepaths["fp_fig"])

    bxwidth = 1 / (len(targets) + 2)
    bxpos_init = np.arange(len(barcode_agg.index)) - bxwidth * len(targets) / 2

    target_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]
    legend_handles = []
    import matplotlib.lines as mlines

    for i, tgt in enumerate(targets):
        col = f"N_{tgt}_per_cluster"
        dflist = [
            subdf.values if idx[2 + i] == "1" else np.array([])
            for idx, subdf in barcode_df.groupby("barcode")[col]
        ]
        # remove values (zeros) if target is not in barcode
        lineprops = {"color": target_colors[i]}
        ax[2].boxplot(
            dflist,
            positions=bxpos_init + i * bxwidth,
            widths=bxwidth,
            showfliers=False,
            boxprops=lineprops,
            whiskerprops=lineprops,
            medianprops=lineprops,
            capprops=lineprops,
        )
        line = mlines.Line2D([], [], color=target_colors[i], label=tgt)
        legend_handles.append(line)
    ax[2].set_ylabel("# targets per cluster")
    ax[2].set_xticks(np.arange(len(barcode_agg.index)))
    xtilabels = [ti[2:] for ti in barcode_agg.index]
    ax[2].set_xticklabels(xtilabels, rotation=90)
    # plot separator lines
    xpos = np.arange(len(barcode_agg.index) - 1) + 0.5
    ylims = ax[2].get_ylim()
    for x in xpos:
        ax[2].plot([x, x], ylims, color="gray")
    ax[2].legend(handles=legend_handles)
    fig.set_size_inches((15, 9))
    filepaths["fp_fig"] = os.path.join(result_folder, f"barcodes_{it}.png")
    fig.savefig(filepaths["fp_fig"])
    """
    ===============================================================================
    mean output of one cell
    ===============================================================================
    """
    # output for each cluster = [N_per_cluster, area, circularity,
    #                            N_CD80, ... % CD80, ...]
    # output for complete cell: [
    #   N_in_cell, N_in_clusters, N_out_clusters, N_per_cluster_mean,
    #   N_per_cluster_CI,, area_mean, area_CI, circularity_mean,
    #   circularity_CI,, N_CD80,, N_CD80_in_clusters, N_CD86_out_clusters,
    #   ... , N_CD80_mean, N_CD80_CI, ...., %_CD80_mean, %_CD80_CI, ....]

    # Calculate and save mean of output metrics of all clusters in the cell
    # + some output metrics specific to the whole cell
    (mean_filename, mean_large_filename, mean_small_filename) = (
        output_metrics.output_cell_mean(
            channel_map,
            df_mask,
            db_locs_df,
            db_cluster_output,
            fp_out_base,
            pixelsize,
            epsilon_nm,
            minpts,
            thresh,
            thresh_type,
            cell_name,
        )
    )

    filepaths["fpoutput_mean_all"] = mean_filename
    filepaths["fpoutput_mean_large"] = mean_large_filename
    filepaths["fpoutput_mean_small"] = mean_small_filename
    # stimulation_exp_dict[cell_name] = db_cell_output
    # stimulation_exp_large_dict[cell_name] = db_cell_output_large
    # stimulation_exp_small_dict[cell_name] = db_cell_output_small

    return filepaths


def degree_of_clustering(
    cluster_info_exp, cluster_info_csr, origin_colors, folder
):
    # plot number of clustered vs non-clustered locs
    data = {
        "exp": [
            cluster_info_exp["n_clustered_locs"],
            cluster_info_exp["n_nonclustered_locs"],
        ],
        "csr": [
            cluster_info_csr["n_clustered_locs"],
            cluster_info_csr["n_nonclustered_locs"],
        ],
    }
    fp_fig_dog = os.path.join(folder, "degree_of_clustering.png")
    _ = _plot_degreeofclustering(
        data, origin_colors, fp_fig_dog, ylabel="# locs per cell"
    )

    # plot fraction of clustered vs non-clustered locs
    data_fract = {
        "exp": [
            np.array(cluster_info_exp["n_clustered_locs"])
            / (
                np.array(cluster_info_exp["n_clustered_locs"])
                + np.array(cluster_info_exp["n_nonclustered_locs"])
            ),
            np.array(cluster_info_exp["n_nonclustered_locs"])
            / (
                np.array(cluster_info_exp["n_clustered_locs"])
                + np.array(cluster_info_exp["n_nonclustered_locs"])
            ),
        ],
        "csr": [
            np.array(cluster_info_csr["n_clustered_locs"])
            / (
                np.array(cluster_info_csr["n_clustered_locs"])
                + np.array(cluster_info_csr["n_nonclustered_locs"])
            ),
            np.array(cluster_info_csr["n_nonclustered_locs"])
            / (
                np.array(cluster_info_csr["n_clustered_locs"])
                + np.array(cluster_info_csr["n_nonclustered_locs"])
            ),
        ],
    }
    fp_fig_tractdog = os.path.join(folder, "fracdegree_of_clustering.png")
    _ = _plot_degreeofclustering(
        data_fract,
        origin_colors,
        fp_fig_tractdog,
        ylabel="fraction of locs per cell",
    )

    return [fp_fig_dog, fp_fig_tractdog]


def _plot_degreeofclustering(
    data, origin_colors, fp_fig, ylabel="fraction of locs per cell"
):
    """
    Plot the degree of clustering of experimental versus simulated
    data in violin plots, including stripplots of the data.
    Args:
        data: dict of array
            the underlying data to plot (numer/fraction of clustered
            or unclustered locs for each cell)
            keys: 'exp' and 'csr'
        origin_colors : list of str
            the colors of 'exp', and 'csr' data, respectively
        fp_fig : str
            the path to save the figure at
    Returns:
        t_stats : xyz
            the results of the t_test between experimental and csr
            data for clustered and non-clustered data comparison
        p_values : array len 2
            the p_values of exp and csr being drawn from the same
            distribution for clustered and non-clustered data
    """
    categories = ["clustered", "non-clustered"]

    bxwidth = 1 / (len(origin_colors) + 2)
    bxpos_init = (
        np.arange(len(categories))
        - bxwidth * len(origin_colors) / 2
        + bxwidth / 2
    )
    legend_handles = []

    fig, ax = plt.subplots()
    for i, org in enumerate(["exp", "csr"]):
        parts = ax.violinplot(
            data[org],
            positions=bxpos_init + i * bxwidth,
            widths=bxwidth,
            showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(origin_colors[i])
            pc.set_edgecolor(origin_colors[i])
        util.stripplot(
            data[org],
            bxpos_init + i * bxwidth,
            bxwidth,
            ax,
            origin_colors[i],
            alpha=0.5,
        )
        line = mlines.Line2D([], [], color=origin_colors[i], label=org)
        legend_handles.append(line)
    # test for significance
    ylims = ax.get_ylim()
    p_values = np.ones(2)
    t_stats = np.ones(2)
    for i, (n_exp, n_csr) in enumerate(zip(data["exp"], data["csr"])):
        t_stats[i], p_values[i] = stats.ttest_ind(n_exp, n_csr)
        if p_values[i] < 1e-3:
            siglabel = "p < 0.001"
        elif p_values[i] < 1e-2:
            siglabel = "p < 0.01"
        else:
            siglabel = "n.s."
        ax.text(
            i,
            0.8 * ylims[1],
            siglabel,
            fontsize=14,
            color="k",
            horizontalalignment="center",
            verticalalignment="center",
        )
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)  # , rotation=90)
    ax.set_title("degree of clustering")
    ax.legend(handles=legend_handles)
    fig.set_size_inches((8, 5))
    fig.savefig(fp_fig)
    plt.close(fig)
    return t_stats, p_values


def _plot_and_compare_barcodes(
    pivot_table,
    origin_colors,
    targets,
    ttest_pvalue_max,
    population_threshold,
    cellfraction_threshold,
    fp_fig,
    title="",
    ylabel="",
):
    """Plot the comparison of barcodes between experiment and simulation,
    and perform a t-test to evaluate whether the distributions are
    different.
    Args:
        pivot_table : pd.DataFrame
            index: barcodes (str, 0b...)
            columns: multiindex, first index: origin (['exp', 'csr'])
        origin_colors : list of str
            the colors to use for the two conditions
        targets : list of str
            the protein targets
        ttest_pvalue_max : float
            the pvalue above which no significance is attributed to
            the difference of exp and csr
        population_threshold : float
            the relative population a barcode needs to be significant
            (e.g. 1% of all clusters need to have a barcode for it
            to pop up)
        population_threshold : float, between 0 and 1
            the fraction of cells that need to have this barcode at least once.
        fp_fig : str
            the filepath to save the figure at
        title : str
            the title addition for the plot
    Returns:
        significant_barcodes : list of str
            the barcodes that evaluated as significantly changed between
            exp and csr
        p_values : list of float
            the p_values of the t-test for all barcodes
    """
    # plot distribution of number of barcodes
    fig, ax = plt.subplots(nrows=1, sharex=True)

    legend_handles = []
    bxwidth = 1 / (len(origin_colors) + 2)
    bxpos_init = (
        np.arange(len(pivot_table.index))
        - bxwidth * len(origin_colors) / 2
        + bxwidth / 2
    )
    all_occurrence_lists = {}
    for i, org in enumerate(["exp", "csr"]):
        dflist = [row[org].values for bc, row in pivot_table.iterrows()]
        all_occurrence_lists[org] = dflist
        # remove values (zeros) if target is not in barcode
        lineprops = {"color": origin_colors[i]}
        ax.boxplot(
            dflist,
            positions=bxpos_init + i * bxwidth,
            widths=bxwidth,
            showfliers=False,
            boxprops=lineprops,
            whiskerprops=lineprops,
            medianprops=lineprops,
            capprops=lineprops,
        )
        # print(bplot.keys())
        # for patch in bplot['boxes']:
        #     patch.set_edgecolor(target_colors[i])
        line = mlines.Line2D([], [], color=origin_colors[i], label=org)
        legend_handles.append(line)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}; barcoding: " + "-".join(targets))
    ax.set_xticks(np.arange(len(pivot_table.index)))
    xtilabels = [ti[2:] for ti in pivot_table.index]
    ax.set_xticklabels(xtilabels, rotation=90)
    # # plot separator lines
    # xpos = np.arange(len(barcode_numbers.index) - 1) + .5
    ylims = ax.get_ylim()
    ax.set_ylim([0, ylims[1]])
    # for x in xpos:
    #     ax.plot([x, x], ylims, color='gray')
    ax.legend(handles=legend_handles)

    # test for significant difference in the number of barcodes found
    # between exp and csr
    p_values = np.ones(len(pivot_table.index))
    t_stats = np.ones(len(pivot_table.index))
    for i, (n_exp, n_csr) in enumerate(
        zip(all_occurrence_lists["exp"], all_occurrence_lists["csr"])
    ):
        t_stats[i], p_values[i] = stats.ttest_ind(n_exp, n_csr)

    significant_barcodes_idx = np.argwhere(
        p_values < ttest_pvalue_max
    ).flatten()
    # print(significant_barcodes_idx)

    # select for barcodes that have a relevant population
    fraction_barcodes_exp = np.array(
        [sum(occ) for occ in all_occurrence_lists["exp"]], dtype=np.float64
    )
    fraction_barcodes_exp /= np.sum(fraction_barcodes_exp)
    # print(fraction_barcodes_exp)
    relevant_barcodes_idx = np.argwhere(
        fraction_barcodes_exp > population_threshold
    ).flatten()

    # select for barcodes that occur in a given fraciton of cells at
    # least once
    cell_fraction_barcodes = np.array(
        [sum(occ > 0) / len(occ) for occ in all_occurrence_lists["exp"]],
        dtype=np.float64,
    )
    enough_cells_have_barcodes_idx = np.argwhere(
        cell_fraction_barcodes > cellfraction_threshold
    ).flatten()
    # print(relevant_barcodes_idx)
    significant_barcodes_idx = [
        idx
        for idx in significant_barcodes_idx
        if (
            idx in relevant_barcodes_idx
            and idx in enough_cells_have_barcodes_idx
        )
    ]

    for pos in significant_barcodes_idx:
        if p_values[pos] < 1e-3:
            siglabel = "p < 0.001"
        elif p_values[pos] < 1e-2:
            siglabel = "p < 0.01"
        elif p_values[pos] < ttest_pvalue_max:
            siglabel = f"p < {ttest_pvalue_max:.2f}"
        else:
            siglabel = "n.s."
        ax.text(
            pos,
            0.8 * ylims[1],
            siglabel,
            fontsize=10,
            color="k",
            horizontalalignment="center",
            verticalalignment="center",
            rotation=90,
        )

    significant_barcodes = [
        pivot_table.index[i] for i in significant_barcodes_idx
    ]

    fig.set_size_inches((15, 6))
    fig.savefig(fp_fig)
    plt.close(fig)

    return significant_barcodes, p_values


def _plot_and_compare_ntargets_in_barcodes(
    df, bc, origin_colors, targets, fp_fig
):
    """For a significant cluster, plot the distribution of
    number of targets for exp and csr cases, and determine
    whether they are stastistically differnt
    Args:
        df : DataFrame
            the list of all clusters with this barcode
        bc : str
            the barcode ('0b...')
        origin_colors : list of str
            the colors for exp and csr
        targets : list of str
            the protein target names
        fp_fig : str
            the filepath to save the figure as
    """
    fig, ax = plt.subplots()
    bxwidth = 1 / (len(origin_colors) + 2)
    bxpos_init = (
        np.arange(len(targets))
        - bxwidth * len(origin_colors) / 2
        + bxwidth / 2
    )
    legend_handles = []

    pts = {}
    ntgt_data = {}
    for i, tgt in enumerate(targets):
        pivot_table = pd.pivot_table(
            df[["origin", "name", "iter", f"N_{tgt}_per_cluster"]],
            index="origin",
            columns=["name", "iter"],
            values=f"N_{tgt}_per_cluster",
            aggfunc="mean",
            fill_value=np.nan,
        )
        # average over 'iter'
        pivot_table = pivot_table.T.groupby(level=["name"]).mean().T
        # fp = os.path.join(os.path.split(fp_fig)[0], f"bc{bc}-{tgt}.xlsx")
        # print(fp)
        # pivot_table.to_excel(fp)
        pts[tgt] = pivot_table
        pts_exp = pts[tgt].loc["exp", :].values.flatten()
        pts_csr = pts[tgt].loc["csr", :].values.flatten()
        ntgt_data[tgt] = {
            "exp": pts_exp[~np.isnan(pts_exp)],
            "csr": pts_csr[~np.isnan(pts_csr)],
        }
    for i, org in enumerate(["exp", "csr"]):
        # subdf = df.loc[df["origin"] == org]
        dflist = [
            (
                # subdf.groupby()[f"N_{tgt}_per_cluster"]
                # pts[tgt].loc[org, :].values
                ntgt_data[tgt][org]
                if bc[2 + k] == "1"
                else np.array([np.nan] * 3)
            )
            for k, tgt in enumerate(targets)
        ]
        parts = ax.violinplot(
            dflist,
            positions=bxpos_init + i * bxwidth,
            widths=bxwidth,
            showmedians=True,
            showextrema=False,
            # quantiles=[.25, .75]
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(origin_colors[i])
            pc.set_edgecolor(origin_colors[i])
        util.stripplot(
            dflist,
            bxpos_init + i * bxwidth,
            bxwidth,
            ax,
            origin_colors[i],
            alpha=0.2,
        )
        # lineprops = {"color": origin_colors[i]}
        # ax.boxplot(
        #     dflist,
        #     positions=bxpos_init + i * bxwidth,
        #     widths=bxwidth,
        #     showfliers=False,
        #     boxprops=lineprops,
        #     whiskerprops=lineprops,
        #     medianprops=lineprops,
        #     capprops=lineprops,
        # )
        line = mlines.Line2D([], [], color=origin_colors[i], label=org)
        legend_handles.append(line)
    ax.set_ylabel("# targets per cluster")
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels(targets, rotation=90)
    ax.legend(handles=legend_handles)

    # evaluate statistical difference
    p_values = np.ones(len(targets))
    t_stats = np.ones(len(targets))
    ylims = ax.get_ylim()
    ax.set_ylim([0, ylims[1]])
    for i, tgt in enumerate(targets):
        if bc[2 + i] != "1":
            continue
        # exp_data = df.loc[df["origin"] == "exp", f"N_{tgt}_per_cluster"]
        # csr_data = df.loc[df["origin"] == "csr", f"N_{tgt}_per_cluster"]
        t_stats[i], p_values[i] = stats.ttest_ind(
            ntgt_data[tgt]["exp"], ntgt_data[tgt]["csr"]
        )
        if p_values[i] < 1e-3:
            siglabel = "p < 0.001"
        elif p_values[i] < 1e-2:
            siglabel = "p < 0.01"
        else:
            siglabel = "n.s."
        # print(f'{bc} target {targets[i]} p-values: ', p_values[i])
        ax.text(
            i,
            0.8 * ylims[1],
            siglabel,
            fontsize=12,
            color="k",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # n_exp = np.sum(df["origin"] == "exp")
    # n_csr = np.sum(df["origin"] == "csr")
    # ax.set_title(
    #     f"Significantly altered barcode {bc[2:]}; "
    #     + f"data points exp {n_exp}, csr {n_csr}")
    ax.set_title(f"Significantly altered barcode {bc[2:]}")
    fig.set_size_inches((8, 5))
    fig.savefig(fp_fig)
    plt.close(fig)


def _plot_interaction_graph(
    node_sizes: np.ndarray,
    edge_sizes: np.ndarray,
    target_colors: list,
    targets: list,
):
    """Create an interaction graph plot to show both density of proteins
    (node sizes), and interaction strength (edge sizes)
    Args:
        node_sizes : np array (N,)
            the size of the nodes
        edge_sizes : np array (N, N)
            the interaction strength between edges, including self
    """
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    N = len(node_sizes)
    # Create a figure and a subplot
    fig, ax = plt.subplots()
    ax.set_xlim([-1.75, 1.75])
    ax.set_ylim([-1.75, 1.75])
    ax.set_aspect("equal")
    # Calculate the positions of the nodes
    theta = -np.linspace(0, 2 * np.pi, N, endpoint=False)
    theta += np.pi * 2 / 3  # shift by 60 deg to match Joschka's positions
    x = np.cos(theta)
    y = np.sin(theta)
    xnode = np.cos(theta + np.pi * 1 / 18)
    ynode = np.sin(theta + np.pi * 1 / 18)
    # Draw the edges
    for i in range(N):
        for j in range(N):
            if i != j:
                # ax.plot(
                #     [x[i], x[j]], [y[i], y[j]],
                #     color='black', linewidth=edge_sizes[i][j])
                # pass
                offset_dist = np.abs(edge_sizes[i][j]) + np.abs(
                    edge_sizes[j][i]
                )
                offset_dir = np.array([(y[j] - y[i]), (x[j] - x[i])])
                offset_dir = offset_dir / np.sqrt(np.sum(offset_dir**2))
                offset = offset_dist * offset_dir
                trans = ax.transData
                coords_start = np.array([x[i], y[i]])
                coords_end = np.array([x[j], y[j]])
                coords_start_pt = trans.transform(coords_start)
                coords_end_pt = trans.transform(coords_end)
                coords_start = trans.inverted().transform(
                    coords_start_pt + offset
                )
                coords_end = trans.inverted().transform(coords_end_pt + offset)
                x_coords = [coords_start[0], coords_end[0]]
                y_coords = [coords_start[1], coords_end[1]]
                lineprops = dict(
                    color=target_colors[i],
                    linewidth=np.abs(edge_sizes[i][j]),
                    solid_capstyle="round",
                )
                if edge_sizes[j][i] < 0:
                    lineprops["linestyle"] = ":"
                line = Line2D(x_coords, y_coords, **lineprops)
                ax.add_line(line)
            else:
                # Draw a circular arrow for self-interaction
                start_angle = int(180 / np.pi * theta[i])
                aA = start_angle + 0
                aB = start_angle + 225
                ax.annotate(
                    "",
                    xy=(1.1 * x[i], 1.1 * y[i]),
                    xytext=(1.65 * x[i], 1.65 * y[i]),
                    arrowprops=dict(
                        arrowstyle="<-",
                        # connectionstyle="arc,rad=.7,angleA=0,angleB=225",
                        connectionstyle=f"angle3,angleA={aA},angleB={aB}",
                        linewidth=np.abs(edge_sizes[i][i]),
                        color=target_colors[i],
                    ),
                )
    # Draw the nodes
    for i in range(N):
        start_angle = 0
        end_angle = 360
        radius = np.sqrt(node_sizes[i])
        wedge = mpatches.Wedge(
            (x[i], y[i]),
            radius,
            start_angle,
            end_angle,
            facecolor=target_colors[i],
        )
        ax.add_patch(wedge)
        ax.text(
            1.5 * xnode[i],
            1.5 * ynode[i],
            targets[i],
            fontsize=14,
            color=target_colors[i],
            horizontalalignment="center",
            verticalalignment="center",
        )
    # ax.scatter(x, y, s=node_sizes, color='blue')
    ax.axis("off")

    return fig, ax


########################################################################
# End Molecular Interaction Patterns (Joschka)
########################################################################


########################################################################
# Start Labeling Efficiency Workflow Modules
########################################################################


def pick_gold(locs, info, diameter=2, std_range=1.4, mean_rmsd=0.4):
    """
    Searches picks similar to Gold clusters.

    Focuses on the number of locs and their root mean square
    displacement from center of mass. Std is defined in Tools
    Settings Dialog.

    Args:
        diameter : float
            the pick similar diameter
        std_range, mean_rmsd : float
            the pick similar parameters identifying gold
    Returns:
        similar : list of [x, y] position pairs
            the positions (picks) of gold beads

    Raises
    ------
    NotImplementedError
        If pick shape is rectangle
    """
    d = diameter

    maxframe = info[0]["Frames"]
    maxheight = info[0]["Height"]
    maxwidth = info[0]["Width"]
    r = d / 2
    d2 = d**2

    # extract n_locs and rmsd from current picks
    (locs_temp, r, _, _, block_starts, block_ends, K, L) = (
        postprocess.get_index_blocks(locs, info, r)
    )

    # calculate min and max n_locs and rmsd for picking similar
    mean_n_locs = maxframe
    std_n_locs = 0.25 * mean_n_locs
    std_rmsd = 0.25 * mean_n_locs
    min_n_locs = mean_n_locs - std_range * std_n_locs
    max_n_locs = mean_n_locs + std_range * std_n_locs
    min_rmsd = mean_rmsd - std_range * std_rmsd
    max_rmsd = mean_rmsd + std_range * std_rmsd

    # x, y coordinates of found regions:
    x_similar = np.array([])
    y_similar = np.array([])

    # preparations for grid search
    x_range = np.arange(d / 2, maxwidth, np.sqrt(3) * d / 2)
    y_range_base = np.arange(d / 2, maxheight - d / 2, d)
    y_range_shift = y_range_base + d / 2

    locs_x = locs_temp.x
    locs_y = locs_temp.y
    locs_xy = np.stack((locs_x, locs_y))
    x_r = np.uint64(x_range / r)
    y_r1 = np.uint64(y_range_shift / r)
    y_r2 = np.uint64(y_range_base / r)
    # print(locs_xy)
    # print("min_n_locs, max_n_locs, min_rmsd, max_rmsd")
    # print(min_n_locs, max_n_locs, min_rmsd, max_rmsd)
    # pick similar
    x_similar, y_similar = postprocess.pick_similar(
        x_range,
        y_range_shift,
        y_range_base,
        min_n_locs,
        max_n_locs,
        min_rmsd,
        max_rmsd,
        x_r,
        y_r1,
        y_r2,
        locs_xy,
        block_starts,
        block_ends,
        K,
        L,
        x_similar,
        y_similar,
        r,
        d2,
    )
    # add picks
    similar = list(zip(x_similar, y_similar))
    return similar


def index_locs(locs, info, pick_diameter):
    """
    Indexes localizations from a given channel in a grid with grid
    size equal to the pick radius.
    """
    d = pick_diameter
    size = d / 2
    index_blocks = postprocess.get_index_blocks(locs, info, size)
    return index_blocks


def get_block_locs_at(x, y, index_blocks, return_indices=False):
    """Copied from picasso.postprocess.get_block_locs_at.
    But the block indices are needed as well.
    """
    locs, size, _, _, block_starts, block_ends, K, L = index_blocks
    x_index = np.uint32(x / size)
    y_index = np.uint32(y / size)
    indices = []
    for k in range(y_index - 1, y_index + 2):
        if 0 <= k < K:
            for li in range(x_index - 1, x_index + 2):
                if 0 <= li < L:
                    indices.append(
                        list(range(block_starts[k, li], block_ends[k, li]))
                    )
    indices = list(itertools.chain(*indices))
    if return_indices:
        return locs[indices], np.array(indices)
    else:
        return locs[indices]


def locs_at(x, y, locs, r, return_indices=False):
    """Returns localizations at position (x, y) within radius r.

    Parameters
    ----------
    x : float
        x-coordinate of the position.
    y : float
        y-coordinate of the position.
    locs : np.rec.array
        Localizations list.
    r : float
        Radius.

    Returns
    -------
    picked_locs : np.rec.array
        Localizations at position.
    """

    is_picked = lib.is_loc_at(x, y, locs, r)
    picked_locs = locs[is_picked]
    if return_indices:
        return picked_locs, is_picked
    else:
        return picked_locs


def picked_locs(
    locs, info, _centers, pick_diameter, add_group=True, return_nonpicked=False
):
    """
    Returns picked localizations in the specified channel.

    Parameters
    ----------
    channel : int
        Channel of locs to be processed
    add_group : boolean (default=True)
        True if group id should be added to locs. Each pick will be
        assigned a different id
    return_nonpicked : bool
        whether to return the non-picked locs

    Returns:
        all_picked_locs : np.recarray
            locs within pick_diameter around _centers, linked to
            common centers by field 'group'
        # all_picked_locs : list of np.recarray
        #     locs within pick_diameter around _centers, linked to
        #     common centers by field 'group'
        non_picked_locs : np.recarray
            locs that have not been picked.
    """

    picked_locs = []
    is_not_picked = []
    d = pick_diameter
    r = d / 2
    index_blocks = index_locs(locs, info, d)
    # print('index blocks: ', index_blocks)
    for i, pick in enumerate(_centers):
        x, y = pick
        block_locs, block_indices = get_block_locs_at(
            x, y, index_blocks, return_indices=True
        )
        # print(f'block locs: {block_locs}')

        group_locs, is_picked = locs_at(
            x, y, block_locs, r, return_indices=True
        )
        # logger.debug(block_indices)
        # logger.debug(is_picked)
        # logger.debug(is_picked.shape)
        is_not_picked.append(block_indices[~is_picked])
        # print(f'grouplocs: {group_locs}')
        if add_group:
            group = i * np.ones(len(group_locs), dtype=np.int32)
            group_locs = lib.append_to_rec(group_locs, group, "group")
        group_locs.sort(kind="mergesort", order="frame")
        picked_locs.append(group_locs)

    all_picked_locs = np.lib.recfunctions.stack_arrays(
        picked_locs, asrecarray=True, usemask=False
    )
    # all_picked_locs = picked_locs

    if return_nonpicked:
        mask = np.isin(
            locs[["frame", "x", "y", "photons"]],
            all_picked_locs[["frame", "x", "y", "photons"]],
        )
        non_picked_locs = locs[~mask]
        return all_picked_locs, non_picked_locs
    else:
        return all_picked_locs


def _undrift_from_picked_coordinate(info, picked_locs, coordinate):
    """
    Calculates drift in a given coordinate.

    Parameters
    ----------
    channel : int
        Channel where locs are being undrifted
    picked_locs : list
        List of np.recarrays with locs for each pick
    coordinate : str
        Spatial coordinate where drift is to be found

    Returns
    -------
    np.array
        Contains average drift across picks for all frames
    """

    n_picks = len(picked_locs)
    n_frames = info[0]["Frames"]

    # Drift per pick per frame
    drift = np.empty((n_picks, n_frames))
    drift.fill(np.nan)

    # Remove center of mass offset
    for i, locs in enumerate(picked_locs):
        coordinates = getattr(locs, coordinate)
        drift[i, locs.frame] = coordinates - np.mean(coordinates)

    # Mean drift over picks
    drift_mean = np.nanmean(drift, 0)
    # Square deviation of each pick's drift to mean drift along frames
    sd = (drift - drift_mean) ** 2
    # Mean of square deviation for each pick
    msd = np.nanmean(sd, 1)
    # New mean drift over picks
    # where each pick is weighted according to its msd
    nan_mask = np.isnan(drift)
    drift = np.ma.MaskedArray(drift, mask=nan_mask)
    drift_mean = np.ma.average(drift, axis=0, weights=1 / msd)
    drift_mean = drift_mean.filled(np.nan)

    # Linear interpolation for frames without localizations
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    nans, nonzero = nan_helper(drift_mean)
    drift_mean[nans] = np.interp(
        nonzero(nans), nonzero(~nans), drift_mean[~nans]
    )

    return drift_mean


def _undrift_from_picked(locs, info, picked_locs):
    """
    Undrifts in x and y based on picked locs in a given channel.
    Parameters
    ----------
    channel : int
        Channel to be undrifted
    """
    drift_x = _undrift_from_picked_coordinate(info, picked_locs, "x")
    drift_y = _undrift_from_picked_coordinate(info, picked_locs, "y")

    locs.x -= drift_x[locs.frame]
    locs.y -= drift_y[locs.frame]

    return locs, info, (drift_x, drift_y)


########################################################################
# End Labeling Efficiency Workflow Modules
########################################################################


def plot_1dhist(locs, field, fig, ax):
    data = locs[field]
    data = data[np.isfinite(data)]
    bins = lib.calculate_optimal_bins(data, 1000)
    # Prepare the figure
    fig.suptitle(field)
    ax.hist(data, bins, rwidth=1, linewidth=0)
    data_range = data.ptp()
    ax.set_xlim([bins[0] - 0.05 * data_range, data.max() + 0.05 * data_range])


def plot_2dhist(locs, field_x, field_y, fig, ax):
    x = locs[field_x]
    y = locs[field_y]
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    # Start hist2 version
    bins_x = lib.calculate_optimal_bins(x, 1000)
    bins_y = lib.calculate_optimal_bins(y, 1000)
    counts, x_edges, y_edges, image = ax.hist2d(
        x, y, bins=[bins_x, bins_y], norm=LogNorm()
    )
    x_range = x.ptp()
    ax.set_xlim([bins_x[0] - 0.05 * x_range, x.max() + 0.05 * x_range])
    y_range = y.ptp()
    ax.set_ylim([bins_y[0] - 0.05 * y_range, y.max() + 0.05 * y_range])
    fig.colorbar(image, ax=ax)
    ax.grid(False)
    ax.get_xaxis().set_label_text(field_x)
    ax.get_yaxis().set_label_text(field_y)
