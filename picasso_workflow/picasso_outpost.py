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
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
from aicsimageio import AICSImage

from picasso import localize, render, imageprocess

from scipy.special import gamma as _gamma
from scipy.special import factorial as _factorial
from scipy.optimize import minimize


logger = logging.getLogger(__name__)


def align_channels(
    channel_locs, channel_info, max_iterations=5, convergence=0.001
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

    for iteration in range(max_iterations):
        completed = True

        # find shift between channels
        shift = shift_from_rcc(channel_locs, channel_info)
        logger.debug("Shifting channels.")
        temp_shift_x = []
        temp_shift_y = []
        temp_shift_z = []
        for i, locs_ in enumerate(channel_locs):
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
# Start DBSCAN analysis for Molecular Interaction Patterns
########################################################################
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os, h5py
# from matplotlib.ticker import FormatStrFormatter
# ALL_BARCODES = ['{:0{}b}'.format(i, 6) for i in range(64)][1:]

# # auxiliary functions for searching and loading files

# def get_valid_folders(base, folder_names):
#     folder_names = [
#         folder
#         for folder in folder_names
#         if not folder.startswith('.')
#     ]
#     folder_names = [
#         folder
#         for folder in folder_names
#         if os.path.isdir(os.path.join(base, folder))
#     ]
#     try:
#         folder_names.remove('Dataframe_Backup')
#     except:
#         pass
#     return folder_names

# def get_files(path):
#     dirs = os.listdir(path)

#     all_csvs = [file for file in dirs if file.endswith('db-cell-clusters_35_3.csv')]
#     all_csvs = [file for file in all_csvs if '_bc_' not in file] # mhc2 replacement needs to be excluded
#     csvs_csr = [file for file in all_csvs if 'CSR' in file]
#     csvs_cell = [file for file in all_csvs if not 'CSR' in file]
#     csv_csr1 = [file for file in csvs_csr if 'rep_1_' in file][0]
#     csv_cell = csvs_cell[0]

#     all_hdf5s = [file for file in dirs if file.endswith('_multi_ID.hdf5')]
#     hdf5_csr = [file for file in all_hdf5s if (('CSR' in file) & ('rep_1_' in file))][0]
#     hdf5_cell = [file for file in all_hdf5s if file.endswith('_multi_ID.hdf5')][0]

#     return csv_cell, csv_csr1, hdf5_cell, hdf5_csr


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


# def analyze_all_cells(path, title=None, savename=None):
#     """Runs the binary code analysis across all cells in the
#     given path.

#     Parameters
#     ----------
#     path : str
#         Path to the folder of the given cell type
#     title : str, optional
#         Title of the plot produced. If None, no title is shown.
#     savename : str, optional
#         Path for saving the final plot. If None, the plot is
#         displayed but not saved.
#     """

#     barcodes_cell_ = []
#     barcodes_csr_ = []
#     weights_cell_ = []
#     weights_csr_ = []

#     # extract the names of the folders for each cell
#     cells = os.listdir(path)
#     cells = get_valid_folders(path, cells)
#     cells = [_ for _ in cells if 'Cell' in _]
#     cells.sort(key=(lambda x: int(x[5:]))) # sort based on the cell number
#     for cell in cells:
#         try:
#             results_path = os.path.join(path, cell, 'results')

#             # get names of the files to run the analysis
#             clusters_cell_name, clusters_csr_name, _, _ = get_files(results_path)
#             clusters_name = os.path.join(results_path, clusters_cell_name)
#             csr_clusters_name = os.path.join(results_path, clusters_csr_name)

#             # get the barcodes and weights for the cell
#             barcodes_cell, weights_cell = DBSCAN_analysis(
#                 clusters_name, None, weigh_func=None,
#             )
#             # get the barcodes and wegihts for the csr
#             barcodes_csr, weights_csr = DBSCAN_analysis(
#                 csr_clusters_name, None, weigh_func=None,
#             )

#             barcodes_cell_.append(barcodes_cell)
#             barcodes_csr_.append(barcodes_csr)
#             weights_cell_.append(weights_cell)
#             weights_csr_.append(weights_csr)
#         except:
#             pass

#     # take the mean across the cell
#     res_cell = np.zeros((len(cells), len(ALL_BARCODES))) # contains counts for each cell
#     res_csr = np.zeros((len(cells), len(ALL_BARCODES))) # contains counts for each csr simulation
#     count = 0
#     for (barcodes_cell, barcodes_csr, weights_cell, weights_csr) in zip(
#         barcodes_cell_, barcodes_csr_, weights_cell_, weights_csr_
#     ):
#         # get the weighted counts for cell
#         b = [''.join(_.astype(str)) for _ in barcodes_cell]
#         for weight, barcode in zip(weights_cell, b):
#             idx = ALL_BARCODES.index(barcode)
#             res_cell[count, idx] += weight
#         res_cell[count, :] /= res_cell[count, :].sum()

#         # get the weighted counts for csr
#         b = [''.join(_.astype(str)) for _ in barcodes_csr]
#         for weight, barcode in zip(weights_csr, b):
#             idx = ALL_BARCODES.index(barcode)
#             res_csr[count, idx] += weight
#         res_csr[count, :] /= res_csr[count, :].sum()

#         count += 1

#     # plot
#     x = np.arange(len(ALL_BARCODES))
#     width = 0.4
#     fig = plt.figure(figsize=(10, 4), constrained_layout=True)

#     # take mean and error
#     cell_mean = res_cell.mean(axis=0)
#     cell_std = res_cell.std(axis=0)
#     cell_err = 1.96 * cell_std / np.sqrt(len(cells)) # 95% confidence interval
#     csr_mean = res_csr.mean(axis=0)
#     csr_std = res_csr.std(axis=0)
#     csr_err = 1.96 * csr_std / np.sqrt(len(cells)) # 95% confidence interval

#     # save as txt
#     np.savetxt(f"{savename}_cell_mean.txt", cell_mean)
#     np.savetxt(f"{savename}_cell_err.txt", cell_err)
#     np.savetxt(f"{savename}_csr_mean.txt", csr_mean)
#     np.savetxt(f"{savename}_csr_err.txt", csr_err)

#     # frequency bar plot
#     plt.bar(x-width/2, cell_mean, yerr=cell_err, width=width, edgecolor='black', facecolor='lightgray', label="Cell")
#     plt.bar(x+width/2, csr_mean, yerr=csr_err, width=width, edgecolor='black', facecolor='dimgrey', label="CSR")

#     plt.xticks(np.arange(63), labels=ALL_BARCODES, rotation=90, fontsize=8)
#     plt.ylabel("Weighted counts", fontsize=12)
#     plt.xlabel("Barcodes", fontsize=12)
#     plt.legend()
#     fig.axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#     if title is not None:
#         plt.title(title, fontsize=20)

#     if savename is not None:
#         savename = os.path.splitext(savename)[0]
#         plt.savefig(savename + ".png", dpi=300, transparent=True)
#         plt.savefig(savename + ".svg")
#         plt.close()
#     else:
#         plt.show()

# # Mean across cells of the given type
# # the subfolders in path are named 'Cell 1, Cell 2, etc', each of which has the subfolder 'results'
# # where the DBSCAN analysis results are summarized as .csv files
# path = r"Z:\users\hellmeier\DC_Atlas_2\2024_03_16 B16F10 PDL1 T123R\B16F10_6h_PDL1_T123R\B16F10_6h_PDL1_T123R\6h stimulation"

# title = f"B16F10 PD-L1 T123R 6h stimulation"
# savename = f"plots/B16F10_PDL1_T123R_6h_stimulation.png"
# # savename = None
# analyze_all_cells(path, title=title, savename=savename)
########################################################################
# End DBSCAN analysis for Molecular Interaction Patterns
########################################################################
