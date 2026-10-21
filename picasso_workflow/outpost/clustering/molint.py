"""DBSCAN molecular-interaction (molint) analysis (from picasso_outpost).

See the package docstring. `keep` -- barcode / interaction-graph / clustering
metrics + plots, coupled to the molint report format.
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats

from picasso_workflow import util
from picasso_workflow._picasso_api import io


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
    targets = channel_tags
    # [col[: -len("_per_cluster")] for col in per_cluster_cols]
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
    # print(barcodes_agg.columns)

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
    from picasso_workflow.outpost.clustering.dbscan_molint import dbscan

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

    from picasso_workflow.outpost.clustering.dbscan_molint import (
        output_metrics,
    )

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
    barcode_df, barcode_agg, barcode_map = DBSCAN_analysis_pd(
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
    mean_filename, mean_large_filename, mean_small_filename = (
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
    """Plot the degree of clustering of experimental vs simulated data.

    Uses violin plots overlaid with strip plots of the data.

    Parameters
    ----------
    data : dict of array
        The underlying data to plot (number/fraction of clustered or
        unclustered locs per cell), with keys ``'exp'`` and ``'csr'``.
    origin_colors : list of str
        The colors of the ``'exp'`` and ``'csr'`` data, respectively.
    fp_fig : str
        The path to save the figure at.
    ylabel : str, optional
        The y-axis label. Default is ``"fraction of locs per cell"``.

    Returns
    -------
    t_stats : numpy.ndarray
        The t-test statistics between experimental and CSR data, for the
        clustered and non-clustered comparisons.
    p_values : numpy.ndarray
        Length-2 array of p-values for exp vs csr being drawn from the same
        distribution (clustered and non-clustered).
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
    """Plot and compare barcodes between experiment and simulation.

    Performs a t-test to evaluate whether the distributions differ.

    Parameters
    ----------
    pivot_table : pd.DataFrame
        Index: barcodes (str, ``0b...``); columns: a multi-index whose first
        level is the origin (``['exp', 'csr']``).
    origin_colors : list of str
        The colors to use for the two conditions.
    targets : list of str
        The protein targets.
    ttest_pvalue_max : float
        The p-value above which no significance is attributed to the exp-vs-csr
        difference.
    population_threshold : float
        The relative population a barcode needs to be significant (e.g. 1% of
        all clusters must carry a barcode for it to appear).
    cellfraction_threshold : float
        The fraction of cells (0-1) that must carry the barcode at least once.
    fp_fig : str
        The filepath to save the figure at.
    title : str, optional
        Title addition for the plot.
    ylabel : str, optional
        The y-axis label.

    Returns
    -------
    significant_barcodes : list of str
        The barcodes evaluated as significantly changed between exp and csr.
    p_values : list of float
        The t-test p-values for all barcodes.
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
    """Plot the per-target count distribution for a significant barcode.

    Compares the experimental and CSR cases and tests whether they differ
    statistically.

    Parameters
    ----------
    df : pd.DataFrame
        All clusters carrying this barcode.
    bc : str
        The barcode (``'0b...'``).
    origin_colors : list of str
        The colors for the exp and csr cases.
    targets : list of str
        The protein target names.
    fp_fig : str
        The filepath to save the figure as.
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
    """Create an interaction-graph plot of protein density and interaction.

    Node sizes encode protein density and edge sizes encode interaction
    strength.

    Parameters
    ----------
    node_sizes : np.ndarray
        The node sizes, shape ``(N,)``.
    edge_sizes : np.ndarray
        The interaction strength between nodes (including self), shape
        ``(N, N)``.
    target_colors : list
        Per-target colors.
    targets : list
        The protein target names.
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
