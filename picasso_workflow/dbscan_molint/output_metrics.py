# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:20:29 2022

@author: reinhardt
"""

import pandas as pd
import numpy as np
import sys


def check_barcode(cluster_output, barcode, channel_ID):
    # dataframe with columns corresponding to the protein channels
    # True if the existence of the protein matches the target barcode, otherwise False
    check_barcode_df = pd.DataFrame()
    for protein in barcode:
        barcode_target = barcode[protein]  # 0 or 1

        N_exp = cluster_output["N_" + protein + "_per_cluster"]
        idx_1 = np.where(N_exp.values > 0)
        barcode_exp = N_exp.copy()
        barcode_exp.loc[idx_1] = 1

        check_barcode_df[protein] = barcode_exp == barcode_target

    # The cluster matches the target barcode if the values in all columns of
    # check_barcode_df are true for the respoective row.
    check_barcode = check_barcode_df.all(axis=1)  # row-wise comparison

    return cluster_output[check_barcode]


def output_cell(
    channel_ID,
    db_input_locs_df,
    input_props_df,
    input_mask_filename,
    px,
    epsilon_nm,
    minpts,
    thresh,
    thresh_type,
    cell_name,
    barcode=None,
):
    # output for each cluster = [N_per_cluster, area, density, convex_hull, circularity, N_CD80, ... % CD80, ...]
    cluster_output = pd.DataFrame()
    cluster_output["N_per_cluster"] = input_props_df["n"]
    cluster_output["area (nm^2)"] = input_props_df["area"] * px * px
    cluster_output["density (/nm^2)"] = input_props_df["density"] / px / px
    cluster_output["convex_area (nm^2)"] = (
        input_props_df["convex_area"] * px * px
    )
    cluster_output["circularity"] = input_props_df["convex_circularity"]

    group_min = db_input_locs_df["group"].min()
    group_max = db_input_locs_df["group"].max()

    db_N_proteins = db_input_locs_df.groupby(["protein", "group"]).size()

    for protein in channel_ID:
        protein_ID = channel_ID[protein]
        try:
            cluster_output["N_" + protein + "_per_cluster"] = (
                db_N_proteins.loc[protein_ID].reindex(
                    range(group_min, group_max + 1), fill_value=0
                )
            )
        except (
            KeyError
        ):  # There is no single cluster containing the current protein
            cluster_output["N_" + protein + "_per_cluster"] = np.full(
                len(range(group_min, group_max + 1)), 0
            )

    for protein in channel_ID:
        protein_ID = channel_ID[protein]
        cluster_output["%_" + protein + "_per_cluster"] = (
            cluster_output["N_" + protein + "_per_cluster"]
            / cluster_output["N_per_cluster"]
            * 100
        )

    if barcode != None:
        if len(barcode) != len(channel_ID):
            raise Exception(
                "Barcode length does not match number of channels!"
            )
            sys.exit(0)
        else:
            cluster_output = check_barcode(cluster_output, barcode, channel_ID)

    # Add cell_key as column to the dataframes
    cluster_output.insert(loc=0, column="Cell_ID", value=cell_name)

    cluster_filename = input_mask_filename.replace(
        ".hdf5", "_db-cell-clusters_%s_%d.csv" % (str(epsilon_nm), minpts)
    )
    cluster_output.to_csv(cluster_filename)

    # Save the file containing only clusters above the threshold
    if thresh_type == "area":
        cluster_output_large = cluster_output[
            cluster_output["area (nm^2)"] >= thresh
        ]
    if thresh_type == "density":
        cluster_output_large = cluster_output[
            cluster_output["density (/nm^2)"] >= thresh
        ]
    cluster_large_filename = input_mask_filename.replace(
        ".hdf5",
        "_db-cell-clusters_%s_%d_above_threshold.csv"
        % (str(epsilon_nm), minpts),
    )
    cluster_output_large.insert(loc=0, column="Cell_ID", value=cell_name)
    cluster_output_large.to_csv(cluster_large_filename)

    # below the threshold
    if thresh_type == "area":
        cluster_output_small = cluster_output[
            cluster_output["area (nm^2)"] < thresh
        ]
    if thresh_type == "density":
        cluster_output_small = cluster_output[
            cluster_output["density (/nm^2)"] < thresh
        ]
    cluster_small_filename = input_mask_filename.replace(
        ".hdf5",
        "_db-cell-clusters_%s_%d_below_threshold.csv"
        % (str(epsilon_nm), minpts),
    )
    cluster_output_small.insert(loc=0, column="Cell_ID", value=cell_name)
    cluster_output_small.to_csv(cluster_small_filename)

    return (
        cluster_filename,
        cluster_large_filename,
        cluster_small_filename,
        cluster_output,
    )


def _output_cell_mean(
    columns,
    channel_ID,
    input_locs_df,
    db_input_locs_df,
    cluster_output,
    input_mask_filename,
    epsilon_nm,
    minpts,
    cell,
):
    """#  FAILS IF DATAFRAME EMPTY
    cell_column = cluster_output['Cell_ID']
    print('cell_names', cell_column)
    cell = cell_column.iloc[0]
    print('cell name', cell)


    if not (np.array(cell_column) == np.full(len(cell_column), cell)).all():
        raise Exception("Multiple cell_IDs in one dataframe refering to only one cell!")
        sys.exit()
    """

    mean_output = pd.DataFrame(columns=columns, index=[cell])

    # Number of clusters that were found by DBSCAN
    mean_output["N_clusters"] = len(cluster_output)

    # Number of proteins in and outside of clusters for all channel together
    mean_output["N_in_cell"] = len(input_locs_df)
    mean_output["N_in_clusters"] = len(db_input_locs_df)
    # N_out_cluster = len(df.loc[df['group'] == -1])
    mean_output["N_out_clusters"] = (
        mean_output["N_in_cell"] - mean_output["N_in_clusters"]
    )

    # N_per_cluster_mean: Number of Proteins per DBSCAN cluster
    mean_output["N_per_cluster_mean"] = cluster_output["N_per_cluster"].mean()
    mean_output["N_per_cluster_std"] = cluster_output["N_per_cluster"].std()

    # area_mean: Area of DBSCAN cluster
    mean_output["area_mean"] = cluster_output["area (nm^2)"].mean()
    mean_output["area_std"] = cluster_output["area (nm^2)"].std()

    # density_mean: Density of DBSCAN cluster
    mean_output["density_mean"] = cluster_output["density (/nm^2)"].mean()
    mean_output["density_std"] = cluster_output["density (/nm^2)"].std()

    # area_mean: Area of DBSCAN cluster
    mean_output["convex_area_mean"] = cluster_output[
        "convex_area (nm^2)"
    ].mean()
    mean_output["convex_area_std"] = cluster_output["convex_area (nm^2)"].std()

    # circularity_mean: Circularity of DBSCAN cluster
    mean_output["circularity_mean"] = cluster_output["circularity"].mean()
    mean_output["circularity_std"] = cluster_output["circularity"].std()

    # Number of proteins in and outside of clusters for each channel
    N_proteins = input_locs_df.groupby("protein").size()
    # If there is a protein that is never inside the mask, it's ID will be missing in the
    # inedex of the series. This will cause KeyErrors. Instead add the missing index to
    # the Series with a value of 0.
    ID_min = min(list(channel_ID.values()))
    ID_max = max(list(channel_ID.values()))
    N_proteins = N_proteins.reindex(range(ID_min, ID_max + 1), fill_value=0)
    # print(N_proteins)
    """
    protein
    0     1615
    1      414
    2     7951
    3    10609
    4     6689
    5      171
    dtype: int64
    """

    N_proteins_in_clusters = db_input_locs_df.groupby("protein").size()
    # If there is a protein that is never part of a cluster, it's ID will be missing in the
    # inedex of the series. This will cause KeyErrors. Instead add the missing index to
    # the Series with a value of 0.
    ID_min = min(list(channel_ID.values()))
    ID_max = max(list(channel_ID.values()))
    N_proteins_in_clusters = N_proteins_in_clusters.reindex(
        range(ID_min, ID_max + 1), fill_value=0
    )
    # print(N_proteins_in_clusters)
    """
    protein
    0     40
    1     43
    2    471
    3    839
    4    591
    5     12
    dtype: int64
    """
    N_proteins_out_clusters = N_proteins - N_proteins_in_clusters
    # print(N_proteins_out_clusters)
    """
    protein
    0    1575
    1     371
    2    7480
    3    9770
    4    6098
    5     159
    dtype: int64
    """

    for protein in channel_ID:
        protein_ID = channel_ID[protein]

        mean_output["N_" + protein + "_in_cell"] = N_proteins.loc[protein_ID]
        mean_output["N_" + protein + "_in_clusters"] = (
            N_proteins_in_clusters.loc[protein_ID]
        )
        mean_output["N_" + protein + "_out_clusters"] = (
            N_proteins_out_clusters.loc[protein_ID]
        )

    # mean protein composition of clusters: mean amount of each protein in clusters
    for protein in channel_ID:
        mean_output["N_" + protein + "_per_cluster_mean"] = cluster_output[
            "N_" + protein + "_per_cluster"
        ].mean()
        mean_output["N_" + protein + "_per_cluster_std"] = cluster_output[
            "N_" + protein + "_per_cluster"
        ].std()

    # mean protein composition of clusters: mean percentage of each protein in clusters
    for protein in channel_ID:
        mean_output["%_" + protein + "_per_cluster_mean"] = cluster_output[
            "%_" + protein + "_per_cluster"
        ].mean()
        mean_output["%_" + protein + "_per_cluster_std"] = cluster_output[
            "%_" + protein + "_per_cluster"
        ].std()

    return mean_output


def output_cell_mean(
    channel_ID,
    input_locs_df,
    db_input_locs_df,
    cluster_output,
    input_mask_filename,
    px,
    epsilon_nm,
    minpts,
    thresh,
    thresh_type,
    cell,
):
    # output for each cluster = [N_per_cluster, area, density, convex_hull, circularity, N_CD80, ... % CD80, ...]
    # output for complete cell: [N_in_cell, N_in_clusters, N_out_clusters, N_per_cluster_mean, N_per_cluster_CI,
    #                           area_mean, area_CI, density_mean, density_CI,
    #                           convex_area_mean, convex_area_CI, circularity_mean, circularity_CI,
    #                           N_CD80, N_CD80_in_clusters, N_CD86_out_clusters, ... , N_CD80_mean, N_CD80_CI, ...., %_CD80_mean, %_CD80_CI, ....]

    columns = [
        "N_clusters",
        "N_in_cell",
        "N_in_clusters",
        "N_out_clusters",
        "N_per_cluster_mean",
        "N_per_cluster_std",
        "area_mean",
        "area_std",
        "density_mean",
        "density_std",
        "convex_area_mean",
        "convex_area_std",
        "circularity_mean",
        "circularity_std",
    ]

    for protein in channel_ID:
        columns.append("N_" + protein + "_in_cell")
        columns.append("N_" + protein + "_in_clusters")
        columns.append("N_" + protein + "_out_clusters")

    for protein in channel_ID:
        columns.append("N_" + protein + "_per_cluster_mean")
        columns.append("N_" + protein + "_per_cluster_std")

    for protein in channel_ID:
        columns.append("%_" + protein + "_per_cluster_mean")
        columns.append("%_" + protein + "_per_cluster_std")

    mean_output = _output_cell_mean(
        columns,
        channel_ID,
        input_locs_df,
        db_input_locs_df,
        cluster_output,
        input_mask_filename,
        epsilon_nm,
        minpts,
        cell,
    )

    mean_filename = input_mask_filename.replace(
        ".hdf5", "_db-cell-mean_%s_%d.csv" % (str(epsilon_nm), minpts)
    )
    mean_output.to_csv(mean_filename)

    # Save the file containing only clusters above the threshold
    if thresh_type == "area":
        db_input_locs_large_clusters_df = db_input_locs_df[
            db_input_locs_df["area"] * px * px >= thresh
        ]
        cluster_output_large = cluster_output[
            cluster_output["area (nm^2)"] >= thresh
        ]
    if thresh_type == "density":
        db_input_locs_large_clusters_df = db_input_locs_df[
            db_input_locs_df["density"] / px / px >= thresh
        ]
        cluster_output_large = cluster_output[
            cluster_output["density (/nm^2)"] >= thresh
        ]

    mean_output_large = _output_cell_mean(
        columns,
        channel_ID,
        input_locs_df,
        db_input_locs_large_clusters_df,
        cluster_output_large,
        input_mask_filename,
        epsilon_nm,
        minpts,
        cell,
    )
    mean_large_filename = input_mask_filename.replace(
        ".hdf5",
        "_db-cell-mean_%s_%d_above_threshold.csv" % (str(epsilon_nm), minpts),
    )
    mean_output_large.to_csv(mean_large_filename)

    # Save the file containing only clusters below the threshold
    if thresh_type == "area":
        db_input_locs_small_clusters_df = db_input_locs_df[
            db_input_locs_df["area"] * px * px < thresh
        ]
        cluster_output_small = cluster_output[
            cluster_output["area (nm^2)"] < thresh
        ]
    if thresh_type == "density":
        db_input_locs_small_clusters_df = db_input_locs_df[
            db_input_locs_df["density"] / px / px < thresh
        ]
        cluster_output_small = cluster_output[
            cluster_output["density (/nm^2)"] < thresh
        ]

    mean_output_small = _output_cell_mean(
        columns,
        channel_ID,
        input_locs_df,
        db_input_locs_small_clusters_df,
        cluster_output_small,
        input_mask_filename,
        epsilon_nm,
        minpts,
        cell,
    )
    # print(mean_output_small)
    mean_small_filename = input_mask_filename.replace(
        ".hdf5",
        "_db-cell-mean_%s_%d_below_threshold.csv" % (str(epsilon_nm), minpts),
    )
    mean_output_small.to_csv(mean_small_filename)

    return mean_filename, mean_large_filename, mean_small_filename


def output_stimulation_mean(channel_ID, cell_output_merge, stimulation_key):
    # input: dataframe for means of single cells:
    # [N_in_cell, N_in_clusters, N_out_clusters, N_per_cluster_mean, N_per_cluster_CI,
    #  area_mean, area_CI, density_mean, density_CI,
    #  convex_area_mean, convex_area_CI, circularity_mean, circularity_CI,
    #  N_CD80, N_CD80_in_clusters, N_CD86_out_clusters, ... , N_CD80_mean, N_CD80_CI, ...., %_CD80_mean, %_CD80_CI, ....]
    # output: dataframe for mean accross several cells (their means):
    # [N_in_cell_mean, N_in_cell_std, N_in_clusters_mean, N_in_clusters_std, N_out_clusters_mean, N_out_clusters_std,
    #  N_per_cluster_mean, N_per_cluster_std, area_mean, area_CI, density_mean, density_std,
    #  convex_area_mean, convex_area_std, circularity_mean, circularity_CI,
    #  N_CD80, N_CD80_in_clusters, N_CD86_out_clusters, ... , N_CD80_mean, N_CD80_CI, ...., %_CD80_mean, %_CD80_CI, ....]

    columns = [
        "N_clusters_mean",
        "N_clusters_sem",
        "N_in_cell_mean",
        "N_in_cell_sem",
        "N_in_clusters_mean",
        "N_in_clusters_sem",
        "N_out_clusters_mean",
        "N_out_clusters_sem",
        "N_per_cluster_mean",
        "N_per_cluster_sem",
        "area_mean",
        "area_sem",
        "density_mean",
        "density_sem",
        "convex_area_mean",
        "convex_area_sem",
        "circularity_mean",
        "circularity_sem",
    ]

    for protein in channel_ID:
        # protein_ID = channel_ID[protein]
        columns.append("N_" + protein + "_in_cell_mean")
        columns.append("N_" + protein + "_in_cell_sem")

    for protein in channel_ID:
        # protein_ID = channel_ID[protein]
        columns.append("N_" + protein + "_in_clusters_mean")
        columns.append("N_" + protein + "_in_clusters_sem")

    for protein in channel_ID:
        # protein_ID = channel_ID[protein]
        columns.append("N_" + protein + "_out_clusters_mean")
        columns.append("N_" + protein + "_out_clusters_sem")

    for protein in channel_ID:
        # protein_ID = channel_ID[protein]
        columns.append("N_" + protein + "_per_cluster_mean")
        columns.append("N_" + protein + "_per_cluster_sem")

    for protein in channel_ID:
        # protein_ID = channel_ID[protein]
        columns.append("%_" + protein + "_per_cluster_mean")
        columns.append("%_" + protein + "_per_cluster_sem")

    # Calculate standard error of the mean via std/sqrt(N_cells):
    N_cells = len(cell_output_merge)
    N_sqrt = np.sqrt(N_cells)

    mean_output = pd.DataFrame(columns=columns, index=[stimulation_key])

    mean_output["N_clusters_mean"] = cell_output_merge["N_clusters"].mean()
    mean_output["N_clusters_sem"] = (
        cell_output_merge["N_clusters"].std() / N_sqrt
    )

    # N_in_cell_mean: mean number of proteins per cell
    mean_output["N_in_cell_mean"] = cell_output_merge["N_in_cell"].mean()
    mean_output["N_in_cell_sem"] = (
        cell_output_merge["N_in_cell"].std() / N_sqrt
    )

    # N_in_clusters_mean: mean number of proteins per cell that are in clusters
    mean_output["N_in_clusters_mean"] = cell_output_merge[
        "N_in_clusters"
    ].mean()
    mean_output["N_in_clusters_sem"] = (
        cell_output_merge["N_in_clusters"].std() / N_sqrt
    )

    # N_out_clusters_mean: mean number of proteins per cell that are outside of clusters
    mean_output["N_out_clusters_mean"] = cell_output_merge[
        "N_out_clusters"
    ].mean()
    mean_output["N_out_clusters_sem"] = (
        cell_output_merge["N_out_clusters"].std() / N_sqrt
    )

    # N_per_cluster_mean: Number of Proteins per DBSCAN cluster
    mean_output["N_per_cluster_mean"] = cell_output_merge[
        "N_per_cluster_mean"
    ].mean()
    mean_output["N_per_cluster_sem"] = (
        cell_output_merge["N_per_cluster_mean"].std() / N_sqrt
    )

    # area_mean: Area of DBSCAN cluster
    mean_output["area_mean"] = cell_output_merge["area_mean"].mean()
    mean_output["area_sem"] = cell_output_merge["area_mean"].std() / N_sqrt

    # density_mean: Density of DBSCAN cluster
    mean_output["density_mean"] = cell_output_merge["density_mean"].mean()
    mean_output["density_sem"] = (
        cell_output_merge["density_mean"].std() / N_sqrt
    )

    # density_mean: Density of DBSCAN cluster
    mean_output["convex_area_mean"] = cell_output_merge[
        "convex_area_mean"
    ].mean()
    mean_output["convex_area_sem"] = (
        cell_output_merge["convex_area_mean"].std() / N_sqrt
    )

    # circularity_mean: Circularity of DBSCAN cluster
    mean_output["circularity_mean"] = cell_output_merge[
        "circularity_mean"
    ].mean()
    mean_output["circularity_sem"] = (
        cell_output_merge["circularity_mean"].std() / N_sqrt
    )

    # For each protein channel individually: f.ex. for CD80
    # N_CD80_mean: mean number of CD80 proteins per cell
    for protein in channel_ID:
        mean_output["N_" + protein + "_in_cell_mean"] = cell_output_merge[
            "N_" + protein + "_in_cell"
        ].mean()
        mean_output["N_" + protein + "_in_cell_sem"] = (
            cell_output_merge["N_" + protein + "_in_cell"].std() / N_sqrt
        )

    # For each protein channel individually: f.ex. for CD80
    # N_CD80_in_clusters_mean: mean number of CD80 proteins per cell that are in clusters
    for protein in channel_ID:
        mean_output["N_" + protein + "_in_clusters_mean"] = cell_output_merge[
            "N_" + protein + "_in_clusters"
        ].mean()
        mean_output["N_" + protein + "_in_clusters_sem"] = (
            cell_output_merge["N_" + protein + "_in_clusters"].std() / N_sqrt
        )

    # For each protein channel individually: f.ex. for CD80
    # N_CD80_out_clusters_mean: mean number of CD80 proteins per cell that are outside of clusters
    for protein in channel_ID:
        mean_output["N_" + protein + "_out_clusters_mean"] = cell_output_merge[
            "N_" + protein + "_out_clusters"
        ].mean()
        mean_output["N_" + protein + "_out_clusters_sem"] = (
            cell_output_merge["N_" + protein + "_out_clusters"].std() / N_sqrt
        )

    # mean protein composition of clusters: mean amount of each protein in clusters
    for protein in channel_ID:
        # protein_ID = channel_ID[protein]
        mean_output["N_" + protein + "_per_cluster_mean"] = cell_output_merge[
            "N_" + protein + "_per_cluster_mean"
        ].mean()
        mean_output["N_" + protein + "_per_cluster_sem"] = (
            cell_output_merge["N_" + protein + "_per_cluster_mean"].std()
            / N_sqrt
        )

    # mean protein composition of clusters: mean percentage of each protein in clusters
    for protein in channel_ID:
        # protein_ID = channel_ID[protein]
        mean_output["%_" + protein + "_per_cluster_mean"] = cell_output_merge[
            "%_" + protein + "_per_cluster_mean"
        ].mean()
        mean_output["%_" + protein + "_per_cluster_sem"] = (
            cell_output_merge["%_" + protein + "_per_cluster_mean"].std()
            / N_sqrt
        )

    return mean_output
