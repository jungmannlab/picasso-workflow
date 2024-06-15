import numpy as np
import os
import os.path
import h5py
import pandas as pd
import sys
import math


from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

# from hnnd import hNND


def convex_hull_f(group, x, y):
    X_group = np.vstack((group[x], group[y])).T
    try:
        hull = ConvexHull(X_group)
        convex_hull = hull.volume
        # hull.volume is area for 2D

    except Exception as e:
        print(e)
        convex_hull = 0

    # print("convex_hull_ready")

    return convex_hull


def convex_perimeter_f(group, x, y):
    X_group = np.vstack((group[x], group[y])).T
    try:
        hull = ConvexHull(X_group)
        convex_perimeter = hull.area
        # hull.area is perimeter for 2D

    except Exception as e:
        print(e)
        convex_perimeter = 0

    # print("convex_perimeter_ready")

    return convex_perimeter


def convex_circularity_f(group, x, y):
    X_group = np.vstack((group[x], group[y])).T
    try:
        hull = ConvexHull(X_group)
        convex_circularity = (4 * np.pi * hull.volume) / (hull.area) ** 2
        # definition of circularity according to wikipedia:
        # Circularity = 4π × Area/Perimeter^2, which is 1 for a perfect circle and
        # goes down as far as 0 for highly non-circular shapes.

    except Exception as e:
        print(e)
        convex_circularity = 0

    # print("convex_circularity_ready")

    return convex_circularity


def dbscan_f(df, epsilon, minpts, sigma_linker):

    X = np.vstack((df["x"], df["y"])).T

    db = DBSCAN(eps=epsilon, min_samples=minpts).fit(X)
    group = np.int32(db.labels_)

    df["group"] = group
    # print(df.columns)
    # print(df['group'])

    df_cluster = df.loc[df["group"] != -1]

    grouped = df_cluster.groupby("group")
    group_means = grouped.mean()
    group_std = grouped.std(ddof=0)
    # Durch ddof = 0 wird der Nenner zu n-0 statt n-1 (ddof=1 ist standard).
    # Damit stimmen die Resultate fuer die ersten Nachkommastellen
    # mit picasso dbscan ueberein.

    print("n clusters", len(group_means))

    group_size = grouped.size()
    group_size.name = "group_size"

    # Circularity of clusters
    convex_circularity = grouped.apply(convex_circularity_f, "x", "y")

    # convex_hull area and perimeter
    convex_hull = grouped.apply(convex_hull_f, "x", "y")
    convex_perimeter = grouped.apply(convex_perimeter_f, "x", "y")

    # estimation of an area based on the convex hull taking the linker
    # uncertainty into account:
    # A = area in convex_hull + perimeter * sigma_linker
    area = convex_hull + convex_perimeter * sigma_linker

    # area approximation done in Picasso's DBSCAN implementation (not used here)
    # area_sx_sy = np.power((group_std['x'] + group_std['y']), 2) * np.pi

    # Add the convex hull (one way to calculate area) value to the df_cluster dataframe
    # This will allow for filtering locs by the size of the cluster to which they belong.
    convex_hull_add = convex_hull.copy()
    if isinstance(convex_hull_add, pd.Series):
        # print("in if")
        convex_hull_add = convex_hull_add.to_frame()
    convex_hull_add = convex_hull_add.rename(columns={0: "convex_area"})
    convex_hull_add["group"] = convex_hull_add.index
    convex_hull_add = convex_hull_add.rename_axis(None)

    df_cluster = df_cluster.merge(convex_hull_add, on="group", how="left")

    # Add the area to the df_cluster based on convex_hull and convex_perimeter
    area_add = area.copy()
    if isinstance(area_add, pd.Series):
        area_add = area_add.to_frame()
    area_add = area_add.rename(columns={0: "area"})
    area_add["group"] = area_add.index
    area_add = area_add.rename_axis(None)
    df_cluster = df_cluster.merge(area_add, on="group", how="left")

    # Calculate the density based on the area
    density = group_size / area  # molecules / pixel^2
    density[area == 0] = float("nan")

    # Add the density to the df_cluster
    density_add = density.copy()
    if isinstance(density_add, pd.Series):
        density_add = density_add.to_frame()
    density_add = density_add.rename(columns={0: "density"})
    density_add["group"] = density_add.index
    density_add = density_add.rename_axis(None)
    df_cluster = df_cluster.merge(density_add, on="group", how="left")

    # print(df_cluster.keys())

    """
    Generating hdf5 file for picasso render with all localizations assigned to a cluster
    Colorcoding = Cluster_ID
    """

    import h5py as _h5py
    import numpy as _np

    try:
        LOCS_DTYPE = [
            ("frame", "u4"),
            ("x", "f4"),
            ("y", "f4"),
            ("photons", "f4"),
            ("sx", "f4"),
            ("sy", "f4"),
            ("bg", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
            ("protein", "u4"),
            ("group", "u4"),
            ("convex_area", "f4"),
            ("area", "f4"),
            ("density", "f4"),
        ]
        db_locs = _np.rec.array(
            (
                df_cluster.frame,
                df_cluster.x,
                df_cluster.y,
                df_cluster.photons,
                df_cluster.sx,
                df_cluster.sy,
                df_cluster.bg,
                df_cluster.lpx,
                df_cluster.lpy,
                df_cluster.protein,
                df_cluster.group,
                df_cluster.convex_area,
                df_cluster.area,
                df_cluster.density,
            ),
            dtype=LOCS_DTYPE,
        )
    except AttributeError:
        # error raised when CSR data is used
        LOCS_DTYPE = [
            ("frame", "u4"),
            ("x", "f4"),
            ("y", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
            ("protein", "u4"),
            ("group", "u4"),
            ("convex_area", "f4"),
            ("area", "f4"),
            ("density", "f4"),
        ]
        db_locs = _np.rec.array(
            (
                df_cluster.frame,
                df_cluster.x,
                df_cluster.y,
                df_cluster.lpx,
                df_cluster.lpy,
                df_cluster.protein,
                df_cluster.group,
                df_cluster.convex_area,
                df_cluster.area,
                df_cluster.density,
            ),
            dtype=LOCS_DTYPE,
        )

    """
    Generating hdf5 file for picasso render with all localizations assigned to a cluster
    Colorcoding = protein_ID

    """

    try:
        LOCS_DTYPE = [
            ("frame", "u4"),
            ("x", "f4"),
            ("y", "f4"),
            ("photons", "f4"),
            ("sx", "f4"),
            ("sy", "f4"),
            ("bg", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
            ("group", "u4"),
            ("Cluster_ID", "u4"),
            ("convex_area", "f4"),
            ("area", "f4"),
            ("density", "f4"),
        ]
        db_locs_protein_ID = _np.rec.array(
            (
                df_cluster.frame,
                df_cluster.x,
                df_cluster.y,
                df_cluster.photons,
                df_cluster.sx,
                df_cluster.sy,
                df_cluster.bg,
                df_cluster.lpx,
                df_cluster.lpy,
                df_cluster.protein,
                df_cluster.group,
                df_cluster.convex_area,
                df_cluster.area,
                df_cluster.density,
            ),
            dtype=LOCS_DTYPE,
        )
    except AttributeError:
        # error raised when CSR data is used
        LOCS_DTYPE = [
            ("frame", "u4"),
            ("x", "f4"),
            ("y", "f4"),
            ("lpx", "f4"),
            ("lpy", "f4"),
            ("group", "u4"),
            ("Cluster_ID", "u4"),
            ("convex_area", "f4"),
            ("area", "f4"),
            ("density", "f4"),
        ]
        db_locs_protein_ID = _np.rec.array(
            (
                df_cluster.frame,
                df_cluster.x,
                df_cluster.y,
                df_cluster.lpx,
                df_cluster.lpy,
                df_cluster.protein,
                df_cluster.group,
                df_cluster.convex_area,
                df_cluster.area,
                df_cluster.density,
            ),
            dtype=LOCS_DTYPE,
        )

    """
    Generating hdf5 file containg cluster properties analogously to the DBSCAN output Picasso saves
    """

    data3_group = group_means.index.values.tolist()
    data3_convex_perimeter = convex_perimeter.values.tolist()
    data3_convex_hull = convex_hull.values.tolist()

    data3_convex_circularity = convex_circularity.values.tolist()
    data3_area = list(area)
    data3_frames = group_means["frame"].values.tolist()
    data3_x = group_means["x"].values.tolist()
    data3_y = group_means["y"].values.tolist()
    data3_std_frame = group_std["frame"].values.tolist()
    data3_std_x = group_std["x"].values.tolist()
    data3_std_y = group_std["y"].values.tolist()
    data3_n = group_size.values.tolist()
    data3_density = density.values.tolist()

    data = {
        "groups": data3_group,
        "convex_area": data3_convex_hull,
        "convex_perimeter": data3_convex_perimeter,
        "convex_circularity": data3_convex_circularity,
        "area": data3_area,
        "mean_frame": data3_frames,
        "com_x": data3_x,
        "com_y": data3_y,
        "std_frame": data3_std_frame,
        "std_x": data3_std_x,
        "std_y": data3_std_y,
        "n": data3_n,
        "density": data3_density,
    }

    df = pd.DataFrame(data, index=range(len(data3_x)))

    df3 = df.reindex(
        columns=[
            "groups",
            "convex_area",
            "convex_perimeter",
            "convex_circularity",
            "area",
            "mean_frame",
            "com_x",
            "com_y",
            "std_frame",
            "std_x",
            "std_y",
            "n",
            "density",
        ],
        fill_value=1,
    )

    LOCS_DTYPE = [
        ("groups", "u4"),
        ("convex_area", "f4"),
        ("convex_perimeter", "f4"),
        ("convex_circularity", "f4"),
        ("area", "f4"),
        ("mean_frame", "f4"),
        ("com_x", "f4"),
        ("com_y", "f4"),
        ("std_frame", "f4"),
        ("std_x", "f4"),
        ("std_y", "f4"),
        ("n", "u4"),
        ("density", "f4"),
    ]
    db_cluster_props = _np.rec.array(
        (
            df3.groups,
            df3.convex_area,
            df3.convex_perimeter,
            df3.convex_circularity,
            df3.area,
            df3.mean_frame,
            df3.com_x,
            df3.com_y,
            df3.std_frame,
            df3.std_x,
            df3.std_y,
            df3.n,
            df3.density,
        ),
        dtype=LOCS_DTYPE,
    )

    # df_cluster = db_locs as Dataframe, df3 = db_cluster_props as dataframe
    return db_locs, db_locs_protein_ID, db_cluster_props, df_cluster, df3
