"""Report histogram helpers (extracted from picasso_outpost).

`keep` -- workflow-side presentation glue that draws 1D / 2D histograms of
localization fields for the HTML / Confluence report modules. Both bin via
``lib.calculate_optimal_bins`` (picasso >= 0.10).

Author: Heinrich Grabmayr
Initial date: March 8, 2024
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LogNorm

from picasso_workflow._picasso_api import lib


def plot_1dhist(locs, field, fig, ax):
    # np.asarray: lib.calculate_optimal_bins (picasso >= 0.10) samples via
    # positional integer indexing (data[rng.choice(...)]), which fails on a
    # pandas Series with a non-RangeIndex. Force a numpy array.
    data = np.asarray(locs[field])
    data = data[np.isfinite(data)]
    bins = lib.calculate_optimal_bins(data, 1000)
    # Prepare the figure
    fig.suptitle(field)
    ax.hist(data, bins, rwidth=1, linewidth=0)
    data_range = np.ptp(data)
    ax.set_xlim([bins[0] - 0.05 * data_range, data.max() + 0.05 * data_range])


def plot_2dhist(locs, field_x, field_y, fig, ax):
    # np.asarray: see plot_1dhist - lib.calculate_optimal_bins samples by
    # positional index, which breaks on a pandas Series with a non-default
    # index. Force numpy arrays.
    x = np.asarray(locs[field_x])
    y = np.asarray(locs[field_y])
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    # Start hist2 version
    bins_x = lib.calculate_optimal_bins(x, 1000)
    bins_y = lib.calculate_optimal_bins(y, 1000)
    counts, x_edges, y_edges, image = ax.hist2d(
        x, y, bins=[bins_x, bins_y], norm=LogNorm()
    )
    x_range = np.ptp(x)
    ax.set_xlim([bins_x[0] - 0.05 * x_range, x.max() + 0.05 * x_range])
    y_range = np.ptp(y)
    ax.set_ylim([bins_y[0] - 0.05 * y_range, y.max() + 0.05 * y_range])
    fig.colorbar(image, ax=ax)
    ax.grid(False)
    ax.get_xaxis().set_label_text(field_x)
    ax.get_yaxis().set_label_text(field_y)
