"""
########################################################
#########           Philipp R. Steen           #########
#########             Jungmann Lab             #########
######### Max Planck Institute of Biochemistry #########
#########     Ludwig Maximilian University     #########
#########                 2024                 #########
########################################################
"""

import numpy as np
import pandas as pd
from lmfit import Model
from itertools import groupby
from operator import itemgetter
import os
import multiprocessing
from pandarallel import pandarallel

from picasso_workflow._picasso_api import postprocess


class Measurement:
    def __init__(
        self,
        in_path="",
        save_path="",
        saving_name="",
        total_n_frames=20000,
        pixelsize=130.0,
    ):
        self.in_path = in_path
        self.save_path = save_path
        self.saving_name = saving_name
        self.total_n_frames = total_n_frames
        # Camera pixel size in nm. picasso 1.0 stores sx/sy in nm, so the
        # 1x1-camera-pixel integration bound must be scaled by it.
        self._pixelsize = pixelsize
        # Metadata list expected by picasso.postprocess (SBR needs
        # ``Pixelsize``; the rate/damage metrics need ``Frames``).
        self.info = [
            {"Frames": int(total_n_frames), "Pixelsize": float(pixelsize)}
        ]

        self.table = []
        self.table_g = []
        self.table_k = []
        # Aggregate picasso-1.0 metrics, populated in Begin(); NaN until then
        # and on sparse/degenerate data (fit failure).
        self.loc_rate_drop = float("nan")
        self.destroyed_fraction = float("nan")

    def Begin(self):
        self.Import()
        # cleaning_bg returns a filtered copy; assign it back (it does not
        # mutate self.table in place) so negative-bg rows don't reach the SBR.
        self.table = self.cleaning_bg()
        pandarallel.initialize(nb_workers=min(30, multiprocessing.cpu_count()))
        self.table_g = self.table.groupby("group").parallel_apply(
            self.GroupCalcs
        )
        # SBR via picasso.postprocess (computed here in the main process, not
        # inside the pandarallel workers). The old hand-rolled integral used
        # center_photons (0 on edge frames), so edge rows had sbr=0; on center
        # frames center_photons == photons, so signal_to_background matches the
        # old integral exactly. Masking to center_frame reproduces that
        # "0 on edge frames" semantics identically.
        sbr_all = postprocess.signal_to_background(self.table_g, self.info)
        self.table_g["sbr"] = np.where(
            self.table_g["center_frame"], sbr_all, 0.0
        )
        # Table g contains photon and SBR information
        self.table_k = self.table.groupby("group").parallel_apply(
            self.KineticsCalcs
        )
        self.table_k["mean_bright"] = self.table_k[
            "bright_times"
        ].parallel_apply(self.CDF_kinetics)
        self.table_k["mean_dark"] = self.table_k["dark_times"].parallel_apply(
            self.CDF_kinetics
        )
        self.table_k["r"] = (
            self.table_k["end_times"] / self.table_k["mean_dark"]
        )
        # Table k contains kinetics information

        # Aggregate picasso-1.0 metrics. Real acquisitions can be sparse and
        # the underlying curve fits can fail or degenerate; docking_site_damage
        # also needs localized columns (x/y/lpx/lpy) on each pick to link
        # binding events. Each call is guarded and falls back to NaN (the fork
        # already tolerates fit failure) -- so a table missing a required
        # column yields NaN rather than raising. picasso raises a bare KeyError
        # for an absent column/metadata key, so both guards include KeyError.
        try:
            self.loc_rate_drop = float(
                postprocess.localization_rate_drop(self.table, self.info)
            )
        except (ValueError, RuntimeError, KeyError):
            self.loc_rate_drop = float("nan")
        try:
            picked_locs = [grp for _, grp in self.table.groupby("group")]
            self.destroyed_fraction = float(
                postprocess.docking_site_damage(picked_locs, self.info)[
                    "fraction_destroyed"
                ]
            )
        except (ValueError, RuntimeError, KeyError):
            self.destroyed_fraction = float("nan")

    def Import(self):
        """Imports clustered .hdf5 from Picasso. Requires group info."""
        self.table = pd.read_hdf(self.in_path, key="locs")
        self.table = self.table.sort_values(by=["group", "frame"])

    def BindingEvents(self, frames):
        """Links localizations and returns all bright and dark times
        from a trace.
        """
        ranges = []
        for k, g in groupby(enumerate(frames), lambda x: x[0] - x[1]):
            group = map(itemgetter(1), g)
            group = list(map(int, group))
            ranges.append((group[0], group[-1]))
        events = np.asarray(ranges)
        beginnings = events[:, 0]
        dark_times = beginnings[1:] - beginnings[:-1]
        bright_times = events[:, 1] - events[:, 0] + 1
        end_time = self.total_n_frames - events[-1, 1]
        return (events, bright_times, dark_times, end_time)

    def expfunc(self, t, tau):
        """Used for fitting CDF_kinetics."""
        return 1 - np.exp(-(t / tau))

    def CDF_kinetics(self, values):
        """Calculates mean bright or dark times based on an exponential CDF
        fit.
        """
        if len(values) >= 2:
            values_sorted = np.sort(values)
            p = 1.0 * np.arange(len(values_sorted)) / (len(values_sorted) - 1)
            k_model = Model(self.expfunc)
            try:
                k_result = k_model.fit(p, t=values_sorted, tau=200)
                tau = k_result.params["tau"].value
            except Exception:
                tau = -1
        else:
            tau = -1
        return tau

    def GroupCalcs(self, group_df):
        """Finds consecutive binding events, center frames and
        photon counts. SBR is computed vectorized in ``Begin``.
        """

        # Create a new column that numbers binding events consecutively
        group_df["binding_event"] = (
            (group_df.frame.diff(1) != 1).astype("int").cumsum()
        )
        # Create a new column that labels center frames True
        # and other frames False
        group_df["center_frame"] = group_df.binding_event.eq(
            group_df.binding_event.shift()
        ) & group_df.binding_event.eq(group_df.binding_event.shift(periods=-1))
        # Create a new column containing photon counts from edge frames
        # (zero otherwise)
        group_df["edge_photons"] = np.where(
            ~group_df["center_frame"], group_df["photons"], 0
        )
        # Create a new column containing photon counts from center frames
        # (zero otherwise)
        group_df["center_photons"] = np.where(
            group_df["center_frame"], group_df["photons"], 0
        )
        return group_df

    def KineticsCalcs(self, group_df):
        """Finds consecutive binding events."""
        events, bright_times, dark_times, end_times = self.BindingEvents(
            np.asarray(group_df["frame"])
        )
        return pd.Series(
            data=(bright_times, dark_times, end_times),
            index=["bright_times", "dark_times", "end_times"],
        )

    def cleaning_bg(self):
        """If Picasso Localize has a fitting error, the background value
        may be negative. This should be removed.
        """
        return self.table.drop(self.table[self.table["bg"] <= 0.0].index)

    def FileSaver(self):
        """Saves the dataframes for later use"""
        self.table_g.to_csv(
            os.path.join(
                self.save_path, "table_g_" + self.saving_name + ".csv"
            ),
            index=True,
        )
        self.table_g.to_pickle(
            os.path.join(
                self.save_path, "table_g_" + self.saving_name + ".pkl"
            )
        )
        self.table_k.to_csv(
            os.path.join(
                self.save_path, "table_k_" + self.saving_name + ".csv"
            ),
            index=True,
        )
        self.table_k.to_pickle(
            os.path.join(
                self.save_path, "table_k_" + self.saving_name + ".pkl"
            )
        )
        return
