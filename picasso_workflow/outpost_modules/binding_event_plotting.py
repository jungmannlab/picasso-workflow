"""
########################################################
#########           Philipp R. Steen           #########
#########             Jungmann Lab             #########
######### Max Planck Institute of Biochemistry #########
#########     Ludwig Maximilian University     #########
#########                 2024                 #########
########################################################
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import configparser
from datetime import datetime
import os


class Plotting:
    def __init__(
        self,
        table_g="",
        table_k="",
        show=False,
        save=False,
        save_path="",
        saving_name="",
        total_n_frames=20000,
    ):
        self.table_g = table_g
        self.table_k = table_k
        self.show = show
        self.save = save
        self.save_path = save_path
        self.saving_name = saving_name
        self.total_n_frames = total_n_frames
        self.mean_photons = 0
        self.n_mean_photons = 0
        self.mean_bg_photons = 0
        self.n_mean_bg_photons = 0
        self.mean_sbr = 0
        self.n_mean_sbr = 0
        self.mean_tb = 0
        self.n_mean_tb = 0
        self.mean_td = 0
        self.n_mean_td = 0
        self.mu_fit = 0
        self.cut_off = 0
        self.n_r = 0
        self.actual_percentage_dest = 0
        self.avg_start = 0
        self.avg_end = 0
        self.drop = 0
        self.n_locs = 0

    def Plot_photons(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Photons")
        mean_photons = np.mean(
            self.table_g[self.table_g.center_photons > 0]["center_photons"]
        )
        n_mean_photons = len(
            self.table_g[self.table_g.center_photons > 0]["center_photons"]
        )
        binwidth = mean_photons / 20
        ax.hist(
            self.table_g[self.table_g.edge_photons > 0]["edge_photons"],
            alpha=0.5,
            bins=np.arange(
                min(
                    self.table_g[self.table_g.edge_photons > 0]["edge_photons"]
                ),
                max(
                    self.table_g[self.table_g.edge_photons > 0]["edge_photons"]
                )
                + binwidth,
                binwidth,
            ),
            label="Edge photons",
        )
        ax.hist(
            self.table_g[self.table_g.center_photons > 0]["center_photons"],
            alpha=0.5,
            bins=np.arange(
                min(
                    self.table_g[self.table_g.center_photons > 0][
                        "center_photons"
                    ]
                ),
                max(
                    self.table_g[self.table_g.center_photons > 0][
                        "center_photons"
                    ]
                )
                + binwidth,
                binwidth,
            ),
            label="Center photons",
        )
        ax.axvline(
            x=mean_photons,
            label=(
                "Mean photon count = "
                + str(mean_photons)
                + "\nN_locs = "
                + str(n_mean_photons)
            ),
        )
        ax.legend()
        ax.set_xlim(0, 2 * mean_photons)
        if self.save:
            fp = os.path.join(
                self.save_path, self.saving_name + "_photons.pdf"
            )
            plt.savefig(fp)
        if self.show:
            fp = ""
            plt.show()
        self.mean_photons = mean_photons
        self.n_mean_photons = n_mean_photons
        return (mean_photons, n_mean_photons), fp

    def Plot_bg(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Background photons")
        mean_bg_photons = np.mean(self.table_g[self.table_g.bg > 0]["bg"])
        n_mean_bg_photons = len(self.table_g[self.table_g.bg > 0]["bg"])
        binwidth = mean_bg_photons / 20
        ax.hist(
            self.table_g[self.table_g.bg > 0]["bg"],
            bins=np.arange(
                min(self.table_g[self.table_g.bg > 0]["bg"]),
                max(self.table_g[self.table_g.bg > 0]["bg"]) + binwidth,
                binwidth,
            ),
            label="Bg photons",
        )
        ax.axvline(
            x=mean_bg_photons,
            label=(
                "Mean bg photon count = "
                + str(mean_bg_photons)
                + "\nN_locs = "
                + str(n_mean_bg_photons)
            ),
        )
        ax.legend()
        ax.set_xlim(0, 2 * mean_bg_photons)
        if self.save:
            fp = os.path.join(
                self.save_path, self.saving_name + "_background_photons.pdf"
            )
            plt.savefig(fp)
        if self.show:
            fp = ""
            plt.show()
        self.mean_bg_photons = mean_bg_photons
        self.n_mean_bg_photons = n_mean_bg_photons
        return (mean_bg_photons, n_mean_bg_photons), fp

    def Plot_sbr(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("SBR")
        mean_sbr = np.mean(self.table_g[self.table_g.sbr > 0]["sbr"])
        n_mean_sbr = len(self.table_g[self.table_g.sbr > 0]["sbr"])
        binwidth = mean_sbr / 20
        ax.hist(
            self.table_g[self.table_g.sbr > 0]["sbr"],
            bins=np.arange(
                min(self.table_g[self.table_g.sbr > 0]["sbr"]),
                max(self.table_g[self.table_g.sbr > 0]["sbr"]) + binwidth,
                binwidth,
            ),
            label="SBR",
        )
        ax.axvline(
            x=mean_sbr,
            label=(
                "Mean SBR = " + str(mean_sbr) + "\nN_locs = " + str(n_mean_sbr)
            ),
        )
        ax.legend()
        ax.set_xlim(0, 2 * mean_sbr)
        if self.save:
            fp = os.path.join(self.save_path, self.saving_name + "_sbr.pdf")
            plt.savefig(fp)
        if self.show:
            fp = ""
            plt.show()
        self.mean_sbr = mean_sbr
        self.n_mean_sbr = n_mean_sbr
        return (mean_sbr, n_mean_sbr), fp

    def Plot_tb(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Bright times")
        mean_tb = np.mean(
            self.table_k[self.table_k.mean_bright > 0]["mean_bright"]
        )
        n_mean_tb = len(
            self.table_k[self.table_k.mean_bright > 0]["mean_bright"]
        )
        binwidth = mean_tb / 20
        ax.hist(
            self.table_k[self.table_k.mean_bright > 0]["mean_bright"],
            bins=np.arange(
                min(self.table_k[self.table_k.mean_bright > 0]["mean_bright"]),
                max(self.table_k[self.table_k.mean_bright > 0]["mean_bright"])
                + binwidth,
                binwidth,
            ),
            label="Mean bright times",
        )
        ax.axvline(
            x=mean_tb,
            label=(
                "Mean bright time = "
                + str(mean_tb)
                + "\nN binding events = "
                + str(n_mean_tb)
            ),
        )
        ax.legend()
        ax.set_xlim(0, 2 * mean_tb)
        if self.save:
            fp = os.path.join(
                self.save_path, self.saving_name + "_t_bright.pdf"
            )
            plt.savefig(fp)
        if self.show:
            fp = ""
            plt.show()
        self.mean_tb = mean_tb
        self.n_mean_tb = n_mean_tb
        return (mean_tb, n_mean_tb), fp

    def Plot_td(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Dark times")
        mean_td = np.mean(
            self.table_k[self.table_k.mean_dark > 0]["mean_dark"]
        )
        n_mean_td = len(self.table_k[self.table_k.mean_dark > 0]["mean_dark"])
        binwidth = mean_td / 20
        ax.hist(
            self.table_k[self.table_k.mean_dark > 0]["mean_dark"],
            bins=np.arange(
                min(self.table_k[self.table_k.mean_dark > 0]["mean_dark"]),
                max(self.table_k[self.table_k.mean_dark > 0]["mean_dark"])
                + binwidth,
                binwidth,
            ),
            label="Mean dark times",
        )
        ax.axvline(
            x=mean_td,
            label=(
                "Mean dark time = "
                + str(mean_td)
                + "\nN binding events = "
                + str(n_mean_td)
            ),
        )
        ax.legend()
        ax.set_xlim(0, 2 * mean_td)
        if self.save:
            fp = os.path.join(self.save_path, self.saving_name + "_t_dark.pdf")
            plt.savefig(fp)
        if self.show:
            fp = ""
            plt.show()
        self.mean_td = mean_td
        self.n_mean_td = n_mean_td
        return (mean_td, n_mean_td), fp

    def Plot_r(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Ratio r")
        n_r = len(self.table_k[self.table_k.r > 0]["r"])
        binwidth = 0.5
        n, bins, patches = ax.hist(
            self.table_k[self.table_k.r > 0]["r"],
            bins=np.arange(
                min(self.table_k[self.table_k.r > 0]["r"]),
                max(self.table_k[self.table_k.r > 0]["r"]) + binwidth,
                binwidth,
            ),
            label="r values\nN binding events = " + str(n_r),
        )
        bin_middles = bins + (0.5 * binwidth)
        xvals, yvals = bin_middles[:-1], n
        popt_exp, pcov_exp = curve_fit(self.exp_r, xvals, yvals, p0=[500, 1])
        mu_fit = 1 / popt_exp[1]
        cut_off = max(4, 4 * mu_fit)
        apparently_destroyed_sites = self.table_k[self.table_k["r"] >= cut_off]
        apparently_good_sites = self.table_k[self.table_k["r"] < cut_off]
        num_dest = len(apparently_destroyed_sites)
        num_good = len(apparently_good_sites)
        actual_portion_dest = (num_dest / (num_dest + num_good)) - 1 / np.exp(
            4
        )
        actual_percentage_dest = 100 * actual_portion_dest
        x_plot_axis = np.linspace(0, 10, 1000)
        ax.plot(
            x_plot_axis, self.exp_r(x_plot_axis, *popt_exp), label="Exp fit"
        )
        ax.axvline(x=mu_fit, label=("μ = " + str(mu_fit)))
        ax.axvline(
            x=cut_off,
            label=(
                "Cut-off = "
                + str(cut_off)
                + "\nPercentage destroyed = "
                + str(np.round(actual_percentage_dest, 3))
                + "%"
            ),
        )
        ax.legend()
        ax.set_xlim(0, 10)
        if self.save:
            fp = os.path.join(
                self.save_path, self.saving_name + "_ratio_r_destr.pdf"
            )
            plt.savefig(fp)
        if self.show:
            fp = ""
            plt.show()
        self.mu_fit = mu_fit
        self.cut_off = cut_off
        self.n_r = n_r
        self.actual_percentage_dest = actual_percentage_dest
        return (mu_fit, cut_off, n_r, actual_percentage_dest), fp

    def exp_r(self, x, a, t):
        return a * np.exp(-t * x)

    def Plot_locs(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Localizations")
        n_locs = len(self.table_g[self.table_g.frame > 0]["frame"])
        binwidth = 500
        n, bins, patches = ax.hist(
            self.table_g["frame"],
            bins=np.arange(0, self.total_n_frames + binwidth, binwidth),
            label="N_locs = " + str(n_locs),
        )
        upto_20 = self.table_g[
            self.table_g["frame"] <= 0.2 * self.total_n_frames
        ]
        over_80 = self.table_g[
            self.table_g["frame"] >= 0.8 * self.total_n_frames
        ]
        upto_20_len = upto_20["frame"].size
        over_80_len = over_80["frame"].size
        avg_start = upto_20_len / (0.2 * self.total_n_frames / binwidth)
        avg_end = over_80_len / (0.2 * self.total_n_frames / binwidth)
        drop = (1 - avg_end / avg_start) * 100
        ax.axhline(y=avg_start, color="black", linewidth=0.8)
        ax.axhline(
            y=avg_end,
            color="black",
            linewidth=0.8,
            label="Change: " + str(np.round(-drop, 1)) + "%",
        )
        ax.legend()
        if self.save:
            fp = os.path.join(
                self.save_path, self.saving_name + "_locs_over_time.pdf"
            )
            plt.savefig(fp)
        if self.show:
            fp = ""
            plt.show()
        self.avg_start = avg_start
        self.avg_end = avg_end
        self.drop = drop
        self.n_locs = n_locs
        return (avg_start, avg_end, drop, n_locs), fp

    def SaveAllResults(self):
        config = configparser.ConfigParser()
        config["params"] = {
            "Date and time": str(datetime.now()),
            "saving_name": str(self.saving_name),
            "total_n_frames": str(self.total_n_frames),
            "mean_photons": str(self.mean_photons),
            "n_mean_photons": str(self.n_mean_photons),
            "mean_bg_photons": str(self.mean_bg_photons),
            "n_mean_bg_photons": str(self.n_mean_bg_photons),
            "mean_sbr": str(self.mean_sbr),
            "n_mean_sbr": str(self.n_mean_sbr),
            "mean_tb": str(self.mean_tb),
            "n_mean_tb": str(self.n_mean_tb),
            "mean_td": str(self.mean_td),
            "n_mean_td": str(self.n_mean_td),
            "mu_fit": str(self.mu_fit),
            "cut_off": str(self.cut_off),
            "n_r": str(self.n_r),
            "actual_percentage_dest": str(self.actual_percentage_dest),
            "avg_start": str(self.avg_start),
            "avg_end": str(self.avg_end),
            "drop_percentage": str(self.drop),
            "n_locs": str(self.n_locs),
        }
        with open(
            os.path.join(
                self.save_path, self.saving_name + "_analysis_results.txt"
            ),
            "w",
        ) as configfile:
            config.write(configfile)
        return config
