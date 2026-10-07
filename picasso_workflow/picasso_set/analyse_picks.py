#!/usr/bin/env python
"""Tier 2 of the picasso-set: pick-based postprocessing modules.

:class:`PicassoSetPicksMixin` contributes the ``picasso_*`` pick/mask
postprocessing modules to :class:`~picasso_workflow.analyse.AutoPicasso`.
These operations have no picasso CLI; they mirror the picasso Render GUI
operations with the underlying library functions' parameter names and
defaults (``picasso.postprocess`` / ``picasso.masking``).

Picks are exchanged headless as **file paths**: modules take a
``picks_file`` parameter pointing to a picasso pick-region ``.yaml``
(loaded via ``picasso.io.load_picks``), referenceable from a prior module
via ``("$get_prior_result", ...)``; ``picasso_pick_similar`` writes such a
file.

Author: Heinrich Grabmayr
Initial date: October 5, 2026
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from loguru import logger

from picasso import __version__ as picassoversion
from picasso import io, lib, postprocess

from picasso_workflow.module_runtime import module_decorator


class PicassoSetPicksMixin:
    """Pick-based postprocessing picasso-set modules (AutoPicasso mixin)."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _picasso_set_load_picks(self, picks_file):
        """Load a pick-region yaml, sizes converted to camera pixels.

        Returns ``(picks, pick_shape, pick_size)`` from
        ``picasso.io.load_picks``, using the dataset's pixel size for the
        nm-to-camera-pixel conversion of stored pick sizes.
        """
        pixelsize = self._picasso_set_pixelsize()
        return io.load_picks(picks_file, pixelsize=pixelsize)

    def _picasso_set_save_picks(
        self, results, picks, pick_shape, pick_size, filename
    ):
        """Save picks as a pick-region yaml loadable by ``io.load_picks``.

        Sizes are written in camera pixels using the legacy keys
        (``Diameter``/``Width``); square picks, whose size is only ever
        stored in nm (the ``Side Length (nm)`` key), are converted back
        via the dataset's pixel size.
        """
        picks_list = [np.asarray(pick).tolist() for pick in picks]
        data = {"Shape": pick_shape}
        if pick_shape == "Circle":
            data["Centers"] = picks_list
            data["Diameter"] = float(pick_size)
        elif pick_shape == "Rectangle":
            data["Center-Axis-Points"] = picks_list
            data["Width"] = float(pick_size)
        elif pick_shape == "Square":
            pixelsize = self._picasso_set_pixelsize(1.0)
            data["Centers"] = picks_list
            # the bare "Side Length" key would be read back as camera
            # pixels; the nm value belongs under "Side Length (nm)"
            data["Side Length (nm)"] = float(pick_size * pixelsize)
        elif pick_shape == "Box":
            data["Corners"] = picks_list
        else:
            raise ValueError(
                f"Cannot save picks of shape {pick_shape!r} to a "
                "pick-region file."
            )
        fp = os.path.join(results["folder"], filename)
        with open(fp, "w") as f:
            yaml.dump(data, f)
        results["filepath_picks"] = fp
        return fp

    def _picasso_set_picked_locs(self, picks_file, add_group=True):
        """Load picks and return the per-pick localization tables."""
        picks, pick_shape, pick_size = self._picasso_set_load_picks(picks_file)
        picked = postprocess.picked_locs(
            self.locs,
            self.info,
            picks,
            pick_shape,
            pick_size=pick_size,
            add_group=add_group,
        )
        return picked, picks, pick_shape, pick_size

    # ------------------------------------------------------------------
    # pick operations (Render GUI: Pick / Pick similar / Remove picks)
    # ------------------------------------------------------------------

    @module_decorator
    def picasso_picked_locs(self, i, parameters, results):
        """Keep only the localizations inside the given picks
        (Render GUI: picked locs).

        Wraps ``picasso.postprocess.picked_locs``; the picked
        localizations (grouped per pick) become the workflow's current
        dataset.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``picks_file`` : str
                Picasso pick-region .yaml file.

            Optional keys:

            ``add_group`` : bool, default True
                Add a group column indexing the picks.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        add_group = parameters.get("add_group", True)
        picked, picks, pick_shape, pick_size = self._picasso_set_picked_locs(
            parameters["picks_file"], add_group=add_group
        )
        if not len(picked):
            raise ValueError("No localizations in the given picks.")
        self.locs = pd.concat(picked, ignore_index=True)
        self.info.append(
            {
                "Generated by": f"Picasso v{picassoversion} Render : Pick",
                "Pick Shape": pick_shape,
                "Pick Size": pick_size,
                "Number of picks": len(picks),
                "Wrapped by": "picasso-workflow : picasso_picked_locs",
            }
        )
        self._picasso_set_save_locs(results, "locs_picked.hdf5")
        results["n_picks"] = len(picks)
        results["nlocs"] = len(self.locs)
        return parameters, results

    @module_decorator
    def picasso_pick_similar(self, i, parameters, results):
        """Find regions similar to the given picks
        (Render GUI: pick similar).

        Wraps ``picasso.postprocess.pick_similar`` and writes the found
        picks as a pick-region yaml for downstream pick modules.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``picks_file`` : str
                Picasso pick-region .yaml file with the seed picks
                (Circle, Rectangle, Square or Box).

            Optional keys:

            ``std_range`` : float, default 2.0
                Number of standard deviations the number of locs and the
                RMSD may deviate from the seed picks' mean.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        std_range = parameters.get("std_range", 2.0)
        picks, pick_shape, pick_size = self._picasso_set_load_picks(
            parameters["picks_file"]
        )
        similar = postprocess.pick_similar(
            self.locs,
            self.info,
            picks,
            pick_shape=pick_shape,
            pick_size=pick_size,
            std_range=std_range,
        )
        self._picasso_set_save_picks(
            results, similar, pick_shape, pick_size, "picks_similar.yaml"
        )
        results["n_picks_input"] = len(picks)
        results["n_picks_similar"] = len(similar)
        return parameters, results

    @module_decorator
    def picasso_remove_locs_in_picks(self, i, parameters, results):
        """Remove the localizations inside the given picks
        (Render GUI: remove picked locs).

        Wraps ``picasso.postprocess.remove_locs_in_picks``.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``picks_file`` : str
                Picasso pick-region .yaml file.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        picks, pick_shape, pick_size = self._picasso_set_load_picks(
            parameters["picks_file"]
        )
        self.locs = postprocess.remove_locs_in_picks(
            self.locs,
            self.info,
            picks=picks,
            pick_shape=pick_shape,
            pick_size=pick_size,
        )
        self.info.append(
            {
                "Generated by": (
                    f"Picasso v{picassoversion} Render : Remove picks"
                ),
                "Pick Shape": pick_shape,
                "Number of picks": len(picks),
                "Wrapped by": (
                    "picasso-workflow : picasso_remove_locs_in_picks"
                ),
            }
        )
        self._picasso_set_save_locs(results, "locs_picks_removed.hdf5")
        results["nlocs"] = len(self.locs)
        return parameters, results

    # ------------------------------------------------------------------
    # pick analysis (Render GUI: pick properties / kinetics / FRET)
    # ------------------------------------------------------------------

    @module_decorator
    def picasso_pick_properties(self, i, parameters, results):
        """Calculate statistical properties per pick
        (Render GUI: save pick properties).

        Wraps ``picasso.postprocess.pick_properties`` on the picked
        localizations and saves the per-pick property table. The current
        localizations are not modified.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``picks_file`` : str
                Picasso pick-region .yaml file.

            Optional keys:

            ``max_dark_time`` : int, default 3
                Maximum dark time for linking binding events.
            ``influx_rate`` : float, default 0.03
                Influx rate for qPAINT unit calibration.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        max_dark_time = parameters.get("max_dark_time", 3)
        influx_rate = parameters.get("influx_rate", 0.03)
        picked, picks, _, _ = self._picasso_set_picked_locs(
            parameters["picks_file"]
        )
        props = postprocess.pick_properties(
            picked,
            self.info,
            max_dark_time=max_dark_time,
            influx_rate=influx_rate,
        )
        fp = os.path.join(results["folder"], "pick_properties.hdf5")
        io.save_datasets(fp, self.info, groups=props)
        results["filepath_pick_properties"] = fp
        results["n_picks"] = len(picks)
        return parameters, results

    @module_decorator
    def picasso_pick_kinetics(self, i, parameters, results):
        """Estimate binding kinetics per pick
        (Render GUI: pick kinetics).

        Wraps ``picasso.postprocess.pick_kinetics``; the localizations of
        the picks with estimable kinetics (with added ``length``/``dark``/
        ``n`` columns) become the workflow's current dataset.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``picks_file`` : str
                Picasso pick-region .yaml file.

            Optional keys:

            ``max_dark_time`` : int, default 3
                Maximum dark time for linking binding events.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        max_dark_time = parameters.get("max_dark_time", 3)
        picked, picks, _, _ = self._picasso_set_picked_locs(
            parameters["picks_file"]
        )
        length, dark, no_locs, out_locs, kept = postprocess.pick_kinetics(
            picked, self.info, max_dark_time=max_dark_time
        )
        self.locs = out_locs
        self.info.append(
            {
                "Generated by": (
                    f"Picasso v{picassoversion} Render : Pick kinetics"
                ),
                "Maximum dark time": max_dark_time,
                "Wrapped by": "picasso-workflow : picasso_pick_kinetics",
            }
        )
        self._picasso_set_save_locs(results, "locs_pick_kinetics.hdf5")
        results["n_picks"] = len(picks)
        results["n_picks_kept"] = len(kept)
        results["mean_length_frames"] = float(np.mean(length))
        results["mean_dark_frames"] = float(np.mean(dark))
        results["nlocs"] = len(self.locs)
        return parameters, results

    @module_decorator
    def picasso_fret(self, i, parameters, results):
        """Calculate FRET efficiencies from a donor and an acceptor
        dataset (Render GUI: calculate FRET).

        Wraps ``picasso.postprocess.calculate_fret`` on two explicit
        localization files; the current localizations are not modified.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``acc_locs_file`` : str
                The acceptor localizations hdf5 file.
            ``don_locs_file`` : str
                The donor localizations hdf5 file.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        acc_locs, _acc_info = io.load_locs(parameters["acc_locs_file"])
        don_locs, don_info = io.load_locs(parameters["don_locs_file"])
        fret_dict, f_locs = postprocess.calculate_fret(acc_locs, don_locs)
        n_events = len(fret_dict["fret_events"])
        results["n_fret_events"] = n_events
        if n_events == 0:
            results["success"] = False
            results["message"] = "No FRET events found."
            return parameters, results
        results["mean_fret"] = float(np.mean(fret_dict["fret_events"]))
        fret_info = don_info + [
            {
                "Generated by": f"Picasso v{picassoversion} Render : FRET",
                "Wrapped by": "picasso-workflow : picasso_fret",
            }
        ]
        fp = os.path.join(results["folder"], "locs_fret.hdf5")
        io.save_locs(fp, f_locs, fret_info)
        results["filepath_locs_fret"] = fp
        fp_events = os.path.join(results["folder"], "fret_events.txt")
        np.savetxt(
            fp_events,
            np.column_stack(
                [fret_dict["fret_timepoints"], fret_dict["fret_events"]]
            ),
            header="frame fret_efficiency",
        )
        results["filepath_fret_events"] = fp_events
        return parameters, results

    # ------------------------------------------------------------------
    # masking (Render GUI: mask image)
    # ------------------------------------------------------------------

    @module_decorator
    def picasso_mask_locs(self, i, parameters, results):
        """Split localizations by a density mask
        (Render GUI: mask image).

        Mirrors the Render GUI mask flow with ``picasso.masking``: render
        a normalized image (``generate_image``), threshold it into a
        binary mask (``mask_image``), and split the localizations into
        inside/outside (``mask_locs``). The workflow continues with the
        localizations inside the mask.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``disp_px_size`` : float
                Size of the rendered mask pixel in nm.
            ``blur`` : float
                Gaussian blur sigma applied to the rendered image
                (display pixels).

            Optional keys:

            ``method`` : str or float, default "otsu"
                Thresholding method (isodata, li, mean, minimum, otsu,
                triangle, yen, local_gaussian, local_mean, local_median)
                or an explicit threshold value in (0, 1).
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        from picasso import masking

        disp_px_size = parameters["disp_px_size"]
        blur = parameters["blur"]
        method = parameters.get("method", "otsu")
        if isinstance(method, str):
            # an explicit threshold may arrive as text (e.g. from a
            # hand-edited workflow yaml); mask_image expects a float then
            try:
                method = float(method)
            except ValueError:
                pass

        image = masking.generate_image(
            self.locs, self.info, disp_px_size, blur
        )
        mask, threshold = masking.mask_image(image, method)
        locs_in, locs_out = masking.mask_locs(self.locs, self.info, mask)

        fp_mask = os.path.join(results["folder"], "mask.npy")
        np.save(fp_mask, mask)
        results["filepath_mask"] = fp_mask
        fp_fig = os.path.join(results["folder"], "mask.png")
        plt.imsave(fp_fig, mask, cmap="gray")
        results["fp_fig_mask"] = fp_fig

        mask_info = {
            "Generated by": f"Picasso v{picassoversion} Render : Mask",
            "Mask display pixel size (nm)": disp_px_size,
            "Mask blur": blur,
            "Mask method": str(method),
            "Wrapped by": "picasso-workflow : picasso_mask_locs",
        }
        if np.isscalar(threshold):
            mask_info["Mask threshold"] = float(threshold)
            results["threshold"] = float(threshold)
        out_info = self.info + [mask_info]
        fp_out = os.path.join(results["folder"], "locs_mask_out.hdf5")
        io.save_locs(fp_out, locs_out, out_info)
        results["filepath_locs_mask_out"] = fp_out

        self.locs = locs_in
        self.info.append(mask_info)
        self._picasso_set_save_locs(results, "locs_mask_in.hdf5")
        results["nlocs_in"] = len(locs_in)
        results["nlocs_out"] = len(locs_out)
        return parameters, results

    # ------------------------------------------------------------------
    # resolution estimates (Render GUI: NeNA / FRC)
    # ------------------------------------------------------------------

    @module_decorator
    def picasso_nena(self, i, parameters, results):
        """Estimate the localization precision via NeNA
        (Render GUI: NeNA).

        Wraps ``picasso.postprocess.nena``. The current localizations are
        not modified.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            (none)
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        _result, s = postprocess.nena(self.locs, self.info)
        results["nena_px"] = float(s)
        pixelsize = self._picasso_set_pixelsize()
        if pixelsize is not None:
            results["nena_nm"] = float(s) * pixelsize
        return parameters, results

    @module_decorator
    def picasso_frc(self, i, parameters, results):
        """Estimate the image resolution via Fourier Ring Correlation
        (Render GUI: FRC).

        Wraps ``picasso.postprocess.frc`` over the full field of view and
        saves the FRC curve. The current localizations are not modified.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``random_seed`` : int, default 42
                Seed for the random split of the localizations.
            ``viewport`` : list, default the full field of view
                Region to run FRC on, as
                ``[[y_min, x_min], [y_max, x_max]]`` in camera pixels.
            ``max_image_px`` : int, default 16384
                Memory guard: maximum side length (binned pixels) of the
                two rendered half-images. FRC bins at half the NeNA
                precision, so a full modern sensor at good precision can
                exceed 100k px per side -- tens of GB per image, which
                gets the job OOM-killed. A viewport larger than this is
                cropped centrally (recorded in the results as ``note``).
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        random_seed = parameters.get("random_seed", 42)
        max_image_px = parameters.get("max_image_px", 16384)
        if (viewport := parameters.get("viewport")) is not None:
            viewport = (
                (float(viewport[0][0]), float(viewport[0][1])),
                (float(viewport[1][0]), float(viewport[1][1])),
            )
        else:
            height = lib.get_from_metadata(
                self.info, "Height", raise_error=True
            )
            width = lib.get_from_metadata(self.info, "Width", raise_error=True)
            viewport = ((0, 0), (height, width))
        # Estimate the rendered image size the same way postprocess.frc
        # does (bin size = NeNA / 2; the viewport is squared to its smaller
        # side) and crop the viewport centrally when it would exceed the
        # memory guard.
        lp = postprocess.nena(self.locs, self.info)[1]
        binsize = lp / 2
        side = min(
            viewport[1][0] - viewport[0][0],
            viewport[1][1] - viewport[0][1],
        )
        est_px = side / binsize
        if est_px > max_image_px:
            new_side = max_image_px * binsize
            y_c = (viewport[0][0] + viewport[1][0]) / 2
            x_c = (viewport[0][1] + viewport[1][1]) / 2
            viewport = (
                (y_c - new_side / 2, x_c - new_side / 2),
                (y_c + new_side / 2, x_c + new_side / 2),
            )
            note = (
                f"FRC images would be ~{est_px:.0f} px wide (bin size = "
                f"NeNA/2 = {binsize:.4f} camera px); cropped the viewport "
                f"centrally to {new_side:.1f} camera px to respect "
                f"max_image_px={max_image_px}. Pass a 'viewport' to choose "
                "the region, or raise 'max_image_px' (memory scales with "
                "its square)."
            )
            logger.warning(note)
            results["note"] = note
        results["viewport"] = [list(viewport[0]), list(viewport[1])]
        frc_result = postprocess.frc(
            self.locs, self.info, viewport, random_seed=random_seed
        )
        results["resolution_nm"] = float(frc_result["resolution"])

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(
            frc_result["frequencies"],
            frc_result["frc_curve"],
            label="FRC",
            alpha=0.5,
        )
        ax.plot(
            frc_result["frequencies"],
            frc_result["frc_curve_smooth"],
            label="FRC (smoothed)",
        )
        ax.axhline(1 / 7, color="gray", linestyle="--", label="1/7")
        ax.set_xlabel("spatial frequency (nm$^{-1}$)")
        ax.set_ylabel("FRC")
        ax.set_title(f"FRC resolution: {frc_result['resolution']:.1f} nm")
        ax.legend()
        fp_fig = os.path.join(results["folder"], "frc.png")
        fig.savefig(fp_fig)
        results["fp_fig_frc"] = fp_fig
        fp_curve = os.path.join(results["folder"], "frc.txt")
        np.savetxt(
            fp_curve,
            np.column_stack(
                [
                    frc_result["frequencies"],
                    frc_result["frc_curve"],
                    frc_result["frc_curve_smooth"],
                ]
            ),
            header="frequency_per_nm frc frc_smooth",
        )
        results["filepath_frc"] = fp_curve
        return parameters, results
