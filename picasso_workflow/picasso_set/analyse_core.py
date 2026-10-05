#!/usr/bin/env python
"""Tier 1 of the picasso-set: core pipeline modules.

:class:`PicassoSetCoreMixin` contributes the ``picasso_*`` core-pipeline
modules (localize, undrift, link, clustering, render, ...) to
:class:`~picasso_workflow.analyse.AutoPicasso`. Each module mirrors the
corresponding picasso CLI command (``picasso/__main__.py``) with identical
parameter names and defaults, operating on the shared ``self.locs`` /
``self.info`` state like the classic modules, so the two sets mix freely in
one workflow. Deliberate deviations from the CLI (e.g. headless figure
saving instead of interactive display) are noted per module.

Output files land in the module's result folder with the CLI-identical
suffix on the base name ``locs`` (e.g. the CLI's ``<base>_link.hdf5``
becomes ``locs_link.hdf5``).

Author: Heinrich Grabmayr
Initial date: October 5, 2026
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loguru import logger

from picasso import __version__ as picassoversion
from picasso import (
    aim,
    clusterer,
    g5m,
    io,
    lib,
    localize,
    postprocess,
    render,
)

from picasso_workflow.module_runtime import module_decorator


class PicassoSetCoreMixin:
    """Core-pipeline picasso-set modules (mixin for ``AutoPicasso``)."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _picasso_set_camera_info(self, parameters):
        """Resolve the camera parameters for ``picasso_localize``.

        Each parameter defaults to ``None`` = "use the resolved camera
        info" (a documented deviation from the CLI literals, which would
        silently override correct metadata). Resolution order per
        parameter: explicit value > the ``AutoPicasso.camera_info``
        property (analysis_config camera_info, or derived from picasso's
        CONFIG via the camera name in the metadata) > movie metadata >
        the CLI literal as last resort.
        """
        cli_defaults = {
            "baseline": 0,
            "sensitivity": 1,
            "gain": 1,
            "qe": 1,
            "pixelsize": 130,
        }
        try:
            resolved = dict(self.camera_info or {})
        except Exception:
            # no camera_info config and no resolvable camera in the
            # metadata (e.g. simulated data)
            resolved = {}
        camera_info = {}
        for par, key in [
            ("baseline", "Baseline"),
            ("sensitivity", "Sensitivity"),
            ("gain", "Gain"),
            ("qe", "Qe"),
            ("pixelsize", "Pixelsize"),
        ]:
            val = parameters.get(par)
            if val is None:
                val = resolved.get(key)
            if val is None:
                val = lib.get_from_metadata(
                    self.info or [], key, cli_defaults[par]
                )
            camera_info[key] = val
        return camera_info

    def _picasso_set_pixelsize(self, default=None):
        """The dataset pixel size in nm, or ``default``.

        Uses the ``AutoPicasso.pixelsize`` property (metadata >
        channel metadata > camera config/picasso CONFIG), falling back
        to ``default`` when none of those resolves.
        """
        try:
            return self.pixelsize
        except Exception:
            return default

    def _picasso_set_save_locs(self, results, filename, key=None):
        """Save ``self.locs``/``self.info`` into the module result folder.

        Returns the file path and records it in ``results`` under
        ``filepath_<key>`` (``key`` defaults to the file's stem). The file
        is also registered as the module's resume checkpoint, so the
        module decorator does not write a second, byte-identical copy
        when ``always_save``/``save_locs`` is set.
        """
        fp = os.path.join(results["folder"], filename)
        self._save_locs(fp)
        key = key or os.path.splitext(filename)[0]
        results[f"filepath_{key}"] = fp
        results.setdefault("checkpoint", {})["single"] = {"filepath": fp}
        return fp

    def _picasso_set_save_drift(self, results, drift, driftfile_name):
        """Save the drift table and its figure into the result folder.

        Mirrors the CLI's ``io.save_drift`` text file; the figure replaces
        the CLI's interactive ``--display`` (headless deviation).
        """
        fp_drift = os.path.join(results["folder"], driftfile_name)
        io.save_drift(fp_drift, drift)
        results["filepath_driftfile"] = fp_drift
        pixelsize = self._picasso_set_pixelsize(1.0)
        fig = plt.figure(figsize=(10, 6), constrained_layout=True)
        postprocess.plot_drift(drift, pixelsize, fig)
        fp_fig = os.path.join(results["folder"], "drift.png")
        fig.savefig(fp_fig)
        results["fp_fig_drift"] = fp_fig

    # ------------------------------------------------------------------
    # localization (CLI: localize)
    # ------------------------------------------------------------------

    @module_decorator
    def picasso_localize(self, i, parameters, results):
        """Identify and fit single-molecule spots (native CLI: localize).

        Wraps ``picasso.localize.localize`` at the CLI's granularity
        (identification and fitting in one step) on the already-loaded
        movie, with the CLI's parameter names. Deviations from the CLI,
        both documented design decisions: ``drift`` defaults to 0 (the CLI
        default 1000 would also undrift; the workflow has dedicated
        undrift modules), and the camera parameters default to ``None`` =
        "use the loaded movie metadata / camera config" instead of the CLI
        literals. The CLI's ``--regions-separately``, ``--concat`` and
        ``--database`` multi-file conveniences are out of scope.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys (CLI defaults unless noted):

            ``fit_method`` : str, default "mle"
                One of the CLI fit methods (mle, mle-gpu, mle-spherical,
                mle-rotated, lq, lq-spherical, lq-rotated, lq-3d, lq-gpu,
                mle-3d, spline, spline-mle, spline-gpu, spline-mle-gpu,
                avg, and the -gpu/-3d combinations).
            ``box_side_length`` : int, default 7
                Side length of the fit box.
            ``gradient`` : float, default 5000
                Minimum net gradient for spot detection.
            ``roi`` : list of int
                One region of interest [y_min, x_min, y_max, x_max].
            ``frame_bounds`` : list
                One or more [start_frame, end_frame] segments (0-indexed,
                inclusive).
            ``temporal_median`` : int, default 0
                Rolling temporal median background filter window.
            ``gaussian_filter`` : float, default 0.0
                Spatial Gaussian pre-filter sigma.
            ``convergence`` : float, default 0
                Fit tolerance; 0 = the fit method's own default.
            ``max_iterations`` : int, default 0
                Max iterations per spot; 0 = the method's own default.
            ``baseline``, ``sensitivity``, ``gain``, ``qe``, ``pixelsize``
                Camera parameters; None = from loaded metadata/config.
            ``spline_calibration`` : str
                Spline PSF calibration .hdf5 (required for spline methods).
            ``affine_calibration`` : str or list of str
                Lateral-correction calibration file(s).
            ``camera_calibration`` : str
                sCMOS camera calibration .hdf5.
            ``zc`` : str
                3D z-calibration file (required for -3d methods).
            ``mf`` : float
                Magnification factor (required for -3d methods).
            ``fit_z_gpu`` : bool, default False
                Fit z on the GPU (falls back to CPU if unavailable).
            ``drift`` : int, default 0 (CLI: 1000)
                RCC segmentation for post-fit undrift; 0 = off.
            ``suffix`` : str
                Inserted into the output file name.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        from picasso.__main__ import _FIT_METHOD_MAP

        fit_method = parameters.get("fit_method", "mle")
        box = parameters.get("box_side_length", 7)
        gradient = parameters.get("gradient", 5000)
        temporal_median = parameters.get("temporal_median", 0)
        gaussian_filter = parameters.get("gaussian_filter", 0.0)
        convergence = parameters.get("convergence", 0)
        max_iterations = parameters.get("max_iterations", 0)
        frame_bounds = parameters.get("frame_bounds")
        suffix = parameters.get("suffix") or ""
        drift = parameters.get("drift", 0)
        fit_z_gpu = parameters.get("fit_z_gpu", False)

        fitting_method = _FIT_METHOD_MAP[fit_method]
        if fitting_method.endswith("-gpu") and not localize.CUDA_AVAILABLE:
            raise RuntimeError(
                "No CUDA-capable GPU found, so the requested GPU fit "
                "method cannot run. Install the [gpu] extra and check the "
                "driver (see README 'GPU-accelerated fitting')."
            )

        camera_info = self._picasso_set_camera_info(parameters)

        roi = parameters.get("roi")
        if roi is not None:
            y_min, x_min, y_max, x_max = roi
            roi = localize.clip_rois(
                [[[y_min, x_min], [y_max, x_max]]], min_size=box
            )

        z_params = None
        if "-3d" in fit_method:
            zc = parameters.get("zc")
            mf = parameters.get("mf", 0)
            if not zc:
                raise ValueError(
                    "3D fitting requires the 'zc' z-calibration file "
                    "parameter."
                )
            if not mf:
                raise ValueError(
                    "3D fitting requires a magnification factor 'mf' > 0."
                )
            z_params = (zc, mf, io.load_calibration(zc))
            if fit_z_gpu:
                from picasso import zfit

                if not zfit.CUDA_AVAILABLE:
                    logger.warning(
                        "GPU z fitting requested (fit_z_gpu) but no "
                        "CUDA-capable GPU is available. Falling back to "
                        "multiprocessed CPU z fitting."
                    )
                    fit_z_gpu = False

        spline_calibration = None
        if fitting_method.startswith("spline"):
            sc = parameters.get("spline_calibration")
            if not sc:
                raise ValueError(
                    "Spline fitting requires the 'spline_calibration' "
                    "parameter (<file.hdf5>). Build one with 'picasso "
                    "spline-calibrate'."
                )
            spline_calibration = io.load_spline_calibration(sc)

        # extra lateral affine corrections, applied after fitting on top of
        # any the 3D / spline calibration carries (mirrors the CLI)
        lateral_transforms = []
        affine = parameters.get("affine_calibration") or []
        if isinstance(affine, str):
            affine = [affine]
        for affine_path in affine:
            found = lib.lateral_transforms(
                io.load_any_calibration(affine_path)
            )
            if not found:
                raise ValueError(
                    f"No lateral corrections found in {affine_path}."
                )
            found, _duplicates = lib.drop_duplicate_lateral_transforms(
                found, lateral_transforms
            )
            lateral_transforms.extend(found)

        camera_calibration = None
        if parameters.get("camera_calibration"):
            camera_calibration = io.load_camera_calibration(
                parameters["camera_calibration"]
            )

        identification_parameters = {
            "Min. Net Gradient": gradient,
            "Box Size": box,
            "Temporal Median Window": temporal_median,
            "Gaussian Filter Sigma": gaussian_filter,
        }
        locs, info = localize.localize(
            self.movie,
            camera_info=camera_info,
            identification_parameters=identification_parameters,
            roi=roi,
            frame_bounds=frame_bounds,
            movie_info=self.info,
            fitting_method=fitting_method,
            eps=convergence if convergence > 0 else None,
            max_it=max_iterations if max_iterations > 0 else None,
            spline_calibration=spline_calibration,
            camera_calibration=camera_calibration,
            threaded=True,
            return_info=True,
        )

        # post-fit steps, mirroring the CLI's _localize_finish
        if z_params is not None:
            from picasso import zfit

            zpath, mf, z_calibration = z_params
            z_calibration["Magnification Factor"] = mf
            method = "gausslq" if "mle" not in fit_method else "gaussmle"
            locs, info = zfit.zfit(
                locs=locs,
                info=info,
                calibration=z_calibration,
                fitting_method=method,
                filter=0,
                lateral_transforms=lateral_transforms,
                multiprocess=not fit_z_gpu,
                gpu=fit_z_gpu,
            )
            info[-1]["Z Calibration Path"] = zpath
        elif lateral_transforms:
            extra, _duplicates = lib.drop_duplicate_lateral_transforms(
                lateral_transforms, spline_calibration
            )
            if extra:
                locs = lib.apply_lateral_transforms(locs, extra)
                info[-1]["Lateral corrections applied"] = info[-1].get(
                    "Lateral corrections applied", []
                ) + lib.describe_lateral_transforms(extra)

        info[-1]["Wrapped by"] = "picasso-workflow : picasso_localize"
        self.locs = locs
        self.info = info
        self._picasso_set_save_locs(results, f"locs{suffix}.hdf5", key="locs")
        results["nlocs"] = len(self.locs)

        if drift > 0:
            undrift_info = {
                "Generated by": f"Picasso v{picassoversion} Undrift",
                "Segmentation": drift,
                "Wrapped by": "picasso-workflow : picasso_localize",
            }
            drift_df, self.locs = postprocess.undrift(
                self.locs, self.info, drift, display=False
            )
            undrift_info["Drift X"] = float(drift_df["x"].mean())
            undrift_info["Drift Y"] = float(drift_df["y"].mean())
            self.drift = drift_df
            self.info.append(undrift_info)
            self._picasso_set_save_drift(results, drift_df, "drift.txt")
            self._picasso_set_save_locs(results, f"locs{suffix}_undrift.hdf5")

        return parameters, results

    # ------------------------------------------------------------------
    # drift correction (CLI: undrift / aim / undrift_fiducials)
    # ------------------------------------------------------------------

    @module_decorator
    def picasso_undrift_rcc(self, i, parameters, results):
        """Correct drift using RCC (native CLI: undrift).

        Wraps ``picasso.postprocess.undrift``. Alternatively applies a
        pre-computed drift file (CLI: ``--fromfile``).

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``segmentation`` : float, default 1000
                Number of frames combined into one temporal segment.
            ``fromfile`` : str
                Apply drift from this drift .txt file instead of
                computing it.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        segmentation = parameters.get("segmentation", 1000)
        fromfile = parameters.get("fromfile")
        undrift_info = {
            "Generated by": f"Picasso v{picassoversion} Undrift",
            "Wrapped by": "picasso-workflow : picasso_undrift_rcc",
        }
        if fromfile is not None:
            undrift_info["From File"] = fromfile
            drift = io.load_drift(fromfile)
            self.locs.x -= drift.loc[self.locs.frame, "x"].to_numpy()
            self.locs.y -= drift.loc[self.locs.frame, "y"].to_numpy()
        else:
            undrift_info["Segmentation"] = segmentation
            drift, self.locs = postprocess.undrift(
                self.locs, self.info, segmentation, display=False
            )
            undrift_info["Drift X"] = float(drift["x"].mean())
            undrift_info["Drift Y"] = float(drift["y"].mean())
        self.drift = drift
        self.info.append(undrift_info)
        self._picasso_set_save_drift(results, drift, "drift.txt")
        self._picasso_set_save_locs(results, "locs_undrift.hdf5")
        results["nlocs"] = len(self.locs)
        return parameters, results

    @module_decorator
    def picasso_undrift_aim(self, i, parameters, results):
        """Correct drift using AIM (native CLI: aim).

        Wraps ``picasso.aim.aim`` (adaptive intersection maximization).

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``segmentation`` : float, default 100
                Number of frames combined into one temporal segment.
            ``intersectdist`` : float, default 20/130
                Max. distance (camera pixels) between localizations in
                consecutive segments to be considered as intersecting.
            ``roiradius`` : float, default 60/130
                Max. drift (camera pixels) between two consecutive
                segments.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        segmentation = parameters.get("segmentation", 100)
        intersectdist = parameters.get("intersectdist", 20 / 130)
        roiradius = parameters.get("roiradius", 60 / 130)
        self.locs, new_info, drift = aim.aim(
            self.locs, self.info, segmentation, intersectdist, roiradius
        )
        self.info = new_info
        self.info.append(
            {"Wrapped by": "picasso-workflow : picasso_undrift_aim"}
        )
        self.drift = drift
        self._picasso_set_save_drift(results, drift, "aimdrift.txt")
        self._picasso_set_save_locs(results, "locs_aim.hdf5")
        results["nlocs"] = len(self.locs)
        return parameters, results

    @module_decorator
    def picasso_undrift_fiducials(self, i, parameters, results):
        """Correct drift using fiducials (native CLI: undrift_fiducials).

        Mirrors the CLI composite: RCC pre-undrift with segmentation 2000
        so fiducials can be detected, then
        ``picasso.postprocess.undrift_from_fiducials``.

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
        segmentation = 2000
        undrift_info = {
            "Generated by": (
                f"Picasso v{picassoversion} Undrift by fiducials"
            ),
            "pre-RCC segmentation": segmentation,
            "Wrapped by": "picasso-workflow : picasso_undrift_fiducials",
        }
        drift, self.locs = postprocess.undrift(
            self.locs, self.info, segmentation, display=False
        )
        self.locs, new_info, drift = postprocess.undrift_from_fiducials(
            self.locs, self.info
        )
        undrift_info["Drift X"] = float(drift["x"].mean())
        undrift_info["Drift Y"] = float(drift["y"].mean())
        undrift_info["Number of picks"] = new_info[-1]["Number of picks"]
        undrift_info["Pick radius (nm)"] = new_info[-1]["Pick radius (nm)"]
        self.info.append(undrift_info)
        self.drift = drift
        self._picasso_set_save_drift(results, drift, "drift_fiducials.txt")
        self._picasso_set_save_locs(results, "locs_undrift_fiducials.hdf5")
        results["nlocs"] = len(self.locs)
        return parameters, results

    # ------------------------------------------------------------------
    # linking / kinetics (CLI: link / dark / groupprops)
    # ------------------------------------------------------------------

    @module_decorator
    def picasso_link(self, i, parameters, results):
        """Link localizations in consecutive frames (native CLI: link).

        Wraps ``picasso.postprocess.link``. The CLI's follow-up update of a
        pre-existing ``_clusters.hdf5`` sibling file is out of scope here.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``distance`` : float, default 1.0
                Maximum distance (camera pixels) between localizations to
                consider them the same binding event.
            ``tolerance`` : int, default 1
                Maximum dark time between localizations to still consider
                them the same binding event.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        distance = parameters.get("distance", 1.0)
        tolerance = parameters.get("tolerance", 1)
        self.locs = postprocess.link(self.locs, self.info, distance, tolerance)
        link_info = {
            "Maximum Distance": distance,
            "Maximum Transient Dark Time": tolerance,
            "Generated by": f"Picasso v{picassoversion} Link",
            "Wrapped by": "picasso-workflow : picasso_link",
        }
        self.info.append(link_info)
        self._picasso_set_save_locs(results, "locs_link.hdf5")
        results["nlocs"] = len(self.locs)
        return parameters, results

    @module_decorator
    def picasso_dark(self, i, parameters, results):
        """Compute dark times for grouped localizations (native CLI: dark).

        Wraps ``picasso.postprocess.compute_dark_times``.

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
        self.locs = postprocess.compute_dark_times(self.locs)
        d_info = {
            "Generated by": f"Picasso v{picassoversion} Dark",
            "Wrapped by": "picasso-workflow : picasso_dark",
        }
        self.info.append(d_info)
        self._picasso_set_save_locs(results, "locs_dark.hdf5")
        results["nlocs"] = len(self.locs)
        return parameters, results

    @module_decorator
    def picasso_groupprops(self, i, parameters, results):
        """Calculate kinetics/properties of localization groups
        (native CLI: groupprops).

        Wraps ``picasso.postprocess.groupprops`` and saves locs + groups
        into one file via ``io.save_datasets``, like the CLI (which appends
        no info entry).

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
        groups = postprocess.groupprops(self.locs)
        fp = os.path.join(results["folder"], "locs_groupprops.hdf5")
        io.save_datasets(fp, self.info, locs=self.locs, groups=groups)
        results["filepath_locs_groupprops"] = fp
        results["n_groups"] = len(groups)
        return parameters, results

    # ------------------------------------------------------------------
    # density / pair correlation (CLI: density / pc)
    # ------------------------------------------------------------------

    @module_decorator
    def picasso_density(self, i, parameters, results):
        """Compute the local localization density (native CLI: density).

        Wraps ``picasso.postprocess.compute_local_density`` exactly like the
        ``picasso density`` CLI command: annotates every localization with
        the number of neighbors within ``radius`` and saves the result as
        ``locs_density.hdf5`` (the CLI's ``_density.hdf5`` output).

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``radius`` : float
                Maximum distance (camera pixels) for localizations to count
                as local.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        radius = float(parameters["radius"])
        self.locs = postprocess.compute_local_density(
            self.locs, self.info, radius
        )
        density_info = {
            "Generated by": f"Picasso v{picassoversion} Density",
            "Radius": radius,
            "Wrapped by": "picasso-workflow : picasso_density",
        }
        self.info.append(density_info)
        self._picasso_set_save_locs(results, "locs_density.hdf5")
        results["nlocs"] = len(self.locs)
        return parameters, results

    @module_decorator
    def picasso_pair_correlation(self, i, parameters, results):
        """Calculate the pair-correlation of localizations
        (native CLI: pc).

        Wraps ``picasso.postprocess.pair_correlation``. The CLI shows the
        curve interactively; here it is saved as a figure plus the raw
        curve data (headless deviation).

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``binsize`` : float, default 0.1
                The bin size (camera pixels).
            ``rmax`` : float, default 10
                The maximum distance to calculate the pair-correlation.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        binsize = parameters.get("binsize", 0.1)
        rmax = parameters.get("rmax", 10)
        bins_lower, pc = postprocess.pair_correlation(
            self.locs, self.info, binsize, rmax
        )
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(bins_lower - binsize / 2, pc)
        ax.set_xlabel("r (pixel)")
        ax.set_ylabel("pair-correlation (pixel^-2)")
        ax.set_title(f"Pair-correlation. Bin size: {binsize}, R max: {rmax}")
        fp_fig = os.path.join(results["folder"], "pair_correlation.png")
        fig.savefig(fp_fig)
        results["fp_fig_pair_correlation"] = fp_fig
        fp_data = os.path.join(results["folder"], "pair_correlation.txt")
        np.savetxt(
            fp_data,
            np.column_stack([bins_lower, pc]),
            header="bins_lower pair_correlation",
        )
        results["filepath_pair_correlation"] = fp_data
        return parameters, results

    # ------------------------------------------------------------------
    # clustering (CLI: dbscan / hdbscan / smlm_cluster / nneighbor /
    # clusterfilter / cluster_combine / cluster_combine_dist)
    # ------------------------------------------------------------------

    @module_decorator
    def picasso_dbscan(self, i, parameters, results):
        """Cluster localizations with DBSCAN (native CLI: dbscan).

        Wraps ``picasso.clusterer.dbscan`` and, like the CLI, also computes
        and saves the cluster centers
        (``picasso.clusterer.find_cluster_centers``).

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``radius`` : float
                Maximal distance (camera pixels) between two localizations
                to be considered local.
            ``density`` : int
                Minimum local density for localizations to be assigned to
                a cluster.

            Optional keys:

            ``pixelsize`` : int
                Camera pixel size in nm (required for 3D localizations
                only).
            ``radius_z`` : float
                DBSCAN epsilon in z (camera pixels). If set, enables
                anisotropic 3D clustering.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        radius = parameters["radius"]
        density = parameters["density"]
        pixelsize = parameters.get("pixelsize")
        radius_z = parameters.get("radius_z")
        self.locs, dbscan_info = clusterer.dbscan(
            self.locs,
            radius,
            density,
            pixelsize=pixelsize,
            radius_z=radius_z,
        )
        clusters = clusterer.find_cluster_centers(self.locs, pixelsize)
        dbscan_info["Wrapped by"] = "picasso-workflow : picasso_dbscan"
        self.info.append(dbscan_info)
        self._picasso_set_save_locs(results, "locs_dbscan.hdf5")
        fp_centers = os.path.join(results["folder"], "locs_dbclusters.hdf5")
        io.save_locs(fp_centers, clusters, self.info)
        results["filepath_locs_dbclusters"] = fp_centers
        results["nclusters"] = len(clusters)
        results["nlocs"] = len(self.locs)
        return parameters, results

    @module_decorator
    def picasso_hdbscan(self, i, parameters, results):
        """Cluster localizations with HDBSCAN (native CLI: hdbscan).

        Wraps ``picasso.clusterer.hdbscan`` and, like the CLI, also saves
        the cluster centers.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``min_cluster`` : int
                Smallest size grouping that is considered a cluster.
            ``min_samples`` : int
                The higher the more points are considered noise.

            Optional keys:

            ``pixelsize`` : int
                Camera pixel size in nm (required for 3D localizations
                only).
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        min_cluster = parameters["min_cluster"]
        min_samples = parameters["min_samples"]
        pixelsize = parameters.get("pixelsize")
        self.locs, hdbscan_info = clusterer.hdbscan(
            self.locs, min_cluster, min_samples, pixelsize
        )
        clusters = clusterer.find_cluster_centers(self.locs, pixelsize)
        hdbscan_info["Wrapped by"] = "picasso-workflow : picasso_hdbscan"
        self.info.append(hdbscan_info)
        self._picasso_set_save_locs(results, "locs_hdbscan.hdf5")
        fp_centers = os.path.join(results["folder"], "locs_hdbclusters.hdf5")
        io.save_locs(fp_centers, clusters, self.info)
        results["filepath_locs_hdbclusters"] = fp_centers
        results["nclusters"] = len(clusters)
        results["nlocs"] = len(self.locs)
        return parameters, results

    @module_decorator
    def picasso_smlm_cluster(self, i, parameters, results):
        """Cluster localizations with the SMLM clusterer
        (native CLI: smlm_cluster).

        Wraps ``picasso.clusterer.cluster`` and, like the CLI, also saves
        the cluster centers.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``radius`` : float
                Clustering radius (camera pixels).
            ``min_locs`` : int
                Minimum number of localizations in a cluster.

            Optional keys:

            ``pixelsize`` : int
                Camera pixel size in nm (required for 3D localizations
                only).
            ``basic_fa`` : bool, default False
                Whether to perform basic frame analysis (sticking event
                removal).
            ``radius_z`` : float
                Clustering radius in axial direction (must be set for 3D).
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        radius = parameters["radius"]
        min_locs = parameters["min_locs"]
        pixelsize = parameters.get("pixelsize")
        basic_fa = parameters.get("basic_fa", False)
        radius_z = parameters.get("radius_z")
        self.locs, smlm_cluster_info = clusterer.cluster(
            self.locs,
            radius_xy=radius,
            radius_z=radius_z,
            min_locs=min_locs,
            frame_analysis=basic_fa,
            pixelsize=pixelsize,
        )
        clusters = clusterer.find_cluster_centers(self.locs, pixelsize)
        smlm_cluster_info["Wrapped by"] = (
            "picasso-workflow : picasso_smlm_cluster"
        )
        self.info.append(smlm_cluster_info)
        self._picasso_set_save_locs(results, "locs_clusters.hdf5")
        fp_centers = os.path.join(
            results["folder"], "locs_cluster_centers.hdf5"
        )
        io.save_locs(fp_centers, clusters, self.info)
        results["filepath_locs_cluster_centers"] = fp_centers
        results["nclusters"] = len(clusters)
        results["nlocs"] = len(self.locs)
        return parameters, results

    @module_decorator
    def picasso_nneighbor(self, i, parameters, results):
        """Calculate nearest-neighbor distances of a clustered dataset
        (native CLI: nneighbor).

        Mirrors the CLI: loads a cluster file, computes each cluster
        center's distance to its nearest neighbor via a KD-tree and saves
        them as a text file (the CLI's ``_minval.txt``). The CLI reads the
        old-style ``clusters`` table (``com_x``/``com_y``); as a workflow
        convenience this module falls back to a cluster-centers locs file
        (``x``/``y``) such as the ones the picasso-set clusterers save.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``files`` : str
                Path to the hdf5 cluster file (e.g. the
                ``filepath_locs_cluster_centers`` result of a picasso-set
                clusterer, via ``$get_prior_result``).
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        from scipy.spatial import KDTree

        path = parameters["files"]
        try:
            clusters = pd.read_hdf(path, key="clusters")
            points = clusters[["com_x", "com_y"]].to_numpy()
        except (KeyError, ValueError):
            centers, _ = io.load_locs(path)
            points = centers[["x", "y"]].to_numpy()
        tree = KDTree(points)
        minvals, _ = tree.query(points, k=2)
        minvals = minvals[:, 1]
        base = os.path.splitext(os.path.basename(path))[0]
        fp = os.path.join(results["folder"], base + "_minval.txt")
        np.savetxt(fp, minvals, newline="\r\n")
        results["filepath_minval"] = fp
        results["n_points"] = len(points)
        results["nn_mean"] = float(np.mean(minvals))
        return parameters, results

    @module_decorator
    def picasso_clusterfilter(self, i, parameters, results):
        """Filter localizations by properties of their clusters
        (native CLI: clusterfilter).

        Mirrors the CLI: splits the current localizations into in-range
        and out-of-range sets by a cluster-property window and saves both
        (the CLI's ``_filter_in.hdf5`` / ``_filter_out.hdf5``). The
        workflow continues with the in-range localizations.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``clusterfile`` : str
                Path to the hdf5 clusterfile.
            ``parameter`` : str
                Cluster parameter to be filtered.
            ``minval`` : float
                Lower boundary.
            ``maxval`` : float
                Upper boundary.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        clusterfile = parameters["clusterfile"]
        parameter = parameters["parameter"]
        minval = parameters["minval"]
        maxval = parameters["maxval"]

        clusters = io.load_clusters(clusterfile)
        selector = (clusters[parameter] > minval) & (
            clusters[parameter] < maxval
        )
        n_in = int(np.sum(selector))
        if n_in == 0 or n_in == len(selector):
            results["success"] = False
            results["message"] = (
                "No localizations in range. Filtering aborted."
                if n_in == 0
                else "All localizations in range. Filtering aborted."
            )
            return parameters, results

        base_info = {
            "Parameter": parameter,
            "Minval": minval,
            "Maxval": maxval,
            "Wrapped by": "picasso-workflow : picasso_clusterfilter",
        }
        all_locs = self.locs
        for tag, sel in (("in", selector), ("out", ~selector)):
            groups = clusters["groups"][sel]
            part = all_locs[all_locs["group"].isin(groups)].sort_values(
                kind="quicksort", by="frame"
            )
            part_info = dict(base_info)
            part_info["Generated by"] = (
                f"Picasso v{picassoversion} Clusterfilter - {tag}"
            )
            fp = os.path.join(results["folder"], f"locs_filter_{tag}.hdf5")
            io.save_locs(fp, part, self.info + [part_info])
            results[f"filepath_locs_filter_{tag}"] = fp
            results[f"nlocs_{tag}"] = len(part)
            if tag == "in":
                in_locs, in_info = part, part_info
        self.locs = in_locs
        self.info.append(in_info)
        return parameters, results

    @module_decorator
    def picasso_cluster_combine(self, i, parameters, results):
        """Combine localizations in each cluster
        (native CLI: cluster_combine).

        Wraps ``picasso.postprocess.cluster_combine``.

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
        self.locs = postprocess.cluster_combine(self.locs)
        combined_info = {
            "Generated by": f"Picasso v{picassoversion} Combine",
            "Wrapped by": "picasso-workflow : picasso_cluster_combine",
        }
        self.info.append(combined_info)
        self._picasso_set_save_locs(results, "locs_comb.hdf5")
        results["nlocs"] = len(self.locs)
        return parameters, results

    @module_decorator
    def picasso_cluster_combine_dist(self, i, parameters, results):
        """Calculate the distance to the nearest neighbor for combined
        clusters (native CLI: cluster_combine_dist).

        Wraps ``picasso.postprocess.cluster_combine_dist``. The pixel size
        is read from the dataset metadata (the CLI reads ``info[1]``; here
        the whole info list is searched).

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
        pixelsize = self._picasso_set_pixelsize()
        self.locs = postprocess.cluster_combine_dist(self.locs, pixelsize)
        cluster_combine_dist_info = {
            "Generated by": f"Picasso v{picassoversion} CombineDist",
            "Wrapped by": ("picasso-workflow : picasso_cluster_combine_dist"),
        }
        self.info.append(cluster_combine_dist_info)
        self._picasso_set_save_locs(results, "locs_cdist.hdf5")
        results["nlocs"] = len(self.locs)
        return parameters, results

    # ------------------------------------------------------------------
    # molecular mapping (CLI: g5m)
    # ------------------------------------------------------------------

    @module_decorator
    def picasso_g5m(self, i, parameters, results):
        """Gaussian Mixture Modeling with Modifications for Molecular
        Mapping (native CLI: g5m).

        Wraps ``picasso.g5m.g5m`` on the current (clustered)
        localizations; the resulting molecule map becomes the workflow's
        current dataset and is saved (the CLI's ``_molmap.hdf5``).

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys (CLI defaults):

            ``min_locs`` : int, default 10
                Min. number of locs per molecule.
            ``loc_prec_handle`` : str, default "local"
                Localization precision handle, "local" or "abs".
            ``min_sigma`` : float, default 0.8
                Minimum sigma factor/value.
            ``max_sigma`` : float, default 1.5
                Maximum sigma factor/value.
            ``max_rounds`` : int, default 3
                Max. rounds without BIC improvement to terminate.
            ``bootstrap_sem`` : bool, default False
                Bootstrap to estimate SEM of molecule positions.
            ``calibration`` : str
                Astigmatism calibration file; used and required only for
                astigmatism 3D data.
            ``mode`` : str, default "astigmatism"
                3D fitting mode of the input locs: "astigmatism" or
                "spline"; ignored for 2D data.
            ``covariance_type`` : str, default "auto"
                One of auto, spherical, diagonal, rotated.
            ``postprocess`` : bool, default True
                Postprocess results to remove sticking events and
                low-quality fits.
            ``max_locs`` : int, default 100000
                Maximum number of localizations per cluster.
            ``asynch`` : bool, default True
                Fit asynchronously (multiprocessing).
            ``group_column`` : str, default "group"
                Column used to group localizations into clusters
                ("group" or "group_input").
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        import yaml

        min_locs = parameters.get("min_locs", 10)
        loc_prec_handle = parameters.get("loc_prec_handle", "local")
        min_sigma = parameters.get("min_sigma", 0.8)
        max_sigma = parameters.get("max_sigma", 1.5)
        max_rounds = parameters.get("max_rounds", 3)
        bootstrap_sem = parameters.get("bootstrap_sem", False)
        calibration = parameters.get("calibration", "")
        mode = parameters.get("mode", "astigmatism")
        covariance_type = parameters.get("covariance_type", "auto")
        postprocess_results = parameters.get("postprocess", True)
        max_locs = parameters.get("max_locs", 100000)
        asynch = parameters.get("asynch", True)
        group_column = parameters.get("group_column", "group")

        calib = None
        # astigmatism 3D data needs a calibration; spline 3D data
        # recovers z directly and needs none (mirrors the CLI)
        if "z" in self.locs.columns and mode == "astigmatism":
            if calibration == "":
                raise ValueError(
                    "A calibration file ('calibration') is required for "
                    "astigmatism 3D data."
                )
            with open(calibration, "r") as f:
                calib = yaml.full_load(f)

        mols, _, g5m_info = g5m.g5m(
            self.locs,
            self.info,
            min_locs=min_locs,
            loc_prec_handle=loc_prec_handle,
            sigma_bounds=(min_sigma, max_sigma),
            max_rounds_without_best_bic=max_rounds,
            bootstrap_check=bootstrap_sem,
            calibration=calib,
            mode=mode,
            covariance_type=covariance_type,
            postprocess=postprocess_results,
            max_locs_per_cluster=max_locs,
            asynch=asynch,
            group_column=group_column,
        )
        g5m_info[-1]["Wrapped by"] = "picasso-workflow : picasso_g5m"
        self.locs = mols
        self.info = g5m_info
        self._picasso_set_save_locs(results, "locs_molmap.hdf5")
        results["n_molecules"] = len(mols)
        return parameters, results

    # ------------------------------------------------------------------
    # rendering (CLI: render)
    # ------------------------------------------------------------------

    @module_decorator
    def picasso_render(self, i, parameters, results):
        """Render a localization-based image (native CLI: render).

        Wraps ``picasso.render.render`` and saves the image as PNG with
        the CLI's colormap scaling. The CLI's colormap default comes from
        the user settings; here it defaults to ``viridis``.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``disp_px_size`` : float, default 10.0
                The size of the rendered pixel in nm.
            ``blur_method`` : str, default "convolve"
                One of "none", "convolve", "gaussian".
            ``min_blur_width`` : float, default 0.0
                Minimum blur width if blur is applied.
            ``vmin`` : float, default 0.0
                Minimum colormap level in range 0-100 or absolute value.
            ``vmax`` : float, default 20.0
                Maximum colormap level in range 0-100 or absolute value.
            ``scaling`` : str, default "yes"
                If "yes", vmin/vmax are relative in the range 0-100.
            ``cmap`` : str, default "viridis"
                One of viridis, inferno, plasma, magma, hot, gray.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        disp_px_size = parameters.get("disp_px_size", 10.0)
        blur_method = parameters.get("blur_method", "convolve")
        min_blur_width = parameters.get("min_blur_width", 0.0)
        vmin = parameters.get("vmin", 0.0)
        vmax = parameters.get("vmax", 20.0)
        scaling = parameters.get("scaling", "yes")
        cmap = parameters.get("cmap") or "viridis"

        if blur_method == "none":
            blur_method = None
        n_rendered, image = render.render(
            self.locs,
            self.info,
            disp_px_size=disp_px_size,
            blur_method=blur_method,
            min_blur_width=min_blur_width,
        )
        fp = os.path.join(results["folder"], "render.png")
        im_max = image.max() / 100
        if scaling == "yes":
            plt.imsave(
                fp, image, vmin=vmin * im_max, vmax=vmax * im_max, cmap=cmap
            )
        else:
            plt.imsave(fp, image, vmin=vmin, vmax=vmax, cmap=cmap)
        results["fp_fig_render"] = fp
        results["n_rendered"] = int(n_rendered)
        return parameters, results

    # ------------------------------------------------------------------
    # multi-file operations (CLI: align / join)
    # ------------------------------------------------------------------

    @module_decorator
    def picasso_align(self, i, parameters, results):
        """Align localization files to each other (native CLI: align).

        Wraps ``picasso.postprocess.align`` (RCC-based). Operates on
        explicit files (like the CLI), not the in-memory localizations;
        the aligned copies are saved into the module folder (the CLI's
        ``_align.hdf5`` suffix).

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``files`` : list of str
                The hdf5 localization files to align.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        files = list(parameters["files"])
        locs_infos = [io.load_locs(fp) for fp in files]
        locs = [li[0] for li in locs_infos]
        infos = [li[1] for li in locs_infos]
        aligned_locs = postprocess.align(locs, infos, display=False)
        align_info = {
            "Generated by": f"Picasso v{picassoversion} Align",
            "Files": files,
            "Wrapped by": "picasso-workflow : picasso_align",
        }
        filepaths = []
        for file, locs_, info in zip(files, aligned_locs, infos):
            info.append(align_info)
            base = os.path.splitext(os.path.basename(file))[0]
            fp = os.path.join(results["folder"], base + "_align.hdf5")
            io.save_locs(fp, locs_, info)
            filepaths.append(fp)
        results["filepaths_aligned"] = filepaths
        return parameters, results

    @module_decorator
    def picasso_join(self, i, parameters, results):
        """Join hdf5 localization lists (native CLI: join).

        Wraps ``picasso.lib.merge_locs``; frame numbers of consecutive
        files are reindexed unless ``keepindex`` is set. The joined
        localizations become the workflow's current dataset.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``files`` : list of str
                The hdf5 localization files to be joined.

            Optional keys:

            ``keepindex`` : bool, default False
                Do not change frame numbers.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        files = list(parameters["files"])
        keepindex = parameters.get("keepindex", False)
        locs_infos = [io.load_locs(file) for file in files]
        all_locs = lib.merge_locs(
            [li[0] for li in locs_infos],
            increment_frames=(not keepindex),
            increment_groups=False,
        )
        join_info = {
            "Generated by": f"Picasso v{picassoversion} Join",
            "Files": files,
            "Wrapped by": "picasso-workflow : picasso_join",
        }
        info = locs_infos[0][1]
        info.append(join_info)
        if not keepindex:
            info[0]["Frames"] = int(all_locs["frame"].max()) + 1
        self.locs = all_locs
        self.info = info
        self._picasso_set_save_locs(results, "locs_join.hdf5")
        results["nlocs"] = len(self.locs)
        return parameters, results
