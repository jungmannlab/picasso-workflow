#!/usr/bin/env python
"""Tier 3 of the picasso-set: calibration & 3D modules.

:class:`PicassoSetCalibrationMixin` contributes the ``picasso_*``
calibration modules to :class:`~picasso_workflow.analyse.AutoPicasso`:
3D z fitting/calibration (``picasso.zfit``), sCMOS camera
characterization (``picasso.scmos``), spline PSF calibration
(``picasso.spline``) and lateral-transform calibration
(``picasso.localize``). Parameter names and defaults mirror the picasso
CLI commands (zfit/calibrate_z have no CLI; they follow the library
signatures). These modules are file-in/file-out: calibration files land
in the module result folder and feed ``picasso_localize`` /
``picasso_zfit`` via result references.

The heavier picasso submodules (``zfit``, ``scmos``, ``spline``) are
imported lazily inside the modules, mirroring ``analyse.py``.

Author: Heinrich Grabmayr
Initial date: October 5, 2026
"""

from __future__ import annotations

import os

from picasso import io, localize

from picasso_workflow.module_runtime import module_decorator


class PicassoSetCalibrationMixin:
    """Calibration & 3D picasso-set modules (mixin for ``AutoPicasso``)."""

    @module_decorator
    def picasso_zfit(self, i, parameters, results):
        """Fit z coordinates to 3D (astigmatism) localizations
        (library: picasso.zfit.zfit).

        Wraps ``picasso.zfit.zfit`` on the current localizations with the
        library's parameter names and defaults; the z-fitted localizations
        become the workflow's current dataset.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``calibration`` : str
                The z-calibration .yaml file (e.g. from
                picasso_calibrate_z).

            Optional keys (library defaults):

            ``magnification_factor`` : float
                Refractive-index magnification factor; None = from the
                calibration.
            ``fitting_method`` : str, default "gausslq"
                Noise model of the z fit: "gausslq" or "gaussmle".
            ``filter`` : int, default 2
                Filter level applied to the z fits.
            ``lateral_transforms`` : str
                Calibration file with lateral corrections to apply after
                the z fit.
            ``multiprocess`` : bool, default False
                Fit with multiprocessing.
            ``gpu`` : bool, default False
                Fit on the GPU (falls back to CPU if unavailable).
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        from picasso import zfit

        magnification_factor = parameters.get("magnification_factor")
        fitting_method = parameters.get("fitting_method", "gausslq")
        filter_ = parameters.get("filter", 2)
        lateral_transforms = parameters.get("lateral_transforms")
        multiprocess = parameters.get("multiprocess", False)
        gpu = parameters.get("gpu", False)
        if gpu and not zfit.CUDA_AVAILABLE:
            results["gpu_fallback"] = True
            gpu = False

        calibration = io.load_calibration(parameters["calibration"])
        locs, info = zfit.zfit(
            self.locs,
            self.info,
            calibration=calibration,
            magnification_factor=magnification_factor,
            fitting_method=fitting_method,
            filter=filter_,
            lateral_transforms=lateral_transforms,
            multiprocess=multiprocess,
            gpu=gpu,
        )
        if locs is None:
            results["success"] = False
            results["message"] = "z fitting failed."
            return parameters, results
        info[-1]["Z Calibration Path"] = parameters["calibration"]
        info[-1]["Wrapped by"] = "picasso-workflow : picasso_zfit"
        self.locs = locs
        self.info = info
        self._picasso_set_save_locs(results, "locs_zfit.hdf5")
        results["nlocs"] = len(self.locs)
        return parameters, results

    @module_decorator
    def picasso_calibrate_z(self, i, parameters, results):
        """Build a 3D (astigmatism) z calibration from a fitted bead
        z-stack (library: picasso.zfit.calibrate_z).

        Wraps ``picasso.zfit.calibrate_z`` on the current localizations
        (a bead z-stack localized with an elliptical Gaussian fit) and
        saves the calibration (plus the diagnostic figure picasso writes
        next to it).

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``d`` : float
                z step size in nm between consecutive stage positions.
            ``magnification_factor`` : float
                Refractive-index magnification factor.

            Optional keys (library defaults):

            ``frame_bounds`` : list
                [start_frame, end_frame] restriction of the stack.
            ``frames_per_step`` : int, default 1
                Frames acquired per z position (multi-FOV).
            ``frame_order`` : str, default "fov"
                Acquisition order when frames_per_step > 1 ("fov"/"z").
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        from picasso import zfit

        d = parameters["d"]
        magnification_factor = parameters["magnification_factor"]
        frame_bounds = parameters.get("frame_bounds")
        frames_per_step = parameters.get("frames_per_step", 1)
        frame_order = parameters.get("frame_order", "fov")

        fp = os.path.join(results["folder"], "z_calibration.yaml")
        calibration = zfit.calibrate_z(
            self.locs,
            self.info,
            d,
            magnification_factor,
            path=fp,
            frame_bounds=frame_bounds,
            frames_per_step=frames_per_step,
            frame_order=frame_order,
        )
        if not os.path.isfile(fp):
            # older picasso versions only return the dict when no path
            # handling applies; make sure the file exists either way
            io.save_calibration(fp, calibration)
        results["filepath_z_calibration"] = fp
        results["num_frames"] = calibration.get("Number of frames")
        return parameters, results

    @module_decorator
    def picasso_camera_calibrate(self, i, parameters, results):
        """Characterize an sCMOS camera: offset, variance and optional
        gain maps (native CLI: camera-calibrate).

        Wraps ``picasso.scmos.calibrate_scmos`` with the CLI's parameter
        names; the calibration is saved into the module folder.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``dark`` : str
                Dark movie: frames recorded with no light on the sensor.

            Optional keys:

            ``light`` : list of str
                Movies at quasi-uniform illumination levels (one per
                level); omit for offset and variance only.
            ``power`` : list of float
                Illumination level of each light movie, same order.
            ``power_unit`` : str, default "mW"
                Unit of ``power``, for the diagnostic plot axis.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        from picasso import scmos

        dark = parameters["dark"]
        light = parameters.get("light") or []
        powers = parameters.get("power") or None
        power_unit = parameters.get("power_unit", "mW")
        if powers is not None and len(powers) != len(light):
            raise ValueError(
                f"Got {len(powers)} 'power' value(s) for {len(light)} "
                "light movie(s). Pass one per movie, in the same order, "
                "or none at all."
            )

        dark_movie, _ = io.load_movie(dark)
        bright_movies = []
        for path in light:
            movie, _ = io.load_movie(path)
            bright_movies.append(movie)

        calibration = scmos.calibrate_scmos(
            dark_movie,
            bright_movies or None,
            dark_path=dark,
            bright_paths=light,
            bright_levels=powers,
            level_unit=power_unit,
        )
        fp = os.path.join(results["folder"], "scmos_calib.hdf5")
        calibration["Path"] = fp
        io.save_camera_calibration(fp, calibration)
        results["filepath_camera_calibration"] = fp
        results["frames_used"] = calibration.get("Frames")
        results["offset_median_adu"] = calibration.get("Offset median (ADU)")
        results["variance_median_adu2"] = calibration.get(
            "Variance median (ADU^2)"
        )
        results["hot_pixels"] = calibration.get("Hot pixels")
        if calibration.get("gain") is not None:
            results["gain_median_adu_per_e"] = calibration.get(
                "Gain median (ADU/e-)"
            )
        return parameters, results

    @module_decorator
    def picasso_camera_validate(self, i, parameters, results):
        """Check an sCMOS calibration against a fresh dark movie
        (native CLI: camera-validate).

        Wraps ``picasso.scmos.validate_calibration``; the report lands in
        the results (the workflow continues either way, with ``valid``
        recording the verdict).

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``calibration`` : str
                Camera calibration (.hdf5).
            ``movie`` : str
                Short fresh dark movie (about 1,000 frames is plenty).
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        from picasso import scmos

        calibration = io.load_camera_calibration(parameters["calibration"])
        test_movie, _ = io.load_movie(parameters["movie"])
        report = scmos.validate_calibration(calibration, test_movie)
        results["valid"] = bool(report["valid"])
        results["frames_tested"] = report.get("Frames")
        results["pixels_tested"] = report.get("Pixels tested")
        results["mean_p_value"] = float(report["mean p-value"])
        results["fraction_p_below_0.05"] = float(report["fraction p < 0.05"])
        results["fraction_p_above_0.95"] = float(report["fraction p > 0.95"])
        return parameters, results

    @module_decorator
    def picasso_spline_calibrate(self, i, parameters, results):
        """Build a cubic-spline PSF calibration from bead z-stack(s)
        (native CLI: spline-calibrate).

        Wraps ``picasso.spline.calibrate_spline`` (one movie),
        ``calibrate_spline_split_fov`` (one movie, ``split_fov`` regions
        as channels) or ``calibrate_spline_multichannel`` (several
        movies), exactly like the CLI dispatches. CLI parameter names and
        defaults throughout (camera parameters are the CLI literals here;
        calibration stacks are standalone acquisitions).

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``files`` : list of str
                Bead z-stack movie file(s); several = one per channel.
            ``step`` : float
                z step size in nm between consecutive stage positions.

            Optional keys (CLI defaults):

            ``box_side_length`` : int, default 13
            ``gradient`` : int, default 5000
            ``frames_per_step`` : int, default 1
            ``frame_order`` : str, default "fov" ("fov"/"z")
            ``registration_model`` : str, default "affine"
                Channel registration transform (multichannel/split-FOV).
            ``model`` : str, default "spline-3d" ("spline-3d"/"spline-2d")
            ``magnification_factor`` : float, default 0.79
            ``correct_z_bias`` : bool, default False
            ``photon_ratios`` : str, default ""
                Ratiometric candidates, e.g. "0.7,0.3;0.4,0.6".
            ``split_fov`` : str, default ""
                Single-movie multichannel regions,
                e.g. "0,0,512,256;0,256,512,512".
            ``reference`` : int, default 0
                Split-FOV reference region index.
            ``baseline`` : float, default 0
            ``sensitivity`` : float, default 1
            ``gain`` : int, default 1
            ``pixelsize`` : int, default 130
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        from picasso import spline

        files = list(parameters["files"])
        step = parameters["step"]
        box = parameters.get("box_side_length", 13)
        gradient = parameters.get("gradient", 5000)
        frames_per_step = parameters.get("frames_per_step", 1)
        frame_order = parameters.get("frame_order", "fov")
        registration = parameters.get("registration_model") or "affine"
        model = parameters.get("model", "spline-3d")
        magnification_factor = parameters.get("magnification_factor", 0.79)
        correct_z_bias = parameters.get("correct_z_bias", False)
        reference = parameters.get("reference", 0) or 0
        camera_info = {
            "Baseline": parameters.get("baseline", 0),
            "Sensitivity": parameters.get("sensitivity", 1),
            "Gain": parameters.get("gain", 1),
            "Pixelsize": parameters.get("pixelsize", 130),
        }
        fp = os.path.join(results["folder"], "spline_calib.hdf5")

        # "0.7,0.3;0.4,0.6" -> [[0.7, 0.3], [0.4, 0.6]] (mirrors the CLI)
        photon_ratios = None
        if parameters.get("photon_ratios"):
            photon_ratios = [
                [float(v) for v in row.split(",")]
                for row in parameters["photon_ratios"].split(";")
                if row.strip()
            ]

        split_fov = parameters.get("split_fov")
        if split_fov and len(files) == 1:
            # "y0,x0,y1,x1;..." -> [[[y0,x0],[y1,x1]], ...] (mirrors CLI)
            regions = []
            for row in split_fov.split(";"):
                if not row.strip():
                    continue
                v = [int(t) for t in row.split(",")]
                if len(v) != 4:
                    raise ValueError(
                        "Each split_fov region needs 4 ints y0,x0,y1,x1; "
                        f"got '{row}'."
                    )
                regions.append([[v[0], v[1]], [v[2], v[3]]])
            movie, info = io.load_movie(files[0])
            calibration = spline.calibrate_spline_split_fov(
                movie,
                info=info,
                camera_info=camera_info,
                box=box,
                minimum_ng=gradient,
                d=step,
                regions=regions,
                reference=reference,
                frames_per_step=frames_per_step,
                frame_order=frame_order,
                magnification_factor=magnification_factor,
                correct_z_bias=correct_z_bias,
                photon_ratios=photon_ratios,
                model=registration,
                path=fp,
            )
        elif len(files) == 1:
            movie, info = io.load_movie(files[0])
            calibration = spline.calibrate_spline(
                movie,
                info=info,
                camera_info=camera_info,
                box=box,
                minimum_ng=gradient,
                d=step,
                frames_per_step=frames_per_step,
                frame_order=frame_order,
                model=model,
                magnification_factor=magnification_factor,
                correct_z_bias=correct_z_bias,
                path=fp,
            )
        else:
            movies, infos, camera_infos = [], [], []
            for f in files:
                movie, info = io.load_movie(f)
                movies.append(movie)
                infos.append(info)
                camera_infos.append(dict(camera_info))
            calibration = spline.calibrate_spline_multichannel(
                movies,
                infos=infos,
                camera_infos=camera_infos,
                box=box,
                minimum_ng=gradient,
                d=step,
                frames_per_step=frames_per_step,
                frame_order=frame_order,
                magnification_factor=magnification_factor,
                correct_z_bias=correct_z_bias,
                photon_ratios=photon_ratios,
                model=registration,
                path=fp,
            )
        results["filepath_spline_calibration"] = fp
        results["spline_model"] = calibration.get("model")
        results["n_channels"] = len(files) if not split_fov else None
        return parameters, results

    @module_decorator
    def picasso_lateral_calibrate(self, i, parameters, results):
        """Fit a lateral (astigmatism / chromatic) x-y correction from
        two bead images (native CLI: lateral-calibrate).

        Wraps ``picasso.localize.fit_lateral_transform`` with the CLI's
        parameter names; the calibration and the diagnostic figure land in
        the module folder (unless appending to an existing calibration
        via ``calibration``/``output``).

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``reference`` : str
                Reference bead image (without the cylindrical lens, or in
                the reference color channel).
            ``target`` : str
                Bead image to be mapped onto the reference.

            Optional keys (CLI defaults):

            ``type`` : str, default "astigmatism"
                What the transform corrects ("astigmatism"/"chromatic").
            ``model`` : str, default "affine"
                Transform model (translation, affine, projective,
                polynomial2, polynomial3).
            ``calibration`` : str, default ""
                Existing calibration (.yaml or .hdf5) to append to.
            ``output`` : str, default ""
                Where to write the calibration; defaults to
                ``calibration``, else to the module folder.
            ``box_side_length`` : int, default 7
            ``gradient`` : int, default 5000
            ``pixelsize`` : float
                Camera pixel size in nm, for the reported shift in nm.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        reference = parameters["reference"]
        target = parameters["target"]
        transform_type = parameters.get("type", "astigmatism")
        model = parameters.get("model", "affine")
        calibration_path = parameters.get("calibration", "")
        box = parameters.get("box_side_length", 7)
        gradient = parameters.get("gradient", 5000)
        pixelsize = parameters.get("pixelsize")

        movie_ref, _ = io.load_movie(reference)
        movie_target, _ = io.load_movie(target)
        calibration = (
            io.load_any_calibration(calibration_path)
            if calibration_path
            else {}
        )
        out_path = parameters.get("output") or calibration_path
        if not out_path:
            out_path = os.path.join(results["folder"], "lateral_calib.yaml")

        calibration, qc = localize.fit_lateral_transform(
            movie_ref,
            movie_target,
            calibration,
            box=box,
            minimum_ng=gradient,
            pixelsize=pixelsize,
            transform_type=transform_type,
            ref_path=reference,
            target_path=target,
            model=model,
        )
        fp_fig = os.path.join(results["folder"], "lateral_calibration.png")
        localize.plot_lateral_calibration(qc, save_path=fp_fig)
        results["fp_fig_lateral_calibration"] = fp_fig
        io.save_any_calibration(out_path, calibration)
        results["filepath_lateral_calibration"] = out_path
        results["n_bead_pairs"] = qc.get("n_pairs")
        return parameters, results
