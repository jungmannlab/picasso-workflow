#!/usr/bin/env python
"""Tier 4 of the picasso-set: file-format converter modules.

:class:`PicassoSetIOMixin` contributes the ``picasso_*`` format
converters to :class:`~picasso_workflow.analyse.AutoPicasso`, mirroring
the picasso CLI conversion commands (``csv2hdf``, ``hdf2csv``, ...) with
CLI-identical output suffixes. The importers (``picasso_csv2hdf``,
``picasso_smap2hdf``) write the converted file next to the input like the
CLI and load it as the workflow's current dataset; the exporters convert
the current localizations by default, or an explicit ``files`` path for
CLI parity, into the module result folder.

Author: Heinrich Grabmayr
Initial date: October 5, 2026
"""

from __future__ import annotations

import os

from picasso import io

from picasso_workflow.module_runtime import module_decorator


class PicassoSetIOMixin:
    """Format-converter picasso-set modules (mixin for ``AutoPicasso``)."""

    def _picasso_set_export(self, parameters, results, out_name, exporter):
        """Run one export: current locs by default, or a ``files`` path.

        ``out_name`` is the CLI-identical output suffix (including the
        extension); the output lands in the module result folder, named
        after the input file (or ``locs`` for the current dataset).
        """
        files = parameters.get("files")
        if files:
            locs, info = io.load_locs(files)
            base = os.path.splitext(os.path.basename(files))[0]
        else:
            locs, info = self.locs, self.info
            base = "locs"
        fp = os.path.join(results["folder"], base + out_name)
        exporter(fp, locs, info)
        results["filepath_converted"] = fp
        results["nlocs"] = len(locs)
        return parameters, results

    # ------------------------------------------------------------------
    # importers (CLI: csv2hdf / smap2hdf)
    # ------------------------------------------------------------------

    @module_decorator
    def picasso_csv2hdf(self, i, parameters, results):
        """Convert a ThunderSTORM CSV file to picasso hdf5
        (native CLI: csv2hdf).

        Wraps ``picasso.io.import_ts``, which writes ``<base>_locs.hdf5``
        next to the input like the CLI; the imported localizations become
        the workflow's current dataset.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``files`` : str
                The ThunderSTORM .csv file.
            ``pixelsize`` : float
                Camera pixel size in nm.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        path = parameters["files"]
        pixelsize = parameters["pixelsize"]
        locs, info = io.import_ts(path, pixelsize)
        self.locs = locs
        self.info = info
        base, _ = os.path.splitext(path)
        results["filepath_converted"] = base + "_locs.hdf5"
        results["nlocs"] = len(locs)
        return parameters, results

    @module_decorator
    def picasso_smap2hdf(self, i, parameters, results):
        """Convert a SMAP _sml.mat file to picasso hdf5
        (native CLI: smap2hdf).

        Wraps ``picasso.io.import_smap`` and saves ``<base>_locs.hdf5``
        next to the input like the CLI; the imported localizations become
        the workflow's current dataset.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Required keys:

            ``files`` : str
                The SMAP _sml.mat file.
            ``pixelsize`` : float
                Camera pixel size in nm.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        path = parameters["files"]
        pixelsize = parameters["pixelsize"]
        locs, info = io.import_smap(path, pixelsize)
        self.locs = locs
        self.info = info
        base, _ = os.path.splitext(path)
        fp = base + "_locs.hdf5"
        io.save_locs(fp, locs, info)
        results["filepath_converted"] = fp
        results["nlocs"] = len(locs)
        return parameters, results

    # ------------------------------------------------------------------
    # exporters (CLI: hdf2csv / hdf2ts / hdf2imagej / hdf2nis /
    # hdf2chimera / hdf2visp / hdf2smap)
    # ------------------------------------------------------------------

    @module_decorator
    def picasso_hdf2csv(self, i, parameters, results):
        """Export localizations to plain CSV, columns unchanged
        (native CLI: hdf2csv).

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``files`` : str
                An hdf5 localizations file to convert instead of the
                current dataset.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        return self._picasso_set_export(
            parameters,
            results,
            ".csv",
            lambda fp, locs, info: locs.to_csv(fp, sep=",", encoding="utf-8"),
        )

    @module_decorator
    def picasso_hdf2ts(self, i, parameters, results):
        """Export localizations to ThunderSTORM CSV
        (native CLI: hdf2ts).

        Wraps ``picasso.io.export_thunderstorm``.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``files`` : str
                An hdf5 localizations file to convert instead of the
                current dataset.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        return self._picasso_set_export(
            parameters, results, ".csv", io.export_thunderstorm
        )

    @module_decorator
    def picasso_hdf2imagej(self, i, parameters, results):
        """Export localizations to ImageJ txt (frame, x, y)
        (native CLI: hdf2imagej).

        Wraps ``picasso.io.export_txt_imagej``.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``files`` : str
                An hdf5 localizations file to convert instead of the
                current dataset.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        return self._picasso_set_export(
            parameters, results, ".txt", io.export_txt_imagej
        )

    @module_decorator
    def picasso_hdf2nis(self, i, parameters, results):
        """Export localizations to NIS txt format
        (native CLI: hdf2nis).

        Wraps ``picasso.io.export_txt_nis``.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``files`` : str
                An hdf5 localizations file to convert instead of the
                current dataset.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        return self._picasso_set_export(
            parameters, results, ".nis.txt", io.export_txt_nis
        )

    @module_decorator
    def picasso_hdf2chimera(self, i, parameters, results):
        """Export localizations to Chimera .xyz (3D visualization)
        (native CLI: hdf2chimera).

        Wraps ``picasso.io.export_xyz_chimera``.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``files`` : str
                An hdf5 localizations file to convert instead of the
                current dataset.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        return self._picasso_set_export(
            parameters, results, ".chi.xyz", io.export_xyz_chimera
        )

    @module_decorator
    def picasso_hdf2visp(self, i, parameters, results):
        """Export localizations to VISP format (native CLI: hdf2visp).

        Wraps ``picasso.io.export_3d_visp``.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``files`` : str
                An hdf5 localizations file to convert instead of the
                current dataset.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        return self._picasso_set_export(
            parameters, results, ".visp.3d", io.export_3d_visp
        )

    @module_decorator
    def picasso_hdf2smap(self, i, parameters, results):
        """Export localizations to SMAP _sml.mat (native CLI: hdf2smap).

        Wraps ``picasso.io.export_smap``.

        Parameters
        ----------
        i : int
            Index of the module in the workflow.
        parameters : dict
            Optional keys:

            ``files`` : str
                An hdf5 localizations file to convert instead of the
                current dataset.
        results : dict
            Module results (see
            :class:`~picasso_workflow.util.AbstractModuleCollection`).
        """
        return self._picasso_set_export(
            parameters, results, "_sml.mat", io.export_smap
        )
