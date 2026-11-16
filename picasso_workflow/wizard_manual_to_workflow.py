#!/usr/bin/env python
"""GUI wizard: translate a manual analysis into a picasso-workflow.

A thin PyQt6 front-end over :mod:`picasso_workflow.manual_to_workflow`. The user
browses straight to the manual-analysis folder and chooses whether to
re-localize from the raw movies or start downstream from the localized files;
the engine reconstructs the equivalent aggregation workflow (off the UI thread,
with progress and a Stop button); the wizard shows a per-step review table with
confidence flags and consistency findings; on *Apply* it writes the
reconstructed Files tree and module lists into the main GUI so the user can
review and launch via the normal Run tab.

The GUI-population step is factored into :func:`apply_result_to_window`, a
Qt-free helper that only touches documented ``Window`` attributes/methods, so it
can be unit-tested against a stub window with no display.
"""

from __future__ import annotations

import threading
import time

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from picasso_workflow import manual_to_workflow as mtw
from picasso_workflow import modulespec

# Review-table colours per confidence level.
_CONF_COLOR = {
    mtw.CONF_OK: "#1a7f37",  # green
    mtw.CONF_RETRO: "#9a6700",  # amber
    mtw.CONF_LOW: "#9a6700",  # amber
    mtw.CONF_UNMAPPED: "#cf222e",  # red
}
_CONF_LABEL = {
    mtw.CONF_OK: "OK",
    mtw.CONF_RETRO: "retro-calc",
    mtw.CONF_LOW: "defaults",
    mtw.CONF_UNMAPPED: "unmapped",
}

# Wizard-added plumbing modules (not user processing steps) -- excluded when
# re-splitting single/aggregation stages (imported from engine for consistency).
_BRIDGE_MODULES = mtw.BRIDGE_MODULES

# Loader modules must stay in the single stage (they produce the per-channel
# localizations the aggregation stage loads back), so they cannot be moved.
_LOADER_MODULES = frozenset(
    {"load_dataset_movie", "load_dataset_localizations"}
)

# Aggregation workflow type index in the main GUI's workflow_type combo.
_AGGREGATION_TYPE_INDEX = 1
# "Workflow Config" tab index.
_WORKFLOW_CONFIG_TAB = 0


def _resolve_filepath(channel, tree_data):
    """The per-channel input file behind ``("$$map", "filepath")``."""
    for _ds, chans in tree_data.get("file_paths", {}).items():
        if channel in chans:
            return chans[channel]
    return None


def _format_value(value, channel, tree_data):
    """Render one parameter value, resolving ``$$map`` for ``channel``.

    ``("$$map", "filepath")`` / ``("$$map", "#tags")`` are the per-channel file
    and tag columns; other ``$$map`` columns come from ``tile_params``. Other
    ``$``-commands (e.g. ``$$get_prior_result``) render as a short placeholder.
    """
    tile_params = tree_data.get("tile_params", {})
    if isinstance(value, tuple) and value and value[0] == "$$map":
        col = value[1] if len(value) > 1 else ""
        if col == "filepath":
            return _resolve_filepath(channel, tree_data) or "<filepath>"
        if col == "#tags":
            return channel
        entry = tile_params.get(col)
        if isinstance(entry, dict):
            vals = entry.get("values", {})
            if channel in vals:
                return str(vals[channel])
            if entry.get("default") is not None:
                return str(entry["default"])
        if len(value) > 2 and value[2] is not None:
            return str(value[2])
        return f"<{col}>"
    if isinstance(value, tuple) and value and str(value[0]).startswith("$"):
        return f"<{value[0]}>"
    if isinstance(value, dict):
        return "{…}"
    return str(value)


def _format_params(params, channel, tree_data):
    """Compact ``key=value`` rendering of a module's params for ``channel``."""
    return ", ".join(
        f"{k}={_format_value(v, channel, tree_data)}"
        for k, v in params.items()
    )


def resplit_to_aggregation(
    single, single_meta, agg, agg_meta, single_row_index
):
    """Re-split single/aggregation at a chosen single-stage step.

    The step at ``single_row_index`` (indexing ``single``, bridges included) and
    every later single-stage *processing* step move to the front of the
    aggregation stage; the single stage is closed with ``save_single_dataset``
    and the aggregation stage reopened with ``analysis_documentation`` +
    ``load_datasets_to_aggregate``. Returns new
    ``(single, single_meta, agg, agg_meta)``. Qt-free so it is unit-testable.
    """

    def strip(mods, metas):
        return [
            (m, me)
            for m, me in zip(mods, metas)
            if m[0] not in _BRIDGE_MODULES
        ]

    single_pairs = strip(single, single_meta)
    agg_pairs = strip(agg, agg_meta)
    # Real-step index of the selected row (bridges excluded).
    split = max(
        0,
        sum(
            1
            for i in range(single_row_index + 1)
            if single[i][0] not in _BRIDGE_MODULES
        )
        - 1,
    )
    keep = single_pairs[:split]
    moved = single_pairs[split:]

    new_single = [m for m, _ in keep] + [mtw._save_single_bridge()]
    new_single_meta = [me for _, me in keep] + [mtw._bridge_meta()]
    save_index = len(new_single) - 1
    new_agg_real = moved + agg_pairs
    new_agg = (
        [
            ("analysis_documentation", {}),
            mtw._load_aggregate_bridge(save_index),
        ]
        + [m for m, _ in new_agg_real]
        + [("save_datasets_aggregated", {})]
    )
    new_agg_meta = (
        [mtw._bridge_meta(), mtw._bridge_meta()]
        + [me for _, me in new_agg_real]
        + [mtw._bridge_meta()]
    )
    return new_single, new_single_meta, new_agg, new_agg_meta


def apply_result_to_window(window, result):
    """Write a :class:`~.manual_to_workflow.ReconstructionResult` into the GUI.

    Only touches documented ``Window`` members: switches to the aggregation
    workflow type, installs ``tree_data`` (via ``_restore_dataset_table``), sets
    both module lists, and refreshes the list widgets, module palette and the
    Workflow Config tab. Qt-free so it can be unit-tested with a stub window.

    Parameters
    ----------
    window : picasso_workflow.gui.Window
        The main GUI window to populate (or a duck-typed stub in tests).
    result : manual_to_workflow.ReconstructionResult
        The reconstructed workflow to install.

    Returns
    -------
    None
    """
    window._restore_dataset_table(
        {
            "workflow_type": _AGGREGATION_TYPE_INDEX,
            "tree_data": result.tree_data,
        }
    )
    window.single_workflow_modules[:] = list(result.single_modules)
    window.aggregation_workflow_modules[:] = list(result.agg_modules)
    window._refresh_workflow_list(
        window.single_workflow_list, window.single_workflow_modules
    )
    window._refresh_workflow_list(
        window.aggregation_workflow_list,
        window.aggregation_workflow_modules,
    )
    window._refresh_module_palette()
    window.tabs.setCurrentIndex(_WORKFLOW_CONFIG_TAB)


class _ReconstructWorker(QtCore.QObject):
    """Runs the reconstruction engine off the UI thread."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    # (message, fraction): fraction is a float in [0, 1] or None (indeterminate)
    progress = pyqtSignal(str, object)

    def __init__(self, manual_dir, raw_root, pixelsize, start_from):
        super().__init__()
        self._manual_dir = manual_dir
        self._raw_root = raw_root
        self._pixelsize = pixelsize
        self._start_from = start_from
        self._cancel = threading.Event()

    def cancel(self):
        """Request cooperative cancellation (thread-safe)."""
        self._cancel.set()

    def run(self):
        try:
            result = mtw.reconstruct_workflow(
                self._manual_dir,
                raw_root=self._raw_root,
                pixelsize=self._pixelsize,
                start_from=self._start_from,
                progress=self.progress.emit,
                should_cancel=self._cancel.is_set,
            )
        except mtw.ReconstructionCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class _DragDropLineEdit(QtWidgets.QLineEdit):
    """QLineEdit that accepts dropped folder paths (for drag-drop convenience)."""

    def dragEnterEvent(self, event):
        """Accept drops of files/folders."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Handle dropped paths: use the first folder."""
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            # If it's a file, use its parent directory
            import os

            if os.path.isfile(path):
                path = os.path.dirname(path)
            self.setText(path)
            event.accept()


class _SelectFolderPage(QtWidgets.QWizardPage):
    """Page 1: manual folder, start mode, raw-data folder, and pixel size."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Select manual analysis")
        self.setSubTitle(
            "Choose the folder holding the manual analysis (the one with the "
            "saved .hdf5 step files, e.g. 'analysis_manual')."
        )
        layout = QtWidgets.QFormLayout(self)

        self.manual_edit = _DragDropLineEdit()
        manual_browse = QtWidgets.QPushButton("Browse…")
        manual_browse.clicked.connect(self._browse_manual)
        layout.addRow(
            "Manual-analysis folder:",
            self._with_button(self.manual_edit, manual_browse),
        )

        # Manual workflows start from a localized file (picasso Localize is a
        # separate GUI), so default to loading localizations. Optionally
        # prepend the raw-load + localize steps (parameters recovered from the
        # hdf5 info) to re-localize from the raw movies.
        self.relocalize_check = QtWidgets.QCheckBox(
            "Also re-localize from the raw movies "
            "(recovered localization parameters)"
        )
        self.relocalize_check.setChecked(False)
        layout.addRow("", self.relocalize_check)

        self.raw_edit = QtWidgets.QLineEdit()
        self.raw_edit.setPlaceholderText("auto-detected — leave blank to auto")
        self.raw_edit.setToolTip(
            "Dataset folder holding the raw acquisition subfolders (01_…, "
            "02_…), used to re-localize from the raw movies. Leave blank to "
            "auto-detect by searching upward from the manual folder."
        )
        self.raw_browse = QtWidgets.QPushButton("Browse…")
        self.raw_browse.clicked.connect(self._browse_raw)
        layout.addRow(
            "Raw data folder:",
            self._with_button(self.raw_edit, self.raw_browse),
        )
        # The raw folder only matters when re-localizing from raw.
        self.relocalize_check.toggled.connect(self._on_relocalize_toggled)
        self._on_relocalize_toggled(False)

        self.pixelsize_spin = QtWidgets.QDoubleSpinBox()
        self.pixelsize_spin.setRange(1.0, 10000.0)
        self.pixelsize_spin.setValue(130.0)
        self.pixelsize_spin.setSuffix(" nm")
        layout.addRow("Camera pixel size:", self.pixelsize_spin)

        # The manual folder is mandatory; the raw folder is optional (auto).
        self.registerField("manual_dir*", self.manual_edit)

    def start_from(self):
        """Return the engine ``start_from`` mode for the current selection."""
        return "raw" if self.relocalize_check.isChecked() else "localizations"

    def _on_relocalize_toggled(self, relocalize):
        self.raw_edit.setEnabled(relocalize)
        self.raw_browse.setEnabled(relocalize)

    @staticmethod
    def _with_button(line_edit, button):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(line_edit)
        row.addWidget(button)
        holder = QtWidgets.QWidget()
        holder.setLayout(row)
        return holder

    def _browse_manual(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select manual-analysis folder", self.manual_edit.text()
        )
        if not folder:
            return
        self.manual_edit.setText(folder)
        # Best-effort: auto-fill the raw-data folder (cheap: reads one header
        # and walks up a few levels — no recursive tree walk).
        if not self.raw_edit.text().strip():
            try:
                detected = mtw.detect_raw_root(folder)
            except Exception:  # noqa: BLE001 - best-effort only
                detected = None
            if detected:
                self.raw_edit.setText(detected)

    def _browse_raw(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select raw-data folder", self.raw_edit.text()
        )
        if folder:
            self.raw_edit.setText(folder)


class _ReviewPage(QtWidgets.QWizardPage):
    """Page 2: run the engine and review the reconstructed workflow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Review reconstructed workflow")
        self.setSubTitle(
            "Each step is mapped to a module with a confidence flag. Change a "
            "module via its dropdown, right-click to remove a step, and pick a "
            "channel to see its parameters. Unmapped steps are kept as no-op "
            "placeholders you can delete or reassign here or in the GUI."
        )
        self.result = None
        self._thread = None
        self._worker = None
        self._start_time = None
        # Editable buffers (copies of the reconstruction, mutated by the table).
        self._single = []
        self._agg = []
        self._single_meta = []
        self._agg_meta = []
        self._tree_data = {}
        self._channels = []
        self._apply_ready = False
        self._building = False

        layout = QtWidgets.QVBoxLayout(self)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        progress_row = QtWidgets.QHBoxLayout()
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_row.addWidget(self.progress_bar)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setToolTip("Abort the running scan")
        self.stop_button.clicked.connect(self._request_cancel)
        progress_row.addWidget(self.stop_button)
        layout.addLayout(progress_row)

        channel_row = QtWidgets.QHBoxLayout()
        channel_row.addWidget(QtWidgets.QLabel("Show parameters for channel:"))
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.currentIndexChanged.connect(
            lambda _i: self._rebuild_table()
        )
        channel_row.addWidget(self.channel_combo)
        channel_row.addStretch(1)
        self.to_agg_button = QtWidgets.QPushButton(
            "Move selected step → aggregation (and all after)"
        )
        self.to_agg_button.setToolTip(
            "Make the selected single-stage step the cross-channel boundary: "
            "it and every later step move to the aggregation stage, with the "
            "save/load/documentation bridge modules inserted."
        )
        self.to_agg_button.clicked.connect(self._move_selected_to_aggregation)
        channel_row.addWidget(self.to_agg_button)
        self.remove_button = QtWidgets.QPushButton("Remove selected step")
        self.remove_button.setToolTip(
            "Delete the selected row (a step or bridge module) from the "
            "workflow. Note: loader and bridge modules usually cannot be removed."
        )
        self.remove_button.clicked.connect(self._remove_selected_row)
        channel_row.addWidget(self.remove_button)
        layout.addLayout(channel_row)

        self._columns = [
            "Stage",
            "Source",
            "File",
            "Confidence",
            "Module",
            "Parameters",
            "Notes",
        ]
        self.table = QtWidgets.QTableWidget(0, len(self._columns))
        self.table.setHorizontalHeaderLabels(self._columns)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        # Wrap long cell text instead of eliding it: word-wrap on, elide off,
        # columns at bounded interactive widths (so text must wrap rather than
        # stretch the column to fit), and rows auto-grow to fit the wrapped
        # lines. resizeColumnsToContents() is deliberately NOT used -- it sizes
        # every column to its widest cell, which prevents wrapping entirely.
        self.table.setWordWrap(True)
        self.table.setTextElideMode(Qt.TextElideMode.ElideNone)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Interactive
        )
        # Stage, Source, File, Confidence, Module, Parameters, Notes
        for col, width in enumerate((70, 160, 150, 90, 150, 240, 240)):
            self.table.setColumnWidth(col, width)
        self.table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._row_context_menu)
        layout.addWidget(self.table)

        self.errors_box = QtWidgets.QPlainTextEdit()
        self.errors_box.setReadOnly(True)
        self.errors_box.setMaximumHeight(120)
        layout.addWidget(QtWidgets.QLabel("Consistency findings:"))
        layout.addWidget(self.errors_box)

    def initializePage(self):
        # Re-entering the page (Back then Next) must not leave a prior worker
        # thread running: stop it first so a stale _on_finished cannot race the
        # new run and overwrite the table/result.
        self._stop_worker()
        self.result = None
        self.table.setRowCount(0)
        self.errors_box.setPlainText("")
        self.status.setText("Scanning and reconstructing…")
        # Start indeterminate (busy) until the first determinate update.
        self.progress_bar.setRange(0, 0)
        self.stop_button.setEnabled(True)
        self._start_time = time.monotonic()
        self.completeChanged.emit()

        wizard = self.wizard()
        select = wizard.page(wizard._select_page_id)
        manual_dir = select.manual_edit.text()
        raw_root = select.raw_edit.text().strip() or None
        pixelsize = select.pixelsize_spin.value()
        start_from = select.start_from()

        self._thread = QThread(self)
        self._worker = _ReconstructWorker(
            manual_dir, raw_root, pixelsize, start_from
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)
        self._thread.start()

    def _request_cancel(self):
        """Ask the running worker to stop (cooperative, between channels)."""
        worker = getattr(self, "_worker", None)
        if worker is not None:
            worker.cancel()
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping…")

    def _on_cancelled(self):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.stop_button.setEnabled(False)
        self.status.setText(
            "Scan stopped. Go Back to change settings, or close the wizard."
        )
        self.completeChanged.emit()

    def _on_progress(self, message, fraction):
        """Update the status label + progress bar from the worker thread."""
        if fraction is None:
            self.progress_bar.setRange(0, 0)  # busy / indeterminate
            self.status.setText(message)
            return
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
        eta = self._estimate_eta(fraction)
        self.status.setText(message + eta)

    def _estimate_eta(self, fraction):
        """Return a '  (~Ns remaining)' suffix, or '' when not estimable."""
        if self._start_time is None or not 0.0 < fraction < 1.0:
            return ""
        elapsed = time.monotonic() - self._start_time
        remaining = elapsed * (1.0 - fraction) / fraction
        if remaining < 1.0:
            return ""
        if remaining < 90.0:
            return f"  (~{int(round(remaining))}s remaining)"
        return f"  (~{int(round(remaining / 60.0))} min remaining)"

    def _stop_worker(self):
        """Cancel, quit and dispose any running reconstruction thread/worker.

        Request cancellation first so ``wait()`` returns promptly (``run()`` is
        not interruptible but polls between channels), and disconnect the old
        worker's signals so a late ``finished`` cannot be delivered to
        ``_on_finished`` after a new run has started.
        """
        worker = getattr(self, "_worker", None)
        if worker is not None:
            try:
                worker.cancel()
                worker.finished.disconnect()
                worker.failed.disconnect()
                worker.cancelled.disconnect()
            except (TypeError, RuntimeError):
                pass  # nothing connected / already deleted
        thread = getattr(self, "_thread", None)
        if thread is not None:
            try:
                thread.quit()
                thread.wait()
            except RuntimeError:
                pass  # already deleted
        self._thread = None
        self._worker = None

    def _on_failed(self, message):
        # Stop the busy animation (determinate, empty) on failure.
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.stop_button.setEnabled(False)
        self.status.setText(f"Reconstruction failed: {message}")
        self.completeChanged.emit()

    def _on_finished(self, result):
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(1000)
        self.stop_button.setEnabled(False)
        self.result = result
        # Editable copies the table mutates; Apply uses these.
        self._single = list(result.single_modules)
        self._agg = list(result.agg_modules)
        self._single_meta = [dict(m) for m in result.single_meta]
        self._agg_meta = [dict(m) for m in result.agg_meta]
        self._tree_data = result.tree_data
        self._channels = list(result.channels)

        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        self.channel_combo.addItems(self._channels)
        self.channel_combo.blockSignals(False)

        summary = (
            f"{len(result.channels)} channels, "
            f"{len(self._single)} single + {len(self._agg)} aggregation "
            "modules."
        )
        if result.raw_root:
            summary += f" Raw data: {result.raw_root}."
        self.status.setText(summary)
        # Keep engine findings that _revalidate does not re-derive (the
        # single[/aggregation[ validation lines are recomputed on each edit).
        self._base_findings = [
            e
            for e in result.errors
            if not e.startswith("single[") and not e.startswith("aggregation[")
        ]
        self._rebuild_table()
        self._revalidate()

    # -- table -----------------------------------------------------------
    def _rows(self):
        """Ordered (stage, list_name, index) tuples for the current buffers."""
        rows = [("single", "single", i) for i in range(len(self._single))]
        rows += [("aggregation", "agg", i) for i in range(len(self._agg))]
        return rows

    def _buffers(self, list_name):
        if list_name == "single":
            return self._single, self._single_meta
        return self._agg, self._agg_meta

    def _rebuild_table(self):
        self._building = True
        try:
            channel = self.channel_combo.currentText()
            rows = self._rows()
            self.table.setRowCount(len(rows))
            for row, (stage, list_name, idx) in enumerate(rows):
                modules, meta = self._buffers(list_name)
                name, params = modules[idx]
                m = meta[idx]
                self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(stage))
                self.table.setItem(
                    row, 1, QtWidgets.QTableWidgetItem(m.get("source", ""))
                )
                self.table.setItem(
                    row,
                    2,
                    QtWidgets.QTableWidgetItem(m.get("source_file") or ""),
                )
                conf = m.get("confidence", "")
                conf_item = QtWidgets.QTableWidgetItem(
                    _CONF_LABEL.get(conf, conf)
                )
                color = _CONF_COLOR.get(conf)
                if color:
                    conf_item.setForeground(QtGui.QColor(color))
                self.table.setItem(row, 3, conf_item)
                self.table.setCellWidget(
                    row, 4, self._module_combo(stage, list_name, idx, name)
                )
                self.table.setItem(
                    row,
                    5,
                    QtWidgets.QTableWidgetItem(
                        _format_params(params, channel, self._tree_data)
                    ),
                )
                self.table.setItem(
                    row, 6, QtWidgets.QTableWidgetItem(m.get("notes", ""))
                )
            # Grow row heights to fit the wrapped cell text (columns keep their
            # bounded widths set in the constructor, so text wraps).
            self.table.resizeRowsToContents()
        finally:
            self._building = False

    def _module_combo(self, stage, list_name, idx, current):
        scope = (
            modulespec.Scope.SINGLE
            if stage == "single"
            else modulespec.Scope.AGGREGATION
        )
        names = sorted(
            n
            for n, spec in modulespec.MODULE_REGISTRY.items()
            if scope in spec.scopes and not spec.experimental
        )
        if current not in names:
            names = sorted(set(names) | {current})
        combo = QtWidgets.QComboBox()
        combo.addItems(names)
        combo.setCurrentText(current)
        combo.currentTextChanged.connect(
            lambda text, ln=list_name, i=idx: self._on_module_changed(
                ln, i, text
            )
        )
        return combo

    def _on_module_changed(self, list_name, idx, new_name):
        if self._building:
            return
        modules, meta = self._buffers(list_name)
        _name, params = modules[idx]
        modules[idx] = (new_name, params)
        meta[idx]["confidence"] = mtw.CONF_OK
        meta[idx]["notes"] = "edited"
        self._revalidate()

    def _delete_row(self, list_name, idx):
        """Delete a module row, refusing loaders and bridge modules.

        Loaders are required to produce the per-channel localizations and
        bridge modules are auto-generated, so both are protected here. Shared
        by the right-click menu and the "Remove selected step" button so the
        guards cannot diverge. Returns ``True`` if the row was removed.
        """
        name = (
            self._single[idx][0]
            if list_name == "single"
            else self._agg[idx][0]
        )
        if name in _LOADER_MODULES:
            QtWidgets.QMessageBox.information(
                self,
                "Remove step",
                f"Cannot remove '{name}': loader modules are required to "
                "produce the per-channel localizations.",
            )
            return False
        if name in _BRIDGE_MODULES:
            QtWidgets.QMessageBox.information(
                self,
                "Remove step",
                f"'{name}' is an auto-generated bridge module. It will be "
                "rebuilt when the single/aggregation boundary changes.",
            )
            return False
        modules, meta = self._buffers(list_name)
        del modules[idx]
        del meta[idx]
        self._rebuild_table()
        self._revalidate()
        return True

    def _row_context_menu(self, pos):
        row = self.table.rowAt(pos.y())
        rows = self._rows()
        if row < 0 or row >= len(rows):
            return
        _stage, list_name, idx = rows[row]
        menu = QtWidgets.QMenu(self)
        remove = menu.addAction("Remove this step")
        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen is remove:
            self._delete_row(list_name, idx)

    def _remove_selected_row(self):
        """Remove the selected step/row from the workflow.

        Can remove any module except loaders (which are required) and bridge
        modules (which are auto-generated). Shows a warning if removal is
        blocked.
        """
        row = self.table.currentRow()
        rows = self._rows()
        if row < 0 or row >= len(rows):
            QtWidgets.QMessageBox.information(
                self, "Remove step", "Select a row first."
            )
            return
        _stage, list_name, idx = rows[row]
        self._delete_row(list_name, idx)

    def _move_selected_to_aggregation(self):
        """Split single/aggregation at the selected single-stage step.

        The selected step (e.g. a channel-alignment step reclassified from a
        placeholder) and every later single-stage step move to the aggregation
        stage; the single stage is closed with ``save_single_dataset`` and the
        aggregation stage opened with ``analysis_documentation`` +
        ``load_datasets_to_aggregate``. Existing bridge modules are rebuilt.
        """
        row = self.table.currentRow()
        rows = self._rows()
        if row < 0 or row >= len(rows):
            QtWidgets.QMessageBox.information(
                self, "Move to aggregation", "Select a single-stage row first."
            )
            return
        _stage, list_name, idx = rows[row]
        name = self._single[idx][0] if list_name == "single" else None
        if (
            list_name != "single"
            or name in _BRIDGE_MODULES
            or name in _LOADER_MODULES
        ):
            QtWidgets.QMessageBox.information(
                self,
                "Move to aggregation",
                "Pick a single-stage processing step after the loader (not "
                "the loader or a bridge module).",
            )
            return

        (
            self._single,
            self._single_meta,
            self._agg,
            self._agg_meta,
        ) = resplit_to_aggregation(
            self._single,
            self._single_meta,
            self._agg,
            self._agg_meta,
            idx,
        )
        self._rebuild_table()
        self._revalidate()

    # -- validation / apply ---------------------------------------------
    def _revalidate(self):
        findings = list(getattr(self, "_base_findings", []))
        findings += [
            f"single[{e}"
            for e in modulespec.validate_workflow(
                self._single, modulespec.Scope.SINGLE
            )
        ]
        if self._agg:
            findings += [
                f"aggregation[{e}"
                for e in modulespec.validate_workflow(
                    self._agg, modulespec.Scope.AGGREGATION
                )
            ]
        # Apply is never blocked: placeholder modules (``dummy_module``) are
        # runnable no-ops and can be deleted or reassigned from the main GUI
        # after applying, and every other finding (scope warnings, params not
        # recovered, etc.) is likewise fine-tuned there. They are surfaced as
        # notes only.
        placeholders = sorted(
            {
                n
                for n, _ in (self._single + self._agg)
                if n in ("manual", "dummy_module")
            }
        )
        self._apply_ready = True
        if placeholders:
            findings.insert(
                0,
                f"NOTE: {len(placeholders)} placeholder step(s) kept as no-ops "
                "-- delete or reassign them in the GUI (or via the 'module' "
                "dropdown / right-click > remove): " + ", ".join(placeholders),
            )
        self.errors_box.setPlainText(
            "\n".join(findings) or "No consistency issues found."
        )
        self.completeChanged.emit()

    def current_result(self):
        """Return the reconstruction with the table's edits applied."""
        self.result.single_modules = self._single
        self.result.agg_modules = self._agg
        return self.result

    def isComplete(self):
        return self.result is not None and self._apply_ready


class ManualToWorkflowWizard(QtWidgets.QWizard):
    """Wizard that reconstructs a workflow from a manual analysis folder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual → Workflow")
        self.setWizardStyle(QtWidgets.QWizard.WizardStyle.ModernStyle)
        self.resize(760, 560)

        select = _SelectFolderPage(self)
        review = _ReviewPage(self)
        self._select_page_id = self.addPage(select)
        self._review_page_id = self.addPage(review)
        # "Apply to GUI" applies but keeps the wizard open (so the review table
        # and its notes stay for reference); "Close" dismisses it.
        self.setOption(QtWidgets.QWizard.WizardOption.NoCancelButton, False)
        self.setButtonText(
            QtWidgets.QWizard.WizardButton.FinishButton, "Apply to GUI"
        )
        self.setButtonText(
            QtWidgets.QWizard.WizardButton.CancelButton, "Close"
        )

    def accept(self):
        review = self.page(self._review_page_id)
        parent = self.parent()
        if review.result is not None and parent is not None:
            try:
                apply_result_to_window(parent, review.current_result())
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                QtWidgets.QMessageBox.critical(
                    self,
                    "Manual → Workflow",
                    f"Could not apply the reconstructed workflow: {exc}",
                )
                return
        # Keep the wizard open on Apply so the notes stay visible; the user
        # closes it with "Close". (Do not call super().accept().)
        review.status.setText(
            "Applied to the GUI ✓ — review/edit there; this window stays open "
            "for reference. Use 'Close' when done."
        )

    def reject(self):
        # Cancelling the wizard must abort (and join) any running scan so no
        # worker thread lingers after the window closes.
        self.page(self._review_page_id)._stop_worker()
        super().reject()
