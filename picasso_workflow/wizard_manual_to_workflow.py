#!/usr/bin/env python
"""GUI wizard: translate a manual analysis into a picasso-workflow.

A thin PyQt6 front-end over :mod:`picasso_workflow.manual_to_workflow`. The user
selects the dataset root (holding the raw acquisition folders and a
manual-analysis subfolder); the engine reconstructs the equivalent aggregation
workflow; the wizard shows a per-step review table with confidence flags and
consistency findings; on *Apply* it writes the reconstructed Files tree and
module lists into the main GUI so the user can review and launch via the normal
Run tab.

The GUI-population step is factored into :func:`apply_result_to_window`, a
Qt-free helper that only touches documented ``Window`` attributes/methods, so it
can be unit-tested against a stub window with no display.
"""

from __future__ import annotations

import time

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QThread, pyqtSignal

from picasso_workflow import manual_to_workflow as mtw

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

# Aggregation workflow type index in the main GUI's workflow_type combo.
_AGGREGATION_TYPE_INDEX = 1
# "Workflow Config" tab index.
_WORKFLOW_CONFIG_TAB = 0


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
    # (message, fraction): fraction is a float in [0, 1] or None (indeterminate)
    progress = pyqtSignal(str, object)

    def __init__(self, manual_dir, raw_root, pixelsize):
        super().__init__()
        self._manual_dir = manual_dir
        self._raw_root = raw_root
        self._pixelsize = pixelsize

    def run(self):
        try:
            result = mtw.reconstruct_workflow(
                self._manual_dir,
                raw_root=self._raw_root,
                pixelsize=self._pixelsize,
                progress=self.progress.emit,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class _SelectFolderPage(QtWidgets.QWizardPage):
    """Page 1: pick the dataset root, manual subfolder, and pixel size."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Select manual analysis")
        self.setSubTitle(
            "Choose the folder holding the manual analysis (the one with the "
            "saved .hdf5 step files, e.g. 'analysis_manual')."
        )
        layout = QtWidgets.QFormLayout(self)

        self.manual_edit = QtWidgets.QLineEdit()
        manual_browse = QtWidgets.QPushButton("Browse…")
        manual_browse.clicked.connect(self._browse_manual)
        layout.addRow(
            "Manual-analysis folder:",
            self._with_button(self.manual_edit, manual_browse),
        )

        self.raw_edit = QtWidgets.QLineEdit()
        self.raw_edit.setPlaceholderText("auto-detected — leave blank to auto")
        self.raw_edit.setToolTip(
            "Dataset folder holding the raw acquisition subfolders (01_…, "
            "02_…), used to re-localize from the raw movies. Leave blank to "
            "auto-detect by searching upward from the manual folder."
        )
        raw_browse = QtWidgets.QPushButton("Browse…")
        raw_browse.clicked.connect(self._browse_raw)
        layout.addRow(
            "Raw data folder:",
            self._with_button(self.raw_edit, raw_browse),
        )

        self.pixelsize_spin = QtWidgets.QDoubleSpinBox()
        self.pixelsize_spin.setRange(1.0, 10000.0)
        self.pixelsize_spin.setValue(130.0)
        self.pixelsize_spin.setSuffix(" nm")
        layout.addRow("Camera pixel size:", self.pixelsize_spin)

        # The manual folder is mandatory; the raw folder is optional (auto).
        self.registerField("manual_dir*", self.manual_edit)

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
            "Steps are mapped to modules with a confidence flag. Resolve any "
            "'unmapped' placeholders before applying."
        )
        self.result = None
        self._thread = None
        self._worker = None
        self._start_time = None

        layout = QtWidgets.QVBoxLayout(self)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Stage", "Module", "Confidence", "Source / notes"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
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
        self._start_time = time.monotonic()
        self.completeChanged.emit()

        wizard = self.wizard()
        select = wizard.page(wizard._select_page_id)
        manual_dir = select.manual_edit.text()
        raw_root = select.raw_edit.text().strip() or None
        pixelsize = select.pixelsize_spin.value()

        self._thread = QThread(self)
        self._worker = _ReconstructWorker(manual_dir, raw_root, pixelsize)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

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
        """Quit and dispose any running reconstruction thread/worker.

        Disconnect the old worker's signals first: ``run()`` is not
        interruptible, so it may still emit ``finished`` while ``wait()`` blocks
        the UI thread; without disconnecting, that stale result would be
        delivered to ``_on_finished`` after the new run has started.
        """
        worker = getattr(self, "_worker", None)
        if worker is not None:
            try:
                worker.finished.disconnect()
                worker.failed.disconnect()
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
        self.status.setText(f"Reconstruction failed: {message}")
        self.completeChanged.emit()

    def _on_finished(self, result):
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(1000)
        self.result = result
        self._populate_table(result)
        n_unmapped = sum(
            1 for r in result.review if r["confidence"] == mtw.CONF_UNMAPPED
        )
        summary = (
            f"{len(result.channels)} channels, "
            f"{len(result.single_modules)} single + "
            f"{len(result.agg_modules)} aggregation modules."
        )
        if result.raw_root:
            summary += f" Raw data: {result.raw_root}."
        else:
            summary += " Raw data folder not found (loading manual locs)."
        if n_unmapped:
            summary += (
                f" {n_unmapped} unmapped step(s) must be resolved before "
                "applying."
            )
        self.status.setText(summary)
        self.errors_box.setPlainText(
            "\n".join(result.errors) or "No consistency issues found."
        )
        self.completeChanged.emit()

    def _populate_table(self, result):
        self.table.setRowCount(len(result.review))
        for row, item in enumerate(result.review):
            notes = item["source"]
            if item["messages"]:
                notes += "  —  " + "; ".join(item["messages"])
            cells = [
                item["stage"],
                item["module"],
                _CONF_LABEL.get(item["confidence"], item["confidence"]),
                notes,
            ]
            for col, text in enumerate(cells):
                cell = QtWidgets.QTableWidgetItem(text)
                if col == 2:
                    color = _CONF_COLOR.get(item["confidence"])
                    if color:
                        cell.setForeground(QtGui.QColor(color))
                self.table.setItem(row, col, cell)
        self.table.resizeColumnsToContents()

    def isComplete(self):
        if self.result is None:
            return False
        return not any(
            r["confidence"] == mtw.CONF_UNMAPPED for r in self.result.review
        )


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
        self.setButtonText(
            QtWidgets.QWizard.WizardButton.FinishButton, "Apply to GUI"
        )

    def accept(self):
        review = self.page(self._review_page_id)
        parent = self.parent()
        if review.result is not None and parent is not None:
            try:
                apply_result_to_window(parent, review.result)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                QtWidgets.QMessageBox.critical(
                    self,
                    "Manual → Workflow",
                    f"Could not apply the reconstructed workflow: {exc}",
                )
                return
        super().accept()
