"""``picasso-workflow-admin-gui`` -- PyQt6 admin console over :mod:`.core`.

A small desktop app that SSHes to the cluster login node (via the existing
:class:`picasso_workflow.slurm_submit.SlurmCommunicator`) to inspect the
shared-mount Apptainer images, the SLURM queue, and disk usage -- so the
images on the mount, the versions, and recent jobs are visible at a glance
without hand-rolled SSH. All SSH work runs off the UI thread in a
:class:`QThread` worker (mirrors monet's / picasso-registry's admin console).

Launched standalone (``picasso-workflow-admin-gui``) or opened from the main
GUI's Tools menu (:class:`AdminWindow` takes a parent). Needs the ``[gui]``
extra (PyQt6).
"""

from __future__ import annotations

import datetime
import getpass
import sys
import traceback

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from picasso_workflow.admin import core


def _row_widget(layout) -> QWidget:
    """Wrap a layout in a QWidget (so it can go in a QFormLayout row)."""
    w = QWidget()
    w.setLayout(layout)
    return w


def _size_to_bytes(s: str) -> float:
    """Parse a human size like ``"4.2G"`` back to bytes, for numeric sorting.

    Returns ``-1`` for blank/unparseable input (sorts such rows first).
    """
    units = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    s = (s or "").strip()
    if not s:
        return -1.0
    if s[-1] in units:
        try:
            return float(s[:-1]) * units[s[-1]]
        except ValueError:
            return -1.0
    try:
        return float(s)
    except ValueError:
        return -1.0


class _NumSortItem(QTableWidgetItem):
    """Table item that displays arbitrary text but sorts by a numeric key.

    Used for the Images 'Size' column so ``900M`` sorts below ``4.2G`` instead
    of lexicographically.
    """

    def __init__(self, text: str, sort_key: float):
        super().__init__(text)
        self._sort_key = sort_key

    def __lt__(self, other):
        try:
            return self._sort_key < other._sort_key
        except AttributeError:
            return super().__lt__(other)


class _Worker(QThread):
    """Runs one ``fn() -> result`` off the UI thread."""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:  # noqa: D401 - QThread entry
        try:
            result = self._fn()
        except Exception as exc:  # surface any failure as a signal
            self.failed.emit(str(exc))
            return
        self.done.emit(result)


class AdminWindow(QMainWindow):
    """Standalone cluster admin console (also embeddable via a parent)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("picasso-workflow admin")
        self._worker: _Worker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.addWidget(
            QLabel(
                f"This GUI's picasso-workflow version: {core.this_gui_version()}"
            )
        )
        root.addWidget(self._build_connection_panel())
        self._test_run_dir = None
        self._test_run_jobs: list[str] = []
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_images_tab(), "Images")
        self.tabs.addTab(self._build_testruns_tab(), "Test runs")
        self.tabs.addTab(self._build_jobs_tab(), "Jobs & usage")
        self.activity = QPlainTextEdit(readOnly=True)
        self.activity.setMaximumBlockCount(5000)
        activity_box = QWidget()
        activity_lay = QVBoxLayout(activity_box)
        activity_lay.setContentsMargins(0, 0, 0, 0)
        activity_lay.addWidget(QLabel("Activity"))
        activity_lay.addWidget(self.activity)

        # Draggable vertical split: a plain layout let a window resize mostly
        # enlarge the Activity log. The splitter lets the user pull the tabs area
        # taller, and defaults to favouring the tabs 3:1.
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.tabs)
        split.addWidget(activity_box)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)

        self.resize(1000, 700)

    # -- connection panel -------------------------------------------------
    def _build_connection_panel(self) -> QWidget:
        box = QWidget()
        form = QFormLayout(box)
        self.host = QComboBox()
        self.host.setEditable(True)
        self.host.addItems([str(h) for h in core.cluster_hosts()])
        self.user = QLineEdit(getpass.getuser())
        self.identity = QLineEdit(core.discover_ssh_key() or "")
        self.identity.setMinimumWidth(420)  # SSH key paths are long
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_identity)
        key_row = QHBoxLayout()
        key_row.addWidget(self.identity, 1)
        key_row.addWidget(browse)
        key_w = QWidget()
        key_w.setLayout(key_row)
        self.conn_status = QLabel("not connected")

        form.addRow("Cluster", self.host)
        form.addRow("User", self.user)
        form.addRow("Identity file", key_w)
        # Connection-level actions shared by both tabs: Refresh images lists the
        # mount once and fills BOTH the Images table and the Test-runs dropdown.
        test = QPushButton("Test connection")
        test.clicked.connect(self._on_test)
        refresh = QPushButton("Refresh images")
        refresh.clicked.connect(self._refresh_images)
        actions = QHBoxLayout()
        actions.addWidget(test)
        actions.addWidget(refresh)
        actions.addWidget(self.conn_status, 1)
        form.addRow("", _row_widget(actions))
        return box

    def _pick_identity(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "SSH private key")
        if path:
            self.identity.setText(path)

    @staticmethod
    def _make_table(headers) -> QTableWidget:
        """A read-only table with the given headers and a stretched last col."""
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        return table

    @staticmethod
    def _make_selectable(table) -> None:
        """Allow multi-row selection on a jobs table (for 'Cancel selected')."""
        table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )

    # -- images tab -------------------------------------------------------
    def _build_images_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.image_dir_label = QLabel(
            "Click 'Refresh images' (top) to list the shared mount."
        )
        lay.addWidget(self.image_dir_label)

        # Kind filter (client-side): show lean and/or test images.
        filt = QHBoxLayout()
        filt.addWidget(QLabel("Show:"))
        self.img_show_lean = QCheckBox("Lean")
        self.img_show_lean.setChecked(True)
        self.img_show_test = QCheckBox("Test")
        self.img_show_test.setChecked(True)
        filt.addWidget(self.img_show_lean)
        filt.addWidget(self.img_show_test)
        filt.addStretch(1)
        lay.addLayout(filt)

        self._all_images = []
        self.image_table = self._make_table(
            ["Image", "Version", "Test", "Size", "Modified", "Match"]
        )
        # Click a header to sort; Size sorts by magnitude (see _NumSortItem).
        self.image_table.setSortingEnabled(True)
        lay.addWidget(self.image_table, 1)
        # Connect after the table exists so an early toggle can't outrun it.
        self.img_show_lean.toggled.connect(self._apply_image_filter)
        self.img_show_test.toggled.connect(self._apply_image_filter)
        return w

    def _fill_images_table(self, images) -> None:
        """Render ``images`` into the Images table (sort-aware, numeric Size)."""
        gui_version = core.this_gui_version()
        table = self.image_table
        table.setSortingEnabled(False)  # don't re-sort mid-fill
        table.setRowCount(len(images))
        for r, im in enumerate(images):
            # Match only on a clean release version (im.version); a
            # develop-<sha> image has version=None so never "matches".
            match = (
                "✓ this GUI"
                if im.version and im.version == gui_version
                else ""
            )
            table.setItem(r, 0, QTableWidgetItem(im.name))
            table.setItem(r, 1, QTableWidgetItem(im.tag))
            table.setItem(
                r, 2, QTableWidgetItem("test" if im.is_test else "lean")
            )
            table.setItem(r, 3, _NumSortItem(im.size, _size_to_bytes(im.size)))
            table.setItem(r, 4, QTableWidgetItem(im.mtime))
            table.setItem(r, 5, QTableWidgetItem(match))
        table.resizeColumnsToContents()
        table.setSortingEnabled(True)  # re-applies the current sort indicator

    def _apply_image_filter(self, *args) -> None:
        """Re-render the Images table for the Lean/Test checkbox selection."""
        show_lean = self.img_show_lean.isChecked()
        show_test = self.img_show_test.isChecked()
        filtered = [
            im
            for im in self._all_images
            if (im.is_test and show_test) or (not im.is_test and show_lean)
        ]
        self._fill_images_table(filtered)

    # -- test runs tab ----------------------------------------------------
    def _build_testruns_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        # Editable image combo, filled by the top 'Refresh images' button with
        # the mount's TEST images (the chain runs pytest inside the image, which
        # a lean image lacks). Defaults to the -test- image matching this GUI's
        # version when that forms a clean name (blank on a dev build).
        self.test_image = QComboBox()
        self.test_image.setEditable(True)
        self.test_image.setMinimumWidth(420)
        default = core.default_test_image_basename(core.this_gui_version())
        self.test_image.setCurrentText(default or "")
        self.test_image.setToolTip(
            "The .sif to run pytest inside (needs the [dev]/pytest extra, i.e. "
            "a picasso-workflow-test-<version>.sif). 'Refresh images' (top) "
            "fills this with the mount's test images; a bare name is resolved "
            "against ClusterApptainer.image_dir."
        )
        form.addRow("Test image", self.test_image)
        # Updated by 'Refresh images': warns when the mount has no test image.
        self.test_image_status = QLabel("")
        form.addRow("", self.test_image_status)
        self.test_data_dir = QLineEdit()
        self.test_data_dir.setMinimumWidth(420)
        # Pre-fill from config TestData.directory -- the same source
        # submit_all.sh resolves PW_TEST_DATA_DIR from, so a GUI run exercises
        # the same datasets. Editable; blank = tiers 1-3 only.
        self.test_data_dir.setText(core.configured_test_data_dir())
        self.test_data_dir.setPlaceholderText(
            "e.g. /fs/pool/…/picasso-workflow_testdata  (blank = skip "
            "tier 4 + templates)"
        )
        self.test_data_dir.setToolTip(
            "Cluster path to a directory of real DNA-PAINT test datasets, "
            "exported as PW_TEST_DATA_DIR + bound into the container (tier 4 "
            "runs real_data tests over it; template runs are the "
            "run_workflow_slurm.sh scripts found under it). Pre-filled from "
            "config.yaml TestData.directory -- the same key submit_all.sh "
            "uses. Must be valid on the cluster compute node. Blank = tiers "
            "1-3 only."
        )
        form.addRow("Test data dir", self.test_data_dir)
        # Which jobs to run. Tiers 1-3 are self-contained (no data); tier 4 +
        # templates additionally need the test data dir.
        self.test_tiers13 = QCheckBox("Tiers 1-3 (unit + integration)")
        self.test_tiers13.setToolTip(
            "Unit tests + template validation + synthetic-data integration, "
            "run as `pytest --pyargs picasso_workflow` inside the image. "
            "Self-contained -- needs no test data."
        )
        self.test_tier4 = QCheckBox("Tier 4 (real data)")
        self.test_tier4.setToolTip(
            "The test SUITE's real-data tier: pytest -m 'integration and "
            "real_data' over the datasets in the test data dir (the real_data-"
            "marked tests in test_real_data_integration.py / "
            "test_z_integration.py -- assertion-based correctness checks)."
        )
        self.test_templates = QCheckBox("Template workflow runs")
        self.test_templates.setToolTip(
            "End-to-end WORKFLOW runs (not pytest): submit each "
            "run_workflow_slurm.sh found under the test data dir -- each runs "
            "the full pipeline on its template's data; the verdict is 'ran "
            "through with no error'. Submitted as-is, so they must already be "
            "Apptainer-based (regenerate them from the Run tab) -- a conda-"
            "based script would need a login-node environment."
        )
        self.test_tiers13.setChecked(True)
        self.test_tier4.setChecked(True)
        self.test_templates.setChecked(True)
        toggles = QHBoxLayout()
        toggles.addWidget(self.test_tiers13)
        toggles.addWidget(self.test_tier4)
        toggles.addWidget(self.test_templates)
        toggles.addStretch(1)
        form.addRow("Run", _row_widget(toggles))
        lay.addLayout(form)
        bar = QHBoxLayout()
        submit = QPushButton("Submit test chain")
        submit.clicked.connect(self._submit_test_chain)
        fetch = QPushButton("Fetch SUMMARY")
        fetch.clicked.connect(self._fetch_summary)
        logs = QPushButton("Fetch logs")
        logs.setToolTip(
            "Tail the per-job .log files in the run dir (tiers + summary) so a "
            "failed tier's error output is visible here. Template workflow logs "
            "live under each template's own folder."
        )
        logs.clicked.connect(self._fetch_logs)
        cancel = QPushButton("Cancel run")
        cancel.setToolTip(
            "scancel every job of the last submitted run -- use to clear the "
            "tier4 / template jobs left pending when a gating tier fails. Only "
            "this run's jobs are cancelled, not your whole queue."
        )
        cancel.clicked.connect(self._cancel_run)
        bar.addWidget(submit)
        bar.addWidget(fetch)
        bar.addWidget(logs)
        bar.addWidget(cancel)
        bar.addStretch(1)
        lay.addLayout(bar)
        # Separate panes so submit info / SUMMARY / logs don't overwrite each
        # other -- each button writes (and switches) to its own tab.
        self.test_output_tabs = QTabWidget()
        self.test_submit_info = QPlainTextEdit(readOnly=True)
        self.test_summary_text = QPlainTextEdit(readOnly=True)
        self.test_logs_text = QPlainTextEdit(readOnly=True)
        self.test_output_tabs.addTab(self.test_submit_info, "Submit")
        self.test_output_tabs.addTab(self.test_summary_text, "SUMMARY")
        self.test_output_tabs.addTab(self.test_logs_text, "Logs")
        lay.addWidget(self.test_output_tabs, 1)
        return w

    def _submit_test_chain(self) -> None:
        version = core.this_gui_version()
        # A new submit supersedes any prior run: clear the tracked run up front
        # so a FAILED submit (which never reaches `done`) can't leave 'Fetch
        # logs' / 'Cancel run' pointing at the previous run's dir / job ids.
        self._test_run_dir = None
        self._test_run_jobs = []
        # Snapshot ALL widget inputs on the GUI thread (widgets aren't
        # thread-safe); the worker closure below touches only these locals.
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_sel = self.test_image.currentText().strip()
        data_dir = self.test_data_dir.text().strip() or None
        run_tiers13 = self.test_tiers13.isChecked()
        run_tier4 = self.test_tier4.isChecked()
        run_templates = self.test_templates.isChecked()

        def action(admin):
            from picasso_workflow.slurm_submit import (
                apptainer_config,
                resolve_cluster_image_path,
            )

            image_dir, _m, _b = apptainer_config(admin.host)
            image_path = resolve_cluster_image_path(image_sel, image_dir)
            if not image_path:
                raise RuntimeError(
                    "No test image selected (and no image_dir configured)."
                )
            return admin.submit_test_chain(
                image_path=image_path,
                version=version,
                timestamp=ts,
                test_data_dir=data_dir,
                run_tiers13=run_tiers13,
                run_tier4=run_tier4,
                run_templates=run_templates,
            )

        def done(result):
            self._test_run_dir = result["run_dir"]
            # Track every job id of this run so 'Cancel run' scancels exactly
            # these (not the user's whole queue); drop skipped tiers' None.
            self._test_run_jobs = [
                j
                for j in (
                    result["tiers_job"],
                    result["tier4_job"],
                    result["summary_job"],
                    *(jid for jid, _ in result["template_jobs"]),
                )
                if j
            ]
            lines = [
                f"run dir: {result['run_dir']}",
                f"test data dir: {result['test_data_dir'] or '(none)'}",
                f"tiers 1-3 job: {result['tiers_job'] or '(skipped)'}",
                f"tier 4 job: {result['tier4_job'] or '(skipped)'}",
                f"template jobs: {len(result['template_jobs'])}",
                f"summary job: {result['summary_job']}",
                "",
                "Submitted. Use 'Fetch SUMMARY' once the summary job finishes,"
                " or watch progress in the Jobs & usage tab.",
            ]
            self.test_submit_info.setPlainText("\n".join(lines))
            self.test_output_tabs.setCurrentWidget(self.test_submit_info)
            self.activity.appendPlainText(
                f"✓ submitted test chain -> {result['run_dir']}"
            )

        self._run(action, done, busy="submitting test chain")

    def _fetch_summary(self) -> None:
        if not self._test_run_dir:
            QMessageBox.information(
                self, "Fetch SUMMARY", "Submit a test chain first."
            )
            return
        run_dir = self._test_run_dir

        def done(text):
            self.test_summary_text.setPlainText(
                text
                or "(SUMMARY.txt not present yet — the summary job has "
                "not finished)"
            )
            self.test_output_tabs.setCurrentWidget(self.test_summary_text)

        self._run(
            lambda a: a.fetch_summary(run_dir),
            done,
            busy="fetching SUMMARY",
        )

    def _fetch_logs(self) -> None:
        if not self._test_run_dir:
            QMessageBox.information(
                self, "Fetch logs", "Submit a test chain first."
            )
            return
        run_dir = self._test_run_dir

        def done(text):
            self.test_logs_text.setPlainText(
                text
                or "(no .log files in the run dir yet — the jobs have not "
                "started)"
            )
            self.test_output_tabs.setCurrentWidget(self.test_logs_text)

        self._run(
            lambda a: a.fetch_logs(run_dir),
            done,
            busy="fetching logs",
        )

    def _confirm_and_cancel(
        self, ids, *, title, prompt, busy, note=""
    ) -> None:
        """Confirm, then ``scancel`` ``ids`` off the GUI thread; log the count.

        Shared by the Test-runs 'Cancel run' and the Jobs 'Cancel selected'
        buttons. ``note`` is appended to the success message.
        """
        if (
            QMessageBox.question(self, title, prompt)
            != QMessageBox.StandardButton.Yes
        ):
            return

        def done(n):
            self.activity.appendPlainText(f"✓ cancelled {n} job(s){note}")

        self._run(lambda a: a.cancel_jobs(ids), done, busy=busy)

    def _cancel_run(self) -> None:
        if not self._test_run_jobs:
            QMessageBox.information(
                self, "Cancel run", "Submit a test chain first."
            )
            return
        jobs = list(self._test_run_jobs)
        self._confirm_and_cancel(
            jobs,
            title="Cancel run",
            prompt=f"scancel {len(jobs)} job(s) of this test run?\n"
            + " ".join(jobs),
            busy="cancelling run jobs",
        )

    # -- jobs & usage tab -------------------------------------------------
    def _build_jobs_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        # Row 1: refresh + the sacct look-back window (changing it re-fetches).
        controls = QHBoxLayout()
        refresh = QPushButton("Refresh jobs & usage")
        refresh.clicked.connect(self._refresh_jobs)
        controls.addWidget(refresh)
        controls.addWidget(QLabel("Recent window:"))
        self.recent_days = QComboBox()
        for label, days in (
            ("1 day", 1),
            ("3 days", 3),
            ("7 days", 7),
            ("14 days", 14),
            ("30 days", 30),
        ):
            self.recent_days.addItem(label, days)
        self.recent_days.setCurrentIndex(1)  # default 3 days
        self.recent_days.setToolTip(
            "How far back `sacct` looks for finished jobs. Changing it "
            "re-fetches the queue + recent list."
        )
        self.recent_days.currentIndexChanged.connect(self._refresh_jobs)
        controls.addWidget(self.recent_days)
        controls.addStretch(1)
        lay.addLayout(controls)

        # Row 2: client-side filters (instant, no re-fetch) + cancel selected.
        filt = QHBoxLayout()
        filt.addWidget(QLabel("Filter:"))
        self.jobs_filter = QLineEdit()
        self.jobs_filter.setPlaceholderText(
            "substring match on job id / name / state …"
        )
        self.jobs_filter.textChanged.connect(self._apply_jobs_filter)
        filt.addWidget(self.jobs_filter, 1)
        self.jobs_current_only = QCheckBox("This run only")
        self.jobs_current_only.setToolTip(
            "Show only the jobs of the last submitted test run (by job id)."
        )
        self.jobs_current_only.toggled.connect(self._apply_jobs_filter)
        filt.addWidget(self.jobs_current_only)
        cancel_sel = QPushButton("Cancel selected")
        cancel_sel.setToolTip(
            "scancel the selected (and visible) rows in either table — use "
            "to cancel jobs of runs OTHER than the current one. Select rows "
            "with click / shift-click / ctrl-click."
        )
        cancel_sel.clicked.connect(self._cancel_selected_jobs)
        filt.addWidget(cancel_sel)
        lay.addLayout(filt)

        self.disk_label = QLabel("")
        lay.addWidget(self.disk_label)
        lay.addWidget(QLabel("Queue (squeue)"))
        self.queue_table = self._make_table(
            ["Job", "Name", "State", "Time", "Partition/Reason"]
        )
        self._make_selectable(self.queue_table)
        lay.addWidget(self.queue_table, 1)
        self.recent_label = QLabel("Recent jobs (sacct)")
        lay.addWidget(self.recent_label)
        self.recent_table = self._make_table(
            ["Job", "Name", "State", "Elapsed", "End"]
        )
        self._make_selectable(self.recent_table)
        lay.addWidget(self.recent_table, 1)
        return w

    # -- the async runner (single worker, re-entry guarded) ---------------
    def _run(self, action, on_done, *, busy: str) -> None:
        """Run ``action(admin) -> result`` off the UI thread.

        The connection fields are snapshotted here, on the GUI thread; the
        worker closure only touches the plain snapshot + the built
        :class:`core.ClusterAdmin`, never a Qt widget (widgets aren't
        thread-safe).
        """
        if self._worker is not None and self._worker.isRunning():
            self.activity.appendPlainText("(busy — wait for the current task)")
            return
        host = self.host.currentText().strip()
        user = self.user.text().strip()
        identity = self.identity.text().strip() or None
        if not host:
            QMessageBox.information(self, "Connect", "Select a cluster first.")
            return
        self.activity.appendPlainText(f"• {busy}")
        self.setCursor(Qt.CursorShape.WaitCursor)

        def fn():
            admin = core.build_admin(
                host, username=user, ssh_key_path=identity
            )
            return action(admin)

        worker = _Worker(fn, self)
        worker.done.connect(on_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(lambda w=worker: self._on_finished(w))
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_failed(self, msg: str) -> None:
        self.conn_status.setText("error")
        self.activity.appendPlainText(f"✗ {msg}")
        QMessageBox.critical(self, "Operation failed", msg)

    def _on_finished(self, worker) -> None:
        self.unsetCursor()
        if self._worker is worker:
            self._worker = None

    # -- actions ----------------------------------------------------------
    def _on_test(self) -> None:
        def done(ok):
            self.conn_status.setText("connected" if ok else "unreachable")
            self.activity.appendPlainText(
                "✓ connection ok" if ok else "✗ connection failed"
            )

        self._run(
            lambda a: a.test_connection(), done, busy="testing connection"
        )

    def _refresh_images(self) -> None:
        """List the mount once; fill the Images table AND the test dropdown.

        Shared by both tabs (one SSH listing): the Images table shows every
        image; the Test-runs dropdown gets only the TEST images (the chain runs
        pytest inside the image, which a lean image lacks).
        """

        def done(result):
            image_dir, images = result
            self.image_dir_label.setText(
                f"Shared folder: {image_dir or '(unset)'}"
            )
            self._all_images = images
            self._apply_image_filter()  # renders the table per the kind filter
            self._fill_test_images(images)
            n_test = sum(1 for im in images if im.is_test)
            self.activity.appendPlainText(
                f"✓ {len(images)} image(s), {n_test} test image(s)"
            )

        self._run(
            lambda a: (a.image_dir(), a.list_images()),
            done,
            busy="listing images",
        )

    def _fill_test_images(self, images):
        """Put the TEST images into the Test-runs dropdown (editable).

        A non-empty current text (a deliberately typed name/path, or the version
        default) is always preserved -- items are offered, but a typed entry is
        never silently replaced. When the mount has no test image, a warning
        label points the user at building/downloading one.
        """
        names = [im.name for im in images if im.is_test]
        current = self.test_image.currentText().strip()
        self.test_image.blockSignals(True)
        self.test_image.clear()
        self.test_image.addItems(names)
        # Keep the user's entry if they typed one; else default to the first
        # listed test image. clear() wipes the line-edit, so restore explicitly.
        if current:
            self.test_image.setCurrentText(current)
        elif names:
            self.test_image.setCurrentText(names[0])
        self.test_image.blockSignals(False)
        self.test_image_status.setText(
            ""
            if names
            else "⚠ No test image on the mount — build/download a "
            "picasso-workflow-test-<version>.sif (it carries pytest)."
        )

    def _refresh_jobs(self, *args) -> None:
        # Snapshot the look-back window on the GUI thread (the worker closure
        # must not touch widgets). ``*args`` absorbs the combo's signal arg.
        days = self.recent_days.currentData()

        def done(result):
            queue, recent, disk = result
            if disk is not None:
                self.disk_label.setText(
                    f"Disk ({disk.path}): {disk.used} used / {disk.size} "
                    f"({disk.use_pct}), {disk.avail} free"
                )
            else:
                self.disk_label.setText("Disk: (image_dir unset)")
            self.recent_label.setText(
                f"Recent jobs (sacct, last {days} day(s))"
            )
            self._fill_jobs(self.queue_table, queue, queue=True)
            self._fill_jobs(self.recent_table, recent, queue=False)
            self._apply_jobs_filter()
            self.activity.appendPlainText(
                f"✓ {len(queue)} queued, {len(recent)} recent"
            )

        def action(a):
            return a.queue(), a.recent_jobs(days=days), a.disk_usage()

        self._run(action, done, busy="fetching jobs & usage")

    def _apply_jobs_filter(self, *args) -> None:
        """Show/hide rows per the text filter and the 'This run only' toggle.

        Purely client-side (no SSH): both conditions must hold (AND) for a row
        to stay visible. ``*args`` absorbs the textChanged/toggled signal args.
        """
        text = self.jobs_filter.text().strip().lower()
        current_only = self.jobs_current_only.isChecked()
        run_ids = set(self._test_run_jobs)
        for table in (self.queue_table, self.recent_table):
            for r in range(table.rowCount()):
                id_item = table.item(r, 0)
                job_id = id_item.text() if id_item else ""
                visible = True
                if current_only and job_id not in run_ids:
                    visible = False
                if visible and text:
                    row_text = " ".join(
                        table.item(r, c).text() if table.item(r, c) else ""
                        for c in range(table.columnCount())
                    ).lower()
                    visible = text in row_text
                table.setRowHidden(r, not visible)

    def _cancel_selected_jobs(self) -> None:
        """``scancel`` the selected (and still-visible) rows in either table."""
        ids = []
        for table in (self.queue_table, self.recent_table):
            for index in table.selectionModel().selectedRows():
                r = index.row()
                if table.isRowHidden(r):
                    continue  # don't act on a row filtered out of view
                item = table.item(r, 0)
                if item and item.text():
                    ids.append(item.text())
        ids = list(dict.fromkeys(ids))  # de-dup, preserve order
        if not ids:
            QMessageBox.information(
                self,
                "Cancel selected",
                "Select one or more jobs (Queue or Recent table) first.",
            )
            return
        self._confirm_and_cancel(
            ids,
            title="Cancel selected",
            prompt=f"scancel {len(ids)} selected job(s)?\n" + " ".join(ids),
            busy="cancelling selected jobs",
            note="; click Refresh to update the view.",
        )

    @staticmethod
    def _fill_jobs(table, jobs, *, queue):
        table.setRowCount(len(jobs))
        for r, j in enumerate(jobs):
            cells = [j.job_id, j.name, j.state, j.elapsed, j.extra]
            for c, text in enumerate(cells):
                table.setItem(r, c, QTableWidgetItem(text))


def _install_excepthook() -> None:
    """Show unhandled exceptions in a dialog instead of killing the process."""
    if getattr(sys, "_pw_admin_hook", False):
        return

    def hook(exc_type, exc, tb):
        msg = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            app = QApplication.instance()
            if app is not None and QThread.currentThread() == app.thread():
                QMessageBox.critical(None, "Unexpected error", msg)
        finally:
            sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = hook
    sys._pw_admin_hook = True  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    _install_excepthook()
    window = AdminWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
