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

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
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


def _titled(label, widget) -> QWidget:
    """Stack a title ``label`` above ``widget`` (zero margins), for a splitter."""
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.addWidget(label)
    v.addWidget(widget, 1)
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
        self._test_run_templates: list[tuple[str, str]] = []
        # Auto-refresh (opt-in): one QTimer polls the ACTIVE tab quietly.
        self._summary_seen = False
        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._auto_tick)
        self.tabs = QTabWidget()
        self._images_tab = self._build_images_tab()
        self._testruns_tab = self._build_testruns_tab()
        self._jobs_tab = self._build_jobs_tab()
        self.tabs.addTab(self._images_tab, "Images")
        self.tabs.addTab(self._testruns_tab, "Test runs")
        self.tabs.addTab(self._jobs_tab, "Jobs & usage")
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
        # Wall-clock cap applied to each template workflow run (sbatch --time
        # override). Blank = leave each script's own limit. Default 4h so a
        # hung template is killed well before the partition max (and its
        # afterany summary job unblocks).
        self.test_template_time = QLineEdit("04:00:00")
        self.test_template_time.setMaximumWidth(140)
        self.test_template_time.setPlaceholderText("blank = no cap")
        self.test_template_time.setToolTip(
            "SLURM walltime (e.g. 04:00:00 or 1-00:00:00) capping each template "
            "workflow run via `sbatch --time`, so a hung run is killed instead "
            "of blocking the summary. Blank leaves each script's own --time. "
            "Only applies to template runs, not the tier jobs."
        )
        form.addRow("Template time cap", self.test_template_time)
        lay.addLayout(form)
        bar = QHBoxLayout()
        submit = QPushButton("Submit test chain")
        submit.clicked.connect(self._submit_test_chain)
        fetch = QPushButton("Fetch SUMMARY")
        fetch.clicked.connect(self._fetch_summary)
        logs = QPushButton("Fetch logs")
        logs.setToolTip(
            "Tail the per-job .log files: the run dir's tier + summary logs, "
            "AND each template run's own logs/ dir -- so a failed tier OR a "
            "failed end-to-end template workflow's error output is visible here."
        )
        logs.clicked.connect(self._fetch_logs)
        cancel = QPushButton("Cancel run")
        cancel.setToolTip(
            "scancel every job of the last submitted run -- use to clear the "
            "tier4 / template jobs left pending when a gating tier fails. Only "
            "this run's jobs are cancelled, not your whole queue."
        )
        cancel.clicked.connect(self._cancel_run)
        status = QPushButton("Refresh run status")
        status.setToolTip(
            "sacct the jobs of the last submitted run and show them in the "
            "table below (queued / running / completed / failed). Auto-"
            "refreshes while 'Auto-refresh' (Jobs & usage tab) is on."
        )
        status.clicked.connect(self._refresh_run_status)
        attach = QPushButton("Attach to run…")
        attach.setToolTip(
            "Re-attach to a previously submitted run (pick a recent run dir). "
            "The run's job ids are read from its jobs.tsv on the cluster, so "
            "you can reconnect after a GUI restart or a network drop and use "
            "'Refresh run status' / 'Fetch logs' / 'Cancel run' again."
        )
        attach.clicked.connect(self._attach_run)
        bar.addWidget(submit)
        bar.addWidget(fetch)
        bar.addWidget(logs)
        bar.addWidget(cancel)
        bar.addWidget(status)
        bar.addWidget(attach)
        bar.addStretch(1)
        lay.addLayout(bar)

        # Live status of the CURRENT run's jobs (one table; the Jobs & usage
        # tab's two tables, condensed and filtered to this run's job ids).
        self.run_status_label = QLabel("No test run submitted yet.")
        lay.addWidget(self.run_status_label)
        self.run_status_table = self._make_table(
            ["Job", "Name", "Phase", "State", "Elapsed", "End"]
        )
        lay.addWidget(self.run_status_table, 1)
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
        # Submitting replaces the tracked run, but only on SUCCESS (in `done`).
        # A failed submit -- e.g. an accidental double-click while a run is
        # already live -- must NOT drop the previous run's tracking, or
        # 'Refresh run status' / 'Fetch logs' / 'Cancel run' would lose the
        # still-running run. A confirm guards against replacing it by accident.
        if self._test_run_jobs and (
            QMessageBox.question(
                self,
                "A test run is already tracked",
                f"A previous test run is tracked ({len(self._test_run_jobs)} "
                "job(s)). Submit a NEW run and track that one instead?\n\n"
                "(To just see the current run's status, use 'Refresh run "
                "status'.)",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        version = core.this_gui_version()
        # Snapshot ALL widget inputs on the GUI thread (widgets aren't
        # thread-safe); the worker closure below touches only these locals.
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_sel = self.test_image.currentText().strip()
        data_dir = self.test_data_dir.text().strip() or None
        run_tiers13 = self.test_tiers13.isChecked()
        run_tier4 = self.test_tier4.isChecked()
        run_templates = self.test_templates.isChecked()
        template_time = self.test_template_time.text().strip() or None

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
                template_time=template_time,
            )

        def done(result):
            # Success -> this run replaces any previously tracked one.
            self._test_run_dir = result["run_dir"]
            self._summary_seen = False  # a new run -> a fresh SUMMARY to await
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
            # (folder, job_id) per template run -> their own logs/ dirs, for
            # 'Fetch logs' (the folder is the script's parent directory).
            self._test_run_templates = [
                (script.rsplit("/", 1)[0] if "/" in script else ".", jid)
                for jid, script in result["template_jobs"]
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
            # Populate the run-status table right away. Deferred so the submit
            # worker is fully torn down first (its `finished` handler clears
            # self._worker) -- otherwise the re-entry guard would skip it.
            QTimer.singleShot(0, lambda: self._refresh_run_status(quiet=True))

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
        templates = list(self._test_run_templates)

        def done(text):
            self.test_logs_text.setPlainText(
                text
                or "(no .log files yet — the tier / template jobs have not "
                "started)"
            )
            self.test_output_tabs.setCurrentWidget(self.test_logs_text)

        self._run(
            lambda a: a.fetch_logs(run_dir, templates=templates),
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

    def _refresh_run_status(self, *args, quiet: bool = False) -> None:
        """sacct the current run's jobs into the Test-runs status table.

        ``*args`` absorbs the button's ``clicked`` arg; ``quiet`` is the
        auto-refresh path (no busy chrome / dialog). Once every job has
        finished it fetches the SUMMARY once -- the live table replaces polling
        SUMMARY while jobs are still queued/running.
        """
        jobs = list(self._test_run_jobs)
        if not jobs:
            if not quiet:
                QMessageBox.information(
                    self, "Run status", "Submit a test chain first."
                )
            return

        def done(infos):
            counts = core.summarize_phases(infos)
            text = (
                f"{len(infos)} job(s): {counts['queued']} queued · "
                f"{counts['running']} running · {counts['completed']} "
                f"completed · {counts['failed']} failed"
            )
            if counts["other"]:
                text += f" · {counts['other']} other"
            self.run_status_label.setText(text)
            self._fill_run_status(infos)
            if (
                quiet
                and infos
                and not self._summary_seen
                and counts["queued"] == 0
                and counts["running"] == 0
            ):
                self._auto_fetch_summary()

        self._run(
            lambda a: a.jobs_status(jobs),
            done,
            busy="fetching run status",
            quiet=quiet,
        )

    def _attach_run(self, *args) -> None:
        """Re-attach to a previous run (pick a recent run dir, read jobs.tsv).

        Lets the console reconnect to a run after a GUI restart / network drop:
        its job ids live in ``<run_dir>/jobs.tsv`` on the cluster, so attaching
        repopulates the tracked run and 'Refresh run status' / 'Fetch logs' /
        'Cancel run' work again. Two SSH steps (list recent runs, then read the
        chosen one); the picker modal in between lets the first worker retire,
        so the second op isn't skipped by the re-entry guard.
        """

        def picked(runs):
            if runs:
                run_dir, ok = QInputDialog.getItem(
                    self,
                    "Attach to run",
                    "Pick a recent run directory (newest first), or type a "
                    "path:",
                    runs,
                    0,
                    True,
                )
            else:
                run_dir, ok = QInputDialog.getText(
                    self,
                    "Attach to run",
                    "No recent runs found. Enter a run directory on the "
                    "cluster:",
                )
            run_dir = (run_dir or "").strip()
            if not ok or not run_dir:
                return

            def attached(result):
                self._test_run_dir = result["run_dir"]
                self._test_run_jobs = list(result["job_ids"])
                self._test_run_templates = [
                    tuple(t) for t in result["templates"]
                ]
                self._summary_seen = False
                self.activity.appendPlainText(
                    f"✓ attached to run {result['run_dir']} "
                    f"({len(self._test_run_jobs)} job(s))"
                )
                # Deferred so the attach worker tears down first (its `finished`
                # clears self._worker) -- otherwise the re-entry guard in _run
                # would skip this status refresh.
                QTimer.singleShot(
                    0, lambda: self._refresh_run_status(quiet=True)
                )

            self._run(
                lambda a: a.attach_run(run_dir),
                attached,
                busy="attaching to run",
            )

        self._run(
            lambda a: a.list_test_runs(), picked, busy="listing recent runs"
        )

    def _fill_run_status(self, jobs) -> None:
        self.run_status_table.setRowCount(len(jobs))
        for r, j in enumerate(jobs):
            cells = [
                j.job_id,
                j.name,
                core.job_phase(j.state),
                j.state,
                j.elapsed,
                j.extra,
            ]
            for c, text in enumerate(cells):
                self.run_status_table.setItem(r, c, QTableWidgetItem(text))

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
        # Opt-in auto-refresh: quietly polls the ACTIVE tab (Jobs tables, or
        # SUMMARY on the Test-runs tab until it appears). Images stay manual.
        self.auto_refresh = QCheckBox("Auto-refresh every")
        self.auto_refresh.setToolTip(
            "Periodically refresh the active tab without clicking: the Jobs "
            "tables here, or (on the Test-runs tab) fetch SUMMARY until it "
            "appears. Quiet; pauses while you have a selection or a dialog "
            "open, and stops on an SSH error."
        )
        self.auto_refresh.toggled.connect(self._on_auto_toggled)
        self.auto_interval = QSpinBox()
        self.auto_interval.setRange(5, 600)
        self.auto_interval.setValue(20)
        self.auto_interval.setSuffix(" s")
        self.auto_interval.valueChanged.connect(self._on_auto_interval_changed)
        controls.addWidget(self.auto_refresh)
        controls.addWidget(self.auto_interval)
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

        # Queue / recent tables and the over-time stats graph share a resizable
        # vertical splitter (the tables are table-heavy; the graph sits below).
        self.queue_table = self._make_table(
            ["Job", "Name", "State", "Time", "Partition/Reason"]
        )
        self._make_selectable(self.queue_table)
        self.recent_label = QLabel("Recent jobs (sacct)")
        self.recent_table = self._make_table(
            ["Job", "Name", "State", "Elapsed", "End"]
        )
        self._make_selectable(self.recent_table)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(_titled(QLabel("Queue (squeue)"), self.queue_table))
        split.addWidget(_titled(self.recent_label, self.recent_table))
        split.addWidget(self._build_stats_widget())
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 2)
        split.setStretchFactor(2, 3)
        lay.addWidget(split, 1)
        return w

    def _build_stats_widget(self) -> QWidget:
        """A titled matplotlib canvas: #jobs/day + pass-fraction over time.

        The figure is fed by :meth:`_update_stats_plot` from the same ``sacct``
        data the Recent table uses (no extra SSH). matplotlib is imported lazily
        (it is a heavy dependency, and the console is usable without ever
        opening this tab)."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        self._stats_figure = Figure(figsize=(6, 2.6), layout="tight")
        self._stats_canvas = FigureCanvasQTAgg(self._stats_figure)
        self._stats_canvas.setMinimumHeight(180)
        self._update_stats_plot([])  # draw the empty-state placeholder
        return _titled(
            QLabel("Job statistics (finished jobs over the recent window)"),
            self._stats_canvas,
        )

    def _update_stats_plot(self, jobs) -> None:
        """Redraw the stats graph from ``jobs`` (the Recent/sacct list).

        Bars = passed/failed jobs per day (stacked); the line = pass fraction
        (%) on a right-hand axis. Empty when the window has no finished jobs.
        """
        stats = core.aggregate_jobs_by_day(jobs)
        fig = self._stats_figure
        fig.clear()
        ax = fig.add_subplot(111)
        if not stats:
            ax.text(
                0.5,
                0.5,
                "no finished jobs in the window",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="gray",
            )
            ax.set_axis_off()
            self._stats_canvas.draw_idle()
            return
        x = list(range(len(stats)))
        passed = [s.passed for s in stats]
        failed = [s.failed for s in stats]
        ax.bar(x, passed, color="#4caf50", label="passed")
        ax.bar(x, failed, bottom=passed, color="#e53935", label="failed")
        ax.set_ylabel("jobs")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.date[5:] for s in stats], rotation=45, ha="right", fontsize=8
        )
        ax.legend(loc="upper left", fontsize=8)
        ax2 = ax.twinx()
        frac = [
            s.pass_fraction * 100 if s.pass_fraction is not None else None
            for s in stats
        ]
        ax2.plot(x, frac, color="#1565c0", marker="o", label="pass %")
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("pass %")
        total = sum(s.total for s in stats)
        passed_total = sum(s.passed for s in stats)
        overall = (100 * passed_total / total) if total else 0
        ax.set_title(
            f"{total} finished job(s) over {len(stats)} day(s) — "
            f"{overall:.0f}% passed",
            fontsize=9,
        )
        self._stats_canvas.draw_idle()

    # -- the async runner (single worker, re-entry guarded) ---------------
    def _run(self, action, on_done, *, busy: str, quiet: bool = False) -> None:
        """Run ``action(admin) -> result`` off the UI thread.

        The connection fields are snapshotted here, on the GUI thread; the
        worker closure only touches the plain snapshot + the built
        :class:`core.ClusterAdmin`, never a Qt widget (widgets aren't
        thread-safe). ``quiet`` (used by auto-refresh) suppresses the busy /
        wait-cursor chrome and routes a failure to a non-modal, auto-stopping
        handler instead of a dialog.
        """
        if self._worker is not None and self._worker.isRunning():
            if not quiet:
                self.activity.appendPlainText(
                    "(busy — wait for the current task)"
                )
            return
        host = self.host.currentText().strip()
        user = self.user.text().strip()
        identity = self.identity.text().strip() or None
        if not host:
            if not quiet:
                QMessageBox.information(
                    self, "Connect", "Select a cluster first."
                )
            return
        self._op_quiet = quiet
        if not quiet:
            self.activity.appendPlainText(f"• {busy}")
            self.setCursor(Qt.CursorShape.WaitCursor)

        def fn():
            admin = core.build_admin(
                host, username=user, ssh_key_path=identity
            )
            return action(admin)

        worker = _Worker(fn, self)
        # Mark the connection live again on ANY successful op (fires before the
        # caller's on_done), so a stale 'error' left by a transient auto-refresh
        # blip clears once a later tick succeeds -- not just after a manual
        # 'Test connection'. A test_connection that returns False still ends on
        # 'unreachable' because _on_test's on_done runs after this and wins.
        worker.done.connect(self._on_op_ok)
        worker.done.connect(on_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(lambda w=worker: self._on_finished(w))
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_op_ok(self, _result=None) -> None:
        """Any successful op -> the connection is live (clears a stale error)."""
        self.conn_status.setText("connected")

    def _on_failed(self, msg: str) -> None:
        self.conn_status.setText("error")
        if getattr(self, "_op_quiet", False):
            # An auto-refresh tick failed: log it but KEEP auto-refresh on, so a
            # transient SSH/network blip reconnects on the next tick instead of
            # making the user re-enable it. No modal; ticks are >=5s apart, so
            # a persistent outage logs at most once per interval (not a storm).
            self.activity.appendPlainText(
                f"✗ auto-refresh error (will retry): {msg}"
            )
            return
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

    def _refresh_jobs(self, *args, quiet: bool = False) -> None:
        # Snapshot the look-back window on the GUI thread (the worker closure
        # must not touch widgets). ``*args`` absorbs the combo's signal arg;
        # ``quiet`` is the auto-refresh path (no log, preserve scroll).
        days = self.recent_days.currentData()

        def done(result):
            queue, recent, disk = result
            # Preserve scroll on a quiet auto-refresh so the view doesn't jump.
            qv = self.queue_table.verticalScrollBar().value()
            rv = self.recent_table.verticalScrollBar().value()
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
            # The stats graph reuses the same sacct 'recent' data (no extra SSH).
            self._update_stats_plot(recent)
            self._apply_jobs_filter()
            if quiet:
                self.queue_table.verticalScrollBar().setValue(qv)
                self.recent_table.verticalScrollBar().setValue(rv)
            else:
                self.activity.appendPlainText(
                    f"✓ {len(queue)} queued, {len(recent)} recent"
                )

        def action(a):
            return a.queue(), a.recent_jobs(days=days), a.disk_usage()

        self._run(action, done, busy="fetching jobs & usage", quiet=quiet)

    # -- auto-refresh (opt-in; one timer polls the active tab, quietly) ----
    def _on_auto_toggled(self, on: bool) -> None:
        if on:
            self._auto_timer.start(self.auto_interval.value() * 1000)
        else:
            self._auto_timer.stop()

    def _on_auto_interval_changed(self, seconds: int) -> None:
        if self._auto_timer.isActive():
            self._auto_timer.start(seconds * 1000)

    def _auto_tick(self) -> None:
        """Quietly refresh the active tab's data; skip when it would intrude."""
        if self._worker is not None and self._worker.isRunning():
            return  # an op is in flight; try next tick
        if QApplication.activeModalWidget() is not None:
            return  # a dialog is open -- don't fire SSH underneath it
        if not self.host.currentText().strip():
            return  # not configured
        current = self.tabs.currentWidget()
        if current is self._jobs_tab:
            # Don't clobber a selection the user is building for Cancel selected.
            if any(
                t.selectionModel().selectedRows()
                for t in (self.queue_table, self.recent_table)
            ):
                return
            self._refresh_jobs(quiet=True)
        elif current is self._testruns_tab and self._test_run_jobs:
            # Refresh the live run-status table; once all jobs finish it pulls
            # the SUMMARY once (see _refresh_run_status).
            self._refresh_run_status(quiet=True)

    def _auto_fetch_summary(self) -> None:
        run_dir = self._test_run_dir

        def done(text):
            if text:
                self.test_summary_text.setPlainText(text)
                self.test_output_tabs.setCurrentWidget(self.test_summary_text)
                self._summary_seen = True
                self.activity.appendPlainText("✓ SUMMARY ready")
            # empty -> summary job not finished; leave the pane, retry next tick

        self._run(
            lambda a: a.fetch_summary(run_dir),
            done,
            busy="auto-fetching SUMMARY",
            quiet=True,
        )

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
