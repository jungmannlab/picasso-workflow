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
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_images_tab(), "Images")
        self.tabs.addTab(self._build_testruns_tab(), "Test runs")
        self.tabs.addTab(self._build_jobs_tab(), "Jobs & usage")
        root.addWidget(self.tabs, 1)

        root.addWidget(QLabel("Activity"))
        self.activity = QPlainTextEdit(readOnly=True)
        self.activity.setMaximumBlockCount(5000)
        root.addWidget(self.activity, 1)

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
        test = QPushButton("Test connection")
        test.clicked.connect(self._on_test)
        form.addRow(test, self.conn_status)
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

    # -- images tab -------------------------------------------------------
    def _build_images_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        refresh = QPushButton("Refresh images")
        refresh.clicked.connect(self._refresh_images)
        lay.addWidget(refresh)
        self.image_dir_label = QLabel("")
        lay.addWidget(self.image_dir_label)
        self.image_table = self._make_table(
            ["Image", "Version", "Test", "Size", "Modified", "Match"]
        )
        lay.addWidget(self.image_table, 1)
        return w

    # -- test runs tab ----------------------------------------------------
    def _build_testruns_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        # Editable image combo + a List button that fills it from the shared
        # mount (test images first). Defaults to the -test- image matching this
        # GUI's version when that forms a clean name (blank on a dev build).
        self.test_image = QComboBox()
        self.test_image.setEditable(True)
        self.test_image.setMinimumWidth(420)
        default = core.default_test_image_basename(core.this_gui_version())
        self.test_image.setCurrentText(default or "")
        self.test_image.setToolTip(
            "The .sif to run pytest inside (needs the [dev]/pytest extra, i.e. "
            "a picasso-workflow-test-<version>.sif). Click 'List' to pick from "
            "the shared mount; a bare name is resolved against "
            "ClusterApptainer.image_dir."
        )
        list_images = QPushButton("List")
        list_images.setToolTip(
            "List the test .sif images (those carrying pytest) on the shared "
            "folder over SSH. Use the Images tab to see lean images too."
        )
        list_images.clicked.connect(self._list_test_images)
        img_row = QHBoxLayout()
        img_row.addWidget(self.test_image, 1)
        img_row.addWidget(list_images)
        form.addRow("Test image", _row_widget(img_row))
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
        self.test_templates = QCheckBox("Template workflow runs")
        self.test_templates.setToolTip(
            "Submit each run_workflow_slurm.sh found under the test data dir. "
            "These are submitted as-is, so they must already be Apptainer-"
            "based (regenerate them from the Run tab) -- a conda-based script "
            "would need a login-node environment."
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
        bar.addWidget(submit)
        bar.addWidget(fetch)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.test_summary = QPlainTextEdit(readOnly=True)
        lay.addWidget(self.test_summary, 1)
        return w

    def _list_test_images(self) -> None:
        """Populate the test-image combo from the shared mount (test first)."""

        def done(images):
            # Only TEST images: the chain runs pytest inside the image, which a
            # lean image lacks, so a lean pick would just fail. The combo stays
            # editable, so an oddly-named test image can still be typed.
            names = [im.name for im in images if im.is_test]
            if not names:
                self.activity.appendPlainText(
                    f"✓ {len(images)} image(s), 0 test image(s)"
                )
                QMessageBox.information(
                    self,
                    "No test images",
                    "No test images (picasso-workflow-test-<version>.sif) on "
                    "the shared mount. The test chain runs pytest inside the "
                    "image, which the lean image doesn't carry -- build or "
                    "download a -test- image (or type its name).",
                )
                return
            current = self.test_image.currentText().strip()
            self.test_image.blockSignals(True)
            self.test_image.clear()
            self.test_image.addItems(names)
            if current in names:
                self.test_image.setCurrentText(current)
            else:
                self.test_image.setCurrentText(names[0])
            self.test_image.blockSignals(False)
            self.activity.appendPlainText(
                f"✓ {len(names)} test image(s) of {len(images)}"
            )

        self._run(lambda a: a.list_images(), done, busy="listing images")

    def _submit_test_chain(self) -> None:
        version = core.this_gui_version()
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
            lines = [
                f"run dir: {result['run_dir']}",
                f"tiers 1-3 job: {result['tiers_job'] or '(skipped)'}",
                f"tier 4 job: {result['tier4_job'] or '(skipped)'}",
                f"template jobs: {len(result['template_jobs'])}",
                f"summary job: {result['summary_job']}",
                "",
                "Submitted. Use 'Fetch SUMMARY' once the summary job finishes,"
                " or watch progress in the Jobs & usage tab.",
            ]
            self.test_summary.setPlainText("\n".join(lines))
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
            self.test_summary.setPlainText(
                text
                or "(SUMMARY.txt not present yet — the summary job has "
                "not finished)"
            )

        self._run(
            lambda a: a.fetch_summary(run_dir),
            done,
            busy="fetching SUMMARY",
        )

    # -- jobs & usage tab -------------------------------------------------
    def _build_jobs_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        refresh = QPushButton("Refresh jobs & usage")
        refresh.clicked.connect(self._refresh_jobs)
        lay.addWidget(refresh)
        self.disk_label = QLabel("")
        lay.addWidget(self.disk_label)
        lay.addWidget(QLabel("Queue (squeue)"))
        self.queue_table = self._make_table(
            ["Job", "Name", "State", "Time", "Partition/Reason"]
        )
        lay.addWidget(self.queue_table, 1)
        lay.addWidget(QLabel("Recent jobs (sacct, last 3 days)"))
        self.recent_table = self._make_table(
            ["Job", "Name", "State", "Elapsed", "End"]
        )
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
        gui_version = core.this_gui_version()

        def done(result):
            image_dir, images = result
            self.image_dir_label.setText(
                f"Shared folder: {image_dir or '(unset)'}"
            )
            self.image_table.setRowCount(len(images))
            for r, im in enumerate(images):
                # Match only on a clean release version (im.version); a
                # develop-<sha> image has version=None so never "matches".
                match = (
                    "✓ this GUI"
                    if im.version and im.version == gui_version
                    else ""
                )
                cells = [
                    im.name,
                    im.tag,  # "0.9.0" or "develop-<sha>" (always shown)
                    "test" if im.is_test else "lean",
                    im.size,
                    im.mtime,
                    match,
                ]
                for c, text in enumerate(cells):
                    self.image_table.setItem(r, c, QTableWidgetItem(text))
            self.image_table.resizeColumnsToContents()
            self.activity.appendPlainText(f"✓ {len(images)} image(s)")

        self._run(
            lambda a: (a.image_dir(), a.list_images()),
            done,
            busy="listing images",
        )

    def _refresh_jobs(self) -> None:
        def done(result):
            queue, recent, disk = result
            if disk is not None:
                self.disk_label.setText(
                    f"Disk ({disk.path}): {disk.used} used / {disk.size} "
                    f"({disk.use_pct}), {disk.avail} free"
                )
            else:
                self.disk_label.setText("Disk: (image_dir unset)")
            self._fill_jobs(self.queue_table, queue, queue=True)
            self._fill_jobs(self.recent_table, recent, queue=False)
            self.activity.appendPlainText(
                f"✓ {len(queue)} queued, {len(recent)} recent"
            )

        def action(a):
            return a.queue(), a.recent_jobs(), a.disk_usage()

        self._run(action, done, busy="fetching jobs & usage")

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
