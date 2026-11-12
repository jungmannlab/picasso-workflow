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

import getpass
import sys
import traceback

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
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
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_images_tab(), "Images")
        self.tabs.addTab(self._build_jobs_tab(), "Jobs & usage")
        root.addWidget(self.tabs, 1)

        root.addWidget(QLabel("Activity"))
        self.activity = QPlainTextEdit(readOnly=True)
        self.activity.setMaximumBlockCount(5000)
        root.addWidget(self.activity, 1)

        self.resize(820, 680)

    # -- connection panel -------------------------------------------------
    def _build_connection_panel(self) -> QWidget:
        box = QWidget()
        form = QFormLayout(box)
        self.host = QComboBox()
        self.host.setEditable(True)
        self.host.addItems([str(h) for h in core.cluster_hosts()])
        self.user = QLineEdit(getpass.getuser())
        self.identity = QLineEdit(core.discover_ssh_key() or "")
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

    # -- images tab -------------------------------------------------------
    def _build_images_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        refresh = QPushButton("Refresh images")
        refresh.clicked.connect(self._refresh_images)
        lay.addWidget(refresh)
        self.image_dir_label = QLabel("")
        lay.addWidget(self.image_dir_label)
        self.image_table = QTableWidget(0, 6)
        self.image_table.setHorizontalHeaderLabels(
            ["Image", "Version", "Test", "Size", "Modified", "Match"]
        )
        self.image_table.horizontalHeader().setStretchLastSection(True)
        self.image_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        lay.addWidget(self.image_table, 1)
        return w

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
        self.queue_table = QTableWidget(0, 5)
        self.queue_table.setHorizontalHeaderLabels(
            ["Job", "Name", "State", "Time", "Partition/Reason"]
        )
        self.queue_table.horizontalHeader().setStretchLastSection(True)
        self.queue_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        lay.addWidget(self.queue_table, 1)
        lay.addWidget(QLabel("Recent jobs (sacct, last 3 days)"))
        self.recent_table = QTableWidget(0, 5)
        self.recent_table.setHorizontalHeaderLabels(
            ["Job", "Name", "State", "Elapsed", "End"]
        )
        self.recent_table.horizontalHeader().setStretchLastSection(True)
        self.recent_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
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
                match = "✓ this GUI" if im.version == gui_version else ""
                cells = [
                    im.name,
                    im.version or "",
                    "yes" if im.is_test else "",
                    im.size,
                    im.mtime,
                    match,
                ]
                for c, text in enumerate(cells):
                    self.image_table.setItem(r, c, QTableWidgetItem(text))
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
