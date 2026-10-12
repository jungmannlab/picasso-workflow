#!/usr/bin/env python
"""The GUI "Render" tab: explore a run's localizations with picasso Render.

A left panel lists the localization files a pipeline run saved (one entry
per ``*.hdf5`` under the run's module folders), selectable singly or in
multiples. The right panel embeds the picasso Render GUI's canvas (its
``View`` widget) and its *exploration* features: zoom, pan, contrast,
colormap, blur, scale bar, multichannel display, measuring and picking.
On picasso 0.12+ these are offered as an icon toolbar built from picasso
Render's own actions (``_TOOLBAR_ACTIONS``); on picasso 0.11 (no icon
theme) a set of plain buttons is used instead. Either way the tab stays
an explorer -- file I/O (open / save / export) and the heavier processing
(undrift / cluster / link / pick-similar) are left to standalone Render.

The embedded picasso Render window is created lazily on first use, so
importing this module (and building the tab) does not pull picasso's heavy
render-GUI dependencies at GUI startup.

Author: Heinrich Grabmayr
"""

from __future__ import annotations

import os

from loguru import logger
from PyQt6 import QtCore, QtWidgets

# non-locs artefacts that live next to the locs files in a run folder
_SKIP_DIRS = {
    "assets",
    "logs",
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
    "_pwf_claims",
}

# The exploration actions surfaced on the embedded icon toolbar (picasso
# 0.12+), as ``(menu path, picasso icon name)`` pairs; ``None`` is a
# separator. These are picasso Render's own navigation (zoom / pan /
# measure), picking (pick tool, pick-shape settings, clear picks) and
# display (fit, contrast / colormap, channels, info, 3D) actions -- the
# same QActions the standalone window uses, so they behave identically.
# File I/O (Open / Save / Export) and processing (Pick similar, undrift,
# cluster, link, ...) are intentionally left out; the tab stays an
# explorer. Paths are matched by ``toolbars.menu_actions`` (texts without
# the "..." / "&"); a path not found is skipped, so the toolbar degrades
# gracefully if picasso renames an action.
_TOOLBAR_ACTIONS = [
    ("Tools > Zoom", "tool-zoom"),
    ("Tools > Pick", "tool-pick"),
    ("Tools > Measure", "tool-measure"),
    ("Tools > Move", "tool-move"),
    None,
    ("Tools > Tools settings", "tools-settings"),
    ("Tools > Clear picks", "clear"),
    None,
    ("View > Fit image to window", "fit-view"),
    ("View > Display settings", "display-settings"),
    ("View > Files", "files"),
    ("View > Show info", "info"),
    ("View > 3D view", "view-3d"),
]


def enumerate_locs_files(run_folder: str) -> list[tuple[str, str]]:
    """List the localization files saved under a run folder.

    Scans the run folder on disk (rather than trusting the absolute paths
    recorded in ``WorkflowRunner.yaml``, which may be another machine's
    cluster paths) for ``*.hdf5`` files, so it works regardless of where the
    run was produced.

    Parameters
    ----------
    run_folder : str
        A run's result folder (contains ``*WorkflowRunner.yaml`` and the
        per-module subfolders).

    Returns
    -------
    list of (str, str)
        ``(label, absolute_path)`` pairs, sorted by label. The label is the
        file's path relative to ``run_folder`` (e.g.
        ``"03_picasso_localize/locs.hdf5"``).
    """
    found: list[tuple[str, str]] = []
    if not run_folder or not os.path.isdir(run_folder):
        return found
    run_folder = os.path.abspath(run_folder)
    for root, dirs, files in os.walk(run_folder):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            if not name.endswith(".hdf5"):
                continue
            path = os.path.join(root, name)
            label = os.path.relpath(path, run_folder)
            found.append((label, path))
    return sorted(found)


class RenderTab(QtWidgets.QWidget):
    """A tab to explore a run's localizations with embedded picasso Render."""

    # tree item role carrying a leaf's absolute locs file path
    _PATH_ROLE = QtCore.Qt.ItemDataRole.UserRole

    def __init__(self, main_window) -> None:
        """Initialize the tab.

        Parameters
        ----------
        main_window : picasso_workflow.gui.Window
            The picasso-workflow main window, used to read the current
            results folder and reuse its run discovery.
        """
        super().__init__()
        self.main_window = main_window
        # the picasso Render window, created lazily on first render and kept
        # alive for its dialogs; only its ``view`` widget is embedded.
        self._render_window = None
        self._embedded_view = None
        # one-shot guard: connect the app-shutdown worker-stop hook once
        self._quit_hook_connected = False
        self._build_ui()

    # -- UI construction ----------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # --- left: run selector + locs tree + actions ---
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(QtWidgets.QLabel("Run:"))
        self.run_combo = QtWidgets.QComboBox()
        self.run_combo.currentIndexChanged.connect(self._on_run_changed)
        run_row.addWidget(self.run_combo, 1)
        refresh_button = QtWidgets.QPushButton("Refresh")
        refresh_button.setToolTip(
            "Rescan the results folder for runs and their saved "
            "localization files."
        )
        refresh_button.clicked.connect(self.refresh)
        run_row.addWidget(refresh_button)
        left_layout.addLayout(run_row)

        self.locs_tree = QtWidgets.QTreeWidget()
        self.locs_tree.setHeaderLabels(["Localization files"])
        self.locs_tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.locs_tree.itemDoubleClicked.connect(
            lambda *_: self.render_selected(replace=True)
        )
        left_layout.addWidget(self.locs_tree, 1)

        render_button = QtWidgets.QPushButton("Render selected")
        render_button.setToolTip(
            "Render the selected file(s) in the canvas, replacing what is "
            "shown. Select several files to overlay them as channels."
        )
        render_button.clicked.connect(
            lambda: self.render_selected(replace=True)
        )
        left_layout.addWidget(render_button)
        add_button = QtWidgets.QPushButton("Add selected to view")
        add_button.setToolTip(
            "Overlay the selected file(s) on top of what is already shown."
        )
        add_button.clicked.connect(lambda: self.render_selected(replace=False))
        left_layout.addWidget(add_button)

        # --- right: embedded picasso Render canvas + exploration controls ---
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        controls = QtWidgets.QHBoxLayout()
        # picasso 0.12's native exploration actions (navigation, picking,
        # display) as an icon toolbar; populated lazily once a window is
        # embedded, and hidden on picasso 0.11 (no icon theme) -- see
        # ``_populate_render_toolbar``. It supersedes the equivalent plain
        # buttons below, which stay as the picasso-0.11 fallback.
        self.render_toolbar = QtWidgets.QToolBar()
        self.render_toolbar.setMovable(False)
        self.render_toolbar.setFloatable(False)
        self.render_toolbar.setVisible(False)
        controls.addWidget(self.render_toolbar)
        self.display_button = QtWidgets.QPushButton("Display settings")
        self.display_button.setToolTip(
            "Contrast, colormap, blur method, zoom and scale bar "
            "(picasso Render display settings)."
        )
        self.display_button.clicked.connect(self._open_display_settings)
        controls.addWidget(self.display_button)
        self.channels_button = QtWidgets.QPushButton("Channels")
        self.channels_button.setToolTip(
            "Per-channel color and visibility for multi-file overlays."
        )
        self.channels_button.clicked.connect(self._open_channels)
        controls.addWidget(self.channels_button)
        self.zoom_in_button = QtWidgets.QPushButton("Zoom in")
        self.zoom_in_button.setToolTip(
            "Zoom in. In the canvas you can also drag a box to zoom to it, "
            "right-drag to pan, and Ctrl+scroll to zoom at the cursor."
        )
        self.zoom_in_button.clicked.connect(self._zoom_in)
        controls.addWidget(self.zoom_in_button)
        self.zoom_out_button = QtWidgets.QPushButton("Zoom out")
        self.zoom_out_button.clicked.connect(self._zoom_out)
        controls.addWidget(self.zoom_out_button)
        self.fit_button = QtWidgets.QPushButton("Fit in view")
        self.fit_button.setToolTip("Reset the zoom to show all localizations.")
        self.fit_button.clicked.connect(self._fit_in_view)
        controls.addWidget(self.fit_button)
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_view)
        controls.addWidget(self.clear_button)
        controls.addStretch(1)
        right_layout.addLayout(controls)

        # canvas placeholder; the embedded picasso window is added here lazily
        self.canvas_container = QtWidgets.QWidget()
        self.canvas_layout = QtWidgets.QVBoxLayout(self.canvas_container)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_placeholder = QtWidgets.QLabel(
            "Select a run and localization file, then 'Render selected' to "
            "explore it here with picasso Render.\n\n"
            "Zoom with the Zoom in / Zoom out buttons, by dragging a box in "
            "the canvas, or with Ctrl+scroll; right-drag to pan; "
            "'Fit in view' resets the zoom."
        )
        self.canvas_placeholder.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.canvas_placeholder.setWordWrap(True)
        self.canvas_placeholder.setStyleSheet("color: #666;")
        self.canvas_layout.addWidget(self.canvas_placeholder)
        right_layout.addWidget(self.canvas_container, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 900])

        outer = QtWidgets.QVBoxLayout(self)
        outer.addWidget(splitter)
        self._set_controls_enabled(False)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for btn in (
            self.display_button,
            self.channels_button,
            self.zoom_in_button,
            self.zoom_out_button,
            self.fit_button,
            self.clear_button,
        ):
            btn.setEnabled(enabled)

    # -- run / file discovery ----------------------------------------------

    def refresh(self) -> None:
        """Rescan the results folder for runs and repopulate the selector."""
        base = self.main_window.results_folder_display.text().strip()
        for q in ('"', "'"):
            if base[:1] == q and base[-1:] == q:
                base = base[1:-1]
        previous = self.run_combo.currentData()
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        runs = self.main_window._find_runs(base) if base else []
        for run in runs:
            label = os.path.relpath(run, base) if base else run
            self.run_combo.addItem(label, run)
        self.run_combo.blockSignals(False)
        if previous is not None:
            idx = self.run_combo.findData(previous)
            if idx >= 0:
                self.run_combo.setCurrentIndex(idx)
        self._populate_tree()

    def _on_run_changed(self, _index: int) -> None:
        self._populate_tree()

    def _populate_tree(self) -> None:
        """Fill the tree with the current run's locs files, grouped by dir."""
        self.locs_tree.clear()
        run = self.run_combo.currentData()
        if not run:
            return
        groups: dict[str, QtWidgets.QTreeWidgetItem] = {}
        for label, path in enumerate_locs_files(run):
            head, tail = os.path.split(label)
            group_key = head or "."
            group = groups.get(group_key)
            if group is None:
                group = QtWidgets.QTreeWidgetItem([group_key])
                self.locs_tree.addTopLevelItem(group)
                group.setExpanded(True)
                groups[group_key] = group
            item = QtWidgets.QTreeWidgetItem([tail])
            item.setData(0, self._PATH_ROLE, path)
            item.setToolTip(0, path)
            group.addChild(item)

    def _selected_paths(self) -> list[str]:
        """Absolute paths of the selected leaf (file) items."""
        paths = []
        for item in self.locs_tree.selectedItems():
            path = item.data(0, self._PATH_ROLE)
            if path:
                paths.append(path)
        return paths

    # -- embedded picasso Render window ------------------------------------

    def _ensure_render_window(self, fresh: bool = False):
        """Create and embed the picasso Render canvas; return its window.

        Embeds picasso's ``View`` widget *itself* (not the wrapping
        ``QMainWindow``), exactly as it lives in standalone picasso Render,
        so its mouse / zoom coordinate math and the zoom rubber-band work.
        The ``Window`` is kept alive -- never shown or added to a layout --
        only because the ``View`` references it for its (exploration)
        dialogs; its processing menus are therefore never surfaced.

        Parameters
        ----------
        fresh : bool, optional
            If True, dispose any existing window and build a new (empty) one
            -- used to replace the displayed localizations with a clean
            slate. Default is False (reuse the existing window).

        Returns
        -------
        picasso Render Window, or None if the render GUI cannot be created
        (a message is then shown in the canvas area).
        """
        if self._render_window is not None and not fresh:
            return self._render_window
        try:
            from picasso.gui import render as picasso_render

            self._teardown_render_window()
            window = picasso_render.Window()
            view = window.view
            self.canvas_placeholder.setVisible(False)
            self.canvas_layout.addWidget(view)
            self._render_window = window
            self._embedded_view = view
            self._set_controls_enabled(True)
            self._populate_render_toolbar(window)
            # picasso 0.12 loads and renders on background QThreads;
            # destroying one while it runs aborts the process. The window
            # we keep alive is never shown, so it gets no closeEvent to
            # stop them -- do it ourselves when the application quits.
            self._connect_quit_hook()
        except Exception as e:
            logger.error(f"Could not embed the picasso Render GUI: {e}")
            self.canvas_placeholder.setText(
                "The picasso Render view could not be loaded:\n"
                f"{e}\n\n(Localization files can still be opened in the "
                "standalone picasso Render.)"
            )
            self.canvas_placeholder.setVisible(True)
            return None
        return self._render_window

    def _set_fallback_buttons_visible(self, visible: bool) -> None:
        """Show or hide the plain exploration buttons the native icon
        toolbar supersedes (*Display settings* / *Channels* / *Fit in
        view*); the others (*Zoom in/out* / *Clear*) have no toolbar
        equivalent and always stay."""
        for btn in (
            self.display_button,
            self.channels_button,
            self.fit_button,
        ):
            btn.setVisible(visible)

    def _populate_render_toolbar(self, window) -> None:
        """Fill the icon toolbar with picasso's native exploration actions.

        Best-effort: reuses the embedded window's own QActions (navigation,
        picking, display -- see ``_TOOLBAR_ACTIONS``) and picasso's icon
        theme, so the buttons look and behave exactly as in standalone
        Render and enable themselves once localizations are loaded; the
        equivalent plain buttons are then hidden. If the toolbar cannot be
        built -- picasso 0.11 (no ``toolbars`` helper / icon theme) or any
        failure reaching picasso's action API -- it is left hidden and the
        plain buttons are used. A toolbar is a cosmetic enhancement, so its
        failure must never break the (already embedded) view.

        Parameters
        ----------
        window : picasso Render Window
            The freshly embedded window whose menu actions to surface.
        """
        self.render_toolbar.clear()
        try:
            added = self._fill_render_toolbar(window)
        except Exception as e:  # picasso 0.11, or a changed action API
            logger.debug(f"Render toolbar unavailable: {e}")
            added = False
        if added:
            self.render_toolbar.setVisible(True)
            self._set_fallback_buttons_visible(False)
        else:
            self.render_toolbar.clear()  # drop any partial population
            self.render_toolbar.setVisible(False)
            self._set_fallback_buttons_visible(True)

    def _fill_render_toolbar(self, window) -> bool:
        """Add the allow-listed native actions to the toolbar; return
        whether any were added. Raises if picasso's icon theme / toolbar
        helper is absent or its action API cannot be reached -- the caller
        turns that into the plain-button fallback."""
        from picasso.gui import theme, toolbars

        actions = toolbars.menu_actions(window.menu_bar)
        added = False
        pending_sep = False
        for item in _TOOLBAR_ACTIONS:
            if item is None:
                pending_sep = added  # only between real items, never leading
                continue
            path, icon_name = item
            action = actions.get(path)
            if action is None:
                logger.debug(f"Render toolbar: action {path!r} not found")
                continue
            if pending_sep:
                self.render_toolbar.addSeparator()
                pending_sep = False
            action.setIcon(theme.icon(icon_name))
            self.render_toolbar.addAction(action)
            added = True
        return added

    @staticmethod
    def _stop_view_workers(view) -> None:
        """Stop a picasso ``View``'s background loader / render threads.

        In picasso 0.12 locs loading and scene rendering each run on a
        ``QThread``, and destroying a running ``QThread`` aborts the
        process; ``stop_load`` / ``stop_render_worker`` cancel and join
        those threads. On picasso 0.11, which has neither thread, the
        methods are absent and this is a no-op.
        """
        if view is None:
            return
        for name in ("stop_load", "stop_render_worker"):
            fn = getattr(view, name, None)
            if fn is None:
                continue
            try:
                fn()
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"{name} failed: {e}")

    def _stop_window_workers(self) -> None:
        """Stop the embedded window's background threads (main view and
        the rotation window's view), mirroring picasso Render's own
        ``closeEvent`` cleanup."""
        window = self._render_window
        if window is None:
            return
        self._stop_view_workers(getattr(window, "view", None))
        window_rot = getattr(window, "window_rot", None)
        if window_rot is not None:
            self._stop_view_workers(getattr(window_rot, "view_rot", None))

    def _connect_quit_hook(self) -> None:
        """Stop the render worker threads on application shutdown (once)."""
        if self._quit_hook_connected:
            return
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_window_workers)
            self._quit_hook_connected = True

    def _teardown_render_window(self) -> None:
        """Detach and dispose the currently embedded view and its window."""
        # Stop picasso 0.12's background loader / render threads before
        # the widgets are destroyed: deleting a live QThread aborts the
        # process. No-op on picasso 0.11 (synchronous, no such threads).
        self._stop_window_workers()
        # Drop the native toolbar's actions before their window is
        # destroyed (they belong to it), and restore the plain-button
        # fallback until a new window repopulates the toolbar.
        self.render_toolbar.clear()
        self.render_toolbar.setVisible(False)
        self._set_fallback_buttons_visible(True)
        if self._embedded_view is not None:
            self.canvas_layout.removeWidget(self._embedded_view)
            self._embedded_view.setParent(None)
            self._embedded_view.deleteLater()
            self._embedded_view = None
        if self._render_window is not None:
            self._render_window.deleteLater()
            self._render_window = None

    def render_selected(self, replace: bool = True) -> None:
        """Render the selected file(s) in the embedded canvas.

        Parameters
        ----------
        replace : bool, optional
            If True (default), show only the selection (a fresh canvas); if
            False, overlay the selection on what is already shown.
        """
        paths = self._selected_paths()
        if not paths:
            return
        window = self._ensure_render_window(fresh=replace)
        if window is None:
            return
        try:
            window.view.add_multiple(paths)
        except Exception as e:
            logger.error(f"Could not render localizations: {e}")

    def clear_view(self) -> None:
        """Remove all localizations (reset to an empty canvas)."""
        if self._render_window is not None:
            self._ensure_render_window(fresh=True)

    def _open_display_settings(self) -> None:
        window = self._ensure_render_window()
        if window is not None:
            window.display_settings_dlg.show()
            window.display_settings_dlg.raise_()

    def _open_channels(self) -> None:
        window = self._ensure_render_window()
        if window is not None:
            window.dataset_dialog.show()
            window.dataset_dialog.raise_()

    def _zoom_in(self) -> None:
        if self._render_window is not None:
            try:
                self._render_window.view.zoom_in()
            except Exception as e:
                logger.debug(f"zoom_in failed: {e}")

    def _zoom_out(self) -> None:
        if self._render_window is not None:
            try:
                self._render_window.view.zoom_out()
            except Exception as e:
                logger.debug(f"zoom_out failed: {e}")

    def _fit_in_view(self) -> None:
        if self._render_window is not None:
            try:
                self._render_window.view.fit_in_view()
            except Exception as e:
                logger.debug(f"fit_in_view failed: {e}")
