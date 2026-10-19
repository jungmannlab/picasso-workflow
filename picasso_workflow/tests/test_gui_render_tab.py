#!/usr/bin/env python
"""Tests for the GUI "Render" tab (locs tree + embedded picasso Render)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt6", reason="PyQt6 required for the GUI tests")

from PyQt6 import QtWidgets  # noqa: E402

from picasso_workflow import gui  # noqa: E402
from picasso_workflow.render_tab import (  # noqa: E402
    RenderTab,
    enumerate_locs_files,
)


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _make_run(tmp_path, extra=("05_picasso_render/render.png",)):
    """A fake run folder with a couple of locs files and a non-locs file."""
    run = tmp_path / "AnalysisResults" / "myrun_240101-1200"
    (run / "03_picasso_localize").mkdir(parents=True)
    (run / "04_picasso_undrift_aim").mkdir(parents=True)
    (run / "WorkflowRunner.yaml").write_text("results: {}\n")
    (run / "03_picasso_localize" / "locs.hdf5").write_bytes(b"")
    (run / "04_picasso_undrift_aim" / "locs_aim.hdf5").write_bytes(b"")
    for rel in extra:
        p = run / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return str(run)


def test_enumerate_locs_files_lists_hdf5_only(tmp_path):
    run = _make_run(tmp_path)
    found = enumerate_locs_files(run)
    labels = [label for label, _ in found]
    assert labels == [
        "03_picasso_localize/locs.hdf5",
        "04_picasso_undrift_aim/locs_aim.hdf5",
    ]
    # the .png is not listed, and paths are absolute and exist
    for _, path in found:
        assert path.endswith(".hdf5") and os.path.isabs(path)


def test_enumerate_skips_assets_and_logs(tmp_path):
    run = _make_run(tmp_path)
    for junk in ("assets/x.hdf5", "logs/y.hdf5", "__pycache__/z.hdf5"):
        p = os.path.join(run, junk)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").close()
    labels = [label for label, _ in enumerate_locs_files(run)]
    assert not any(
        label.startswith(("assets", "logs", "__pycache__")) for label in labels
    )


def test_enumerate_missing_folder(tmp_path):
    assert enumerate_locs_files(str(tmp_path / "nope")) == []
    assert enumerate_locs_files("") == []


class _FakeMain:
    """Minimal stand-in for the GUI main window used by RenderTab."""

    def __init__(self, base):
        self._base = base
        self.results_folder_display = type("L", (), {"text": lambda s: base})()

    def _find_runs(self, base):
        # one run under the base
        out = []
        for root, dirs, files in os.walk(base):
            if "WorkflowRunner.yaml" in files:
                out.append(root)
        return sorted(out)


def test_render_tab_populates_tree(qapp, tmp_path):
    _make_run(tmp_path)
    base = str(tmp_path)
    tab = RenderTab(_FakeMain(base))
    tab.refresh()
    assert tab.run_combo.count() == 1
    # two groups (module folders), each with one locs file
    groups = [
        tab.locs_tree.topLevelItem(i)
        for i in range(tab.locs_tree.topLevelItemCount())
    ]
    files = [g.child(j) for g in groups for j in range(g.childCount())]
    assert len(files) == 2
    tab.deleteLater()


def test_render_tab_nests_aggregation_targets(qapp, tmp_path):
    # an aggregation run: per-target subfolders, each with module folders
    run = tmp_path / "AnalysisResults" / "agg_240101-1200"
    for tag in ("sgl_00_ch0", "sgl_01_ch1"):
        loc = run / tag / "03_picasso_localize" / "locs.hdf5"
        loc.parent.mkdir(parents=True)
        loc.write_bytes(b"")
    (run / "WorkflowRunner.yaml").write_text("results: {}\n")
    tab = RenderTab(_FakeMain(str(tmp_path)))
    tab.refresh()
    # level 1: one node per target
    targets = [
        tab.locs_tree.topLevelItem(i)
        for i in range(tab.locs_tree.topLevelItemCount())
    ]
    assert [t.text(0) for t in targets] == ["sgl_00_ch0", "sgl_01_ch1"]
    # level 2: the module folder; level 3: the locs file (a leaf w/ a path)
    module = targets[0].child(0)
    assert targets[0].childCount() == 1
    assert module.text(0) == "03_picasso_localize"
    leaf = module.child(0)
    assert module.childCount() == 1 and leaf.text(0) == "locs.hdf5"
    assert leaf.data(0, tab._PATH_ROLE).endswith("locs.hdf5")
    tab.deleteLater()


def test_render_tab_selected_paths(qapp, tmp_path):
    _make_run(tmp_path)
    tab = RenderTab(_FakeMain(str(tmp_path)))
    tab.refresh()
    g0 = tab.locs_tree.topLevelItem(0)
    g0.child(0).setSelected(True)
    paths = tab._selected_paths()
    assert len(paths) == 1 and paths[0].endswith(".hdf5")
    tab.deleteLater()


def test_main_window_has_render_tab(qapp):
    try:
        win = gui.Window()
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Could not construct GUI window: {e}")
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert "Render" in titles
    win.close()


def _make_real_locs_run(tmp_path):
    """A run folder holding one real (picasso-readable) locs file."""
    picasso_io = pytest.importorskip("picasso.io")
    run = tmp_path / "myrun_240101-1200"
    (run / "03_picasso_localize").mkdir(parents=True)
    (run / "WorkflowRunner.yaml").write_text("results: {}\n")
    locs = pd.DataFrame(
        {
            "frame": np.arange(40, dtype="u4"),
            "x": (np.random.rand(40) * 30).astype("f4"),
            "y": (np.random.rand(40) * 30).astype("f4"),
            "photons": (np.random.rand(40) * 1000).astype("f4"),
            "sx": (np.random.rand(40)).astype("f4"),
            "sy": (np.random.rand(40)).astype("f4"),
            "bg": (np.random.rand(40) * 50).astype("f4"),
            "lpx": (np.random.rand(40) / 10).astype("f4"),
            "lpy": (np.random.rand(40) / 10).astype("f4"),
        }
    )
    info = [{"Width": 32, "Height": 32, "Frames": 40, "Pixelsize": 130}]
    picasso_io.save_locs(
        str(run / "03_picasso_localize" / "locs.hdf5"), locs, info
    )
    return run


def test_embedded_render_loads_real_locs(qapp, tmp_path):
    """Smoke test: the embedded picasso Render canvas loads a real locs
    file, with the processing menu bar hidden."""
    _make_real_locs_run(tmp_path)

    tab = RenderTab(_FakeMain(str(tmp_path)))
    tab.refresh()
    tab.locs_tree.topLevelItem(0).child(0).setSelected(True)
    # picasso's render-GUI import trips a scipy DeprecationWarning that the
    # suite otherwise turns into an error (picasso's own tech debt); allow it
    # here so the embedding can be exercised.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            tab.render_selected(replace=True)
        except Exception as e:  # pragma: no cover
            pytest.skip(f"picasso Render embedding unavailable: {e}")
    if tab._render_window is None:
        pytest.skip("picasso Render window could not be embedded")
    # let the background load thread finish
    for _ in range(50):
        qapp.processEvents()
        if len(getattr(tab._render_window.view, "locs", [])) >= 1:
            break
        import time

        time.sleep(0.1)
    assert len(tab._render_window.view.locs) == 1
    # the View widget itself is embedded (not the wrapping QMainWindow), so
    # its coordinate math / zoom rubber-band behave as in standalone picasso
    assert tab._embedded_view is tab._render_window.view
    assert tab.canvas_layout.indexOf(tab._embedded_view) >= 0

    # zoom controls operate on the embedded view: zoom in shrinks the
    # viewport (shows a smaller region), fit resets it
    def _area(vp):
        return (vp[1][0] - vp[0][0]) * (vp[1][1] - vp[0][1])

    view = tab._render_window.view
    assert tab.zoom_in_button.isEnabled()
    before = _area(view.viewport)
    tab._zoom_in()
    qapp.processEvents()
    assert _area(view.viewport) < before
    tab._fit_in_view()
    qapp.processEvents()
    # stop picasso 0.12's background threads before the window leaks into
    # interpreter teardown (destroying a running QThread aborts the process)
    tab._teardown_render_window()
    tab.deleteLater()


def test_teardown_stops_render_worker_threads(qapp, tmp_path):
    """Replacing the view (or clearing it) must stop picasso 0.12's
    background loader / render threads before the view is destroyed --
    destroying a live QThread aborts the process. On picasso 0.11 (no
    such threads) teardown must still succeed."""
    _make_real_locs_run(tmp_path)

    tab = RenderTab(_FakeMain(str(tmp_path)))
    tab.refresh()
    tab.locs_tree.topLevelItem(0).child(0).setSelected(True)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            tab.render_selected(replace=True)
        except Exception as e:  # pragma: no cover
            pytest.skip(f"picasso Render embedding unavailable: {e}")
    if tab._render_window is None:
        pytest.skip("picasso Render window could not be embedded")

    view = tab._render_window.view
    # spy on whichever thread-stopping methods this picasso exposes
    # (0.11 has stop_load only; 0.12 adds stop_render_worker)
    calls = []
    present = [
        name
        for name in ("stop_load", "stop_render_worker")
        if hasattr(view, name)
    ]
    for name in present:
        orig = getattr(view, name)

        def _spy(*a, _n=name, _o=orig, **k):
            calls.append(_n)
            return _o(*a, **k)

        setattr(view, name, _spy)

    # a replace-render tears the old window down; it must not raise
    tab.locs_tree.topLevelItem(0).child(0).setSelected(True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        tab.render_selected(replace=True)
    qapp.processEvents()

    # every stop method this picasso exposes was invoked before the old
    # view was destroyed (prevents the running-QThread process abort)
    for name in present:
        assert name in calls
    # teardown produced a fresh window regardless of picasso version
    assert tab._render_window is not None
    # dispose cleanly: no running QThread may outlive the window
    tab._teardown_render_window()
    assert tab._render_window is None
    tab.deleteLater()


def _render_a_file(qapp, tmp_path):
    """Embed a window showing one real locs file; return the tab or skip."""
    _make_real_locs_run(tmp_path)
    tab = RenderTab(_FakeMain(str(tmp_path)))
    tab.refresh()
    tab.locs_tree.topLevelItem(0).child(0).setSelected(True)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            tab.render_selected(replace=True)
        except Exception as e:  # pragma: no cover
            pytest.skip(f"picasso Render embedding unavailable: {e}")
    if tab._render_window is None:
        pytest.skip("picasso Render window could not be embedded")
    for _ in range(50):
        qapp.processEvents()
        if len(getattr(tab._render_window.view, "locs", [])) >= 1:
            break
        import time

        time.sleep(0.1)
    return tab


def test_render_toolbar_surfaces_native_exploration_actions(qapp, tmp_path):
    """On picasso 0.12 the icon toolbar shows picasso's own navigation /
    picking / display actions (with icons) and hides the equivalent plain
    buttons; on picasso 0.11 it stays hidden and the buttons remain."""
    try:
        from picasso.gui import toolbars  # noqa: F401

        native = True
    except Exception:
        native = False

    tab = _render_a_file(qapp, tmp_path)

    if not native:  # picasso 0.11: plain-button fallback, no toolbar
        assert tab.render_toolbar.isHidden()
        assert not tab.render_toolbar.actions()
        assert not tab.display_button.isHidden()
        assert not tab.fit_button.isHidden()
        tab._teardown_render_window()
        tab.deleteLater()
        return

    # picasso 0.12: the toolbar carries the exploration actions
    labels = [
        a.text() for a in tab.render_toolbar.actions() if not a.isSeparator()
    ]
    for expected in ("Zoom", "Pick", "Measure", "Move"):
        assert expected in labels
    # navigation + picking + display, never file I/O or heavy processing
    assert not any(
        "Open" in label or "Save" in label or "similar" in label
        for label in labels
    )
    # icons are real (not null) so the buttons show as icons
    assert all(
        not a.icon().isNull()
        for a in tab.render_toolbar.actions()
        if not a.isSeparator()
    )
    # the native toolbar supersedes the equivalent plain buttons
    assert not tab.render_toolbar.isHidden()
    assert tab.display_button.isHidden()
    assert tab.channels_button.isHidden()
    assert tab.fit_button.isHidden()
    # the Pick tool drives picasso's native View mode
    pick = next(a for a in tab.render_toolbar.actions() if a.text() == "Pick")
    pick.trigger()
    qapp.processEvents()
    assert tab._render_window.view._mode == "Pick"

    # teardown drops the native actions and restores the plain buttons
    tab._teardown_render_window()
    assert tab.render_toolbar.isHidden()
    assert not tab.render_toolbar.actions()
    assert not tab.display_button.isHidden()
    assert not tab.fit_button.isHidden()
    tab.deleteLater()


def test_toolbar_build_failure_does_not_break_embed(qapp, tmp_path):
    """A failure building the icon toolbar (e.g. picasso changes its
    action API) must not break the embedded view or skip the worker-thread
    quit hook -- it falls back silently to the plain buttons."""
    _make_real_locs_run(tmp_path)
    tab = RenderTab(_FakeMain(str(tmp_path)))
    tab.refresh()
    tab.locs_tree.topLevelItem(0).child(0).setSelected(True)

    def boom(_window):
        raise RuntimeError("picasso toolbar API changed")

    # simulate picasso's action API raising mid-population
    tab._fill_render_toolbar = boom

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            tab.render_selected(replace=True)
        except Exception as e:  # pragma: no cover
            pytest.skip(f"picasso Render embedding unavailable: {e}")
    if tab._render_window is None:
        pytest.skip("picasso Render window could not be embedded")

    # the view embedded despite the toolbar failure ...
    assert tab._embedded_view is tab._render_window.view
    assert tab.canvas_layout.indexOf(tab._embedded_view) >= 0
    # ... the quit hook that stops the background threads was still wired ...
    assert tab._quit_hook_connected
    # ... and it fell back to the plain buttons with no toolbar shown
    assert tab.render_toolbar.isHidden()
    assert not tab.render_toolbar.actions()
    assert not tab.display_button.isHidden()
    assert not tab.fit_button.isHidden()
    tab._teardown_render_window()
    tab.deleteLater()
