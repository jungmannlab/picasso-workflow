#!/usr/bin/env python
"""Tests for the picasso Render ``View.viewport`` early-resize guard.

picasso's ``View`` assigns ``self.viewport`` only on its first draw, but locs
load on a background thread; a ``resizeEvent`` in between calls
``update_scene`` / ``update_scene_slicer`` and raises ``AttributeError`` on the
missing ``viewport``. ``_patch_render_view_viewport`` wraps those methods to
seed the full field of view first. These tests use a fake ``View`` that
reproduces the bug, so no real picasso import is needed.
"""

import types

import pytest

from picasso_workflow.render_tab import _patch_render_view_viewport

FULL_FOV = [(0, 0), (100.0, 200.0)]


def _make_render_module():
    """A fresh fake ``picasso.gui.render`` with a buggy ``View`` class.

    A new class each call so patching one test's module never leaks into
    another (the patch mutates the class in place).
    """

    class View:
        def __init__(self):
            self.n_channels = 0
            self.drawn_with = "unset"
            # deliberately does NOT set self.viewport -> reproduces the bug

        def full_fov_viewport(self):
            return FULL_FOV

        def update_scene(self, viewport=None):
            if self.n_channels:
                viewport = viewport or self.viewport
                self.drawn_with = viewport

        def update_scene_slicer(self, viewport=None):
            if self.n_channels:
                viewport = viewport or self.viewport
                self.drawn_with = viewport

    return types.SimpleNamespace(View=View)


def test_unpatched_view_raises_on_early_resize():
    """Documents the upstream bug the patch fixes."""
    mod = _make_render_module()
    view = mod.View()
    view.n_channels = 1  # locs loaded, but no draw yet -> no self.viewport
    with pytest.raises(AttributeError):
        view.update_scene()


def test_patch_seeds_full_fov_when_viewport_unset():
    mod = _make_render_module()
    _patch_render_view_viewport(mod)
    view = mod.View()
    view.n_channels = 1

    view.update_scene()  # no raise
    assert view.viewport == FULL_FOV
    assert view.drawn_with == FULL_FOV

    view2 = mod.View()
    view2.n_channels = 1
    view2.update_scene_slicer()
    assert view2.drawn_with == FULL_FOV


def test_patch_respects_explicit_viewport():
    mod = _make_render_module()
    _patch_render_view_viewport(mod)
    view = mod.View()
    view.n_channels = 1
    explicit = [(1, 2), (3, 4)]

    view.update_scene(explicit)
    assert view.drawn_with == explicit  # the passed viewport wins


def test_patch_noop_before_locs_loaded():
    mod = _make_render_module()
    _patch_render_view_viewport(mod)
    view = mod.View()  # n_channels == 0

    view.update_scene()  # no raise, nothing drawn
    assert view.drawn_with == "unset"
    # class-level default avoids AttributeError on a stray read
    assert view.viewport is None


def test_patch_is_idempotent():
    mod = _make_render_module()
    _patch_render_view_viewport(mod)
    _patch_render_view_viewport(mod)  # second call must not double-wrap
    assert mod.View._pwf_viewport_guarded is True
    view = mod.View()
    view.n_channels = 1
    view.update_scene()
    assert view.drawn_with == FULL_FOV


def test_patch_tolerates_missing_view():
    mod = types.SimpleNamespace()  # no View attribute
    _patch_render_view_viewport(mod)  # must not raise
