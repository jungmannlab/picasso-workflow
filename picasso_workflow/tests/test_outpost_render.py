#!/usr/bin/env python
"""The render glue is importable at its new ``render`` location.

Regression guard for the Phase 2 move of ``render`` out of the flat
``outpost_modules/`` into ``outpost/render/``. ``analyse.py`` imports it as
``render`` and calls ``render.plot_scene`` at many sites; the unit tier mocks
``AutoPicasso``, so without this guard a broken move would surface only at
runtime.
"""


def test_render_importable_at_new_location():
    """The functions ``analyse`` calls resolve at the new path."""
    from picasso_workflow.outpost.render import render

    for name in ("plot_scene", "render_scene", "render_single_channel"):
        assert callable(getattr(render, name)), name


def test_analyse_imports_render_from_new_path():
    """``analyse.render`` is the moved module (its import resolved)."""
    import picasso_workflow.analyse as analyse
    from picasso_workflow.outpost.render import render

    assert analyse.render is render
