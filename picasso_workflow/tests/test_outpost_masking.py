#!/usr/bin/env python
"""The cell-masking feature is importable at its new ``masking`` location.

Regression guard for the Phase 2 move of ``mask`` out of the flat
``outpost_modules/`` into ``outpost/masking/``. ``analyse.py`` imports
``CellMask`` from the new path at module scope; the unit tier mocks
``AutoPicasso``, so without this guard a broken move would surface only at
runtime.
"""


def test_cellmask_importable_at_new_location():
    """The symbols ``analyse`` / ``test_mask`` use resolve at the new path."""
    from picasso_workflow.outpost.masking import mask
    from picasso_workflow.outpost.masking.mask import CellMask

    assert mask.CellMask is CellMask
    assert callable(mask.picassolocs_to_coords)


def test_analyse_imports_cellmask_from_new_path():
    """``analyse`` exposes ``CellMask`` (its top-level import resolved)."""
    import picasso_workflow.analyse as analyse
    from picasso_workflow.outpost.masking.mask import CellMask

    assert analyse.CellMask is CellMask


def test_mask_not_left_in_outpost_modules():
    """The old flat ``outpost_modules.mask`` attribute is gone."""
    import picasso_workflow.outpost_modules as outpost_modules

    assert not hasattr(outpost_modules, "mask")
