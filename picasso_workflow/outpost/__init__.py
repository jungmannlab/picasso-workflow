"""Unified home for picasso-workflow's picasso-adjacent ("outpost") code.

This package is the Phase 2 migration target: the 7.6k-line
``picasso_workflow/picasso_outpost.py`` monolith and the flat
``picasso_workflow/outpost_modules/`` grab-bag have been reorganised into
feature-focused submodules here (resolution, alignment, masking,
spatial_stats, picks, registration, origami, clustering, spinna, spots,
formats, binding_events, render, reporting). The monolith split is complete --
``picasso_outpost.py`` is gone, its last two report helpers now live in
``reporting/histograms.py``.

See ``README.md`` for the layout and ``MANIFEST.md`` for the per-feature
migration status (keep / dedup / promote / extend-up / delete).
"""
