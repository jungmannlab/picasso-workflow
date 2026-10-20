"""Unified home for picasso-workflow's picasso-adjacent ("outpost") code.

This package is the Phase 2 migration target: the 7.6k-line
``picasso_workflow/picasso_outpost.py`` monolith and the flat
``picasso_workflow/outpost_modules/`` grab-bag are being reorganised into
feature-focused submodules here (resolution, drift, masking, spatial_stats,
picks, origami, clustering, spinna, binding_events, render).

See ``README.md`` for the target layout and ``MANIFEST.md`` for the per-feature
migration status (keep / dedup / promote / extend-up / delete). Code is moved
in incrementally; this package starts empty on purpose.
"""
