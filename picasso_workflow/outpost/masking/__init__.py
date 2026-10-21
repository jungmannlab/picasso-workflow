"""Cell-masking feature group (Phase 2 restructure).

Holds picasso-workflow's cell-object masking code moved out of the flat
``outpost_modules/``:

* ``mask`` -- the ``CellMask`` class (binary/density masks from molecule
  positions: morphology ``filter_mask`` / ``component_masks`` / ``erode`` /
  ``dilate``, stats ``area`` / ``mean_density``, ``random_points*`` CSR
  sampling) plus the ``otsu`` threshold and ``picassolocs_to_coords`` helpers.

``CellMask`` is the cell-object layer `picasso.masking` lacks (an `extend-up`
candidate). The overlapping Otsu thresholding (also duplicated in
``spatial_stats/ripleys.py`` and ``dbscan_molint/mask.py``) is a `dedup`
against `picasso.masking` -- see ``../MANIFEST.md``.
"""
