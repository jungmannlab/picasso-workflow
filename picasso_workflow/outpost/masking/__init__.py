"""Cell-masking feature group (Phase 2 restructure).

Holds picasso-workflow's cell-object masking code moved out of the flat
``outpost_modules/``:

* ``mask`` -- the ``CellMask`` class (binary/density masks from molecule
  positions: morphology ``filter_mask`` / ``component_masks`` / ``erode`` /
  ``dilate``, stats ``area`` / ``mean_density``, ``random_points*`` CSR
  sampling) plus the ``picassolocs_to_coords`` helper.

The morphology primitives (``filter_mask`` / ``component_masks`` / ``erode`` /
``dilate``) now run on `picasso.masking` (``label_components`` / ``fill_holes``
/ ``erode`` / ``dilate``, via the boundary), while ``CellMask`` remains the
cell-object layer `picasso.masking` lacks (an `extend-up` candidate). The
private Otsu-threshold fork it used to carry (also duplicated
in ``spatial_stats/ripleys.py`` and ``ripley_dcatlas_analysis.py``) has been
**deduped** onto `picasso.masking.threshold_otsu` -- see ``../MANIFEST.md``.
"""
