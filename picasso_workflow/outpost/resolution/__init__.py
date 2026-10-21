"""Resolution-estimation feature group (Phase 2 restructure).

Holds picasso-workflow's resolution estimators moved out of the flat
``outpost_modules/``:

* ``resolution_frc``          -- Fourier Ring Correlation.
* ``resolution_decorrelation`` -- image-decorrelation resolution (reuses
  ``resolution_frc.render_image_histogram``).
* ``resolution_ppac``         -- point-pattern autocorrelation (the third
  estimator), moved here from ``picasso_outpost.py`` by the monolith split.

FRC and decorrelation are `promote` candidates. PPAC is **not** promote-ready --
it never performed well (an O(G**2) KDTree-per-grid-cell algorithm); see its
module docstring and ``../MANIFEST.md`` for the fix (a displacement-vector
histogram) to apply before promoting.
"""
