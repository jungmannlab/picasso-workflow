"""Resolution-estimation feature group (Phase 2 restructure).

Holds picasso-workflow's resolution estimators moved out of the flat
``outpost_modules/``:

* ``resolution_frc``          -- Fourier Ring Correlation.
* ``resolution_decorrelation`` -- image-decorrelation resolution (reuses
  ``resolution_frc.render_image_histogram``).

The third estimator (point-pattern autocorrelation, ``resolution_ppac*``) still
lives in ``picasso_outpost.py`` and will move here when that monolith is split.
All three are `promote` candidates -- see ``../MANIFEST.md``.
"""
