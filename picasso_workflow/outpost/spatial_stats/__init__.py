"""Spatial point-pattern statistics feature group (Phase 2 restructure).

Holds picasso-workflow's Ripley's-K / CSR-envelope code moved out of the flat
``outpost_modules/``:

* ``ripleys``                 -- Ripley's K / RDF / FRC core (KDTree-based),
  CSR and randomized controls, plotting helpers (Rafal Kowalewski).
* ``ripley_dcatlas_analysis`` -- the reproducible DC-Atlas whole-cell Ripley
  pipeline built on top of ``ripleys``.

The analytic CSR k-NN statistics (``nndistribution_from_csr`` and friends,
`promote` candidate #1) still live in ``picasso_outpost.py`` and will move here
when that monolith is split. (The older, separately-wired ``ripleys_analysis``
subpackage that backed the legacy ``ripleysk`` / ``ripleysk_average`` modules
was retired -- those were superseded by ``ripleysk2`` / ``ripleysk_average2``,
which use the ``ripleys`` module here.) See ``../MANIFEST.md``.
"""
