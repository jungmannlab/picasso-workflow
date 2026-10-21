"""Spatial point-pattern statistics feature group (Phase 2 restructure).

Holds picasso-workflow's Ripley's-K / CSR-envelope code moved out of the flat
``outpost_modules/``:

* ``ripleys``                 -- Ripley's K / RDF / FRC core (KDTree-based),
  CSR and randomized controls, plotting helpers (Rafal Kowalewski).
* ``ripley_dcatlas_analysis`` -- the reproducible DC-Atlas whole-cell Ripley
  pipeline built on top of ``ripleys``.

The analytic CSR k-NN statistics (``nndistribution_from_csr`` and friends,
`promote` candidate #1) still live in ``picasso_outpost.py`` and will move here
when that monolith is split. The older, separately-wired ``ripleys_analysis``
subpackage (``run_ripleysAnalysis``) is a distinct implementation and is folded
in by a later slice -- see ``../MANIFEST.md``.
"""
