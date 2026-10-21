"""SPINNA feature group (Phase 2 restructure).

Workflow-side wrappers over ``picasso.spinna`` + plotting/plumbing, extracted
from ``picasso_outpost.py`` by the monolith split (S3):

* ``wrappers`` -- ``spinna_batch``, ``single_spinna_run``,
  ``single_spinna_fit_le_run``, ``screen_label_uncertainty``,
  ``plot_spinna_nnd``, ``load_structures_from_dict`` and helpers.

`keep` -- thin orchestration over the upstream ``picasso.spinna`` routines (and
the guarded private ``_spinna_batch_analysis`` / ``_targets_from_structures``
accessors via ``_picasso_api``), not an upstream candidate itself.
"""
