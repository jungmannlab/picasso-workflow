"""Channel-alignment feature group (Phase 2 restructure).

Extracted from ``picasso_outpost.py`` by the monolith split (S5):

* ``channels`` -- multi-channel alignment orchestration (``align_channels``,
  ``align_by_rcc`` / ``align_by_picked`` / ``_align_by_points``,
  ``shift_from_rcc``, ``plot_shift``). `keep` -- workflow glue over upstream
  ``postprocess.align`` (``method="points"`` replaces the retired forked RSSO
  cross-correlation) and ``imageprocess.rcc``.
"""
