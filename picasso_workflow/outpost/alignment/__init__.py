"""Channel-alignment feature group (Phase 2 restructure).

Extracted from ``picasso_outpost.py`` by the monolith split (S5):

* ``shift_kernel`` -- the point-set pairwise-shift estimation machinery
  (``_calculate_pairwise_shift`` + peak-finding / Gaussian-fit / adaptive-bin /
  quality-metric helpers) and its shift-histogram plot savers. The *numeric
  core* is `promote` candidate #2 (``picasso.imageprocess``) -- but it still has
  plot-saving mixed in (``_calculate_pairwise_shift`` calls the savers), so the
  plots-stripping refactor (MANIFEST row 2, readiness M) must precede upstreaming.
* ``channels`` -- multi-channel alignment orchestration (``align_channels``,
  ``align_by_rcc`` / ``align_by_picked`` / ``align_by_rsso``, ``shift_from_rcc``,
  ``plot_shift``) + the global optimal-shift solve (``_solve_optimal_shifts``).
  `keep` -- workflow glue over the shift kernel + upstream ``imageprocess.rcc`` /
  ``lib.minimize_shifts``.
"""
