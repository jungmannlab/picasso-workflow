"""Pick / fiducial processing feature group (Phase 2 restructure).

* ``picks`` -- pick-finding (``pick_gold``, ``picked_locs``, ``pick_similar`` +
  the numba-jitted ``pick_similar_analysis`` path), fiducial drift
  (``shift_from_picked``, ``sort_picked_locs``, ``_undrift_from_picked``), and
  helpers, extracted from ``picasso_outpost.py`` by the monolith split (S6).

Mostly `keep`; ``picked_locs`` is deduped onto ``picasso.postprocess.picked_locs``
and ``_undrift_from_picked`` / ``_solve_optimal_shifts`` are dedup-maybes -- see
``../MANIFEST.md``.
"""
