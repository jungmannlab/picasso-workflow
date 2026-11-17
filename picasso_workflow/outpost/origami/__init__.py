"""DNA-origami picking feature group (Phase 2 restructure).

* ``origami`` -- design-aware origami picking: ``OrigamiTemplate`` + template
  builders (grid / design-file / explicit sites), the kinetics simulation
  (``simulate_origami_nlocs_rmsd``, ``origami_phase_space_preview``), the
  structure / lattice-defect descriptors and their clustering, and the
  ``pick_origami`` orchestrator. Extracted from ``picasso_outpost.py`` by the
  monolith split (S8).

`keep` -- workflow-shaped orchestration. Builds on ``outpost/registration`` (the
correspondence-free ICP) and ``outpost/picks`` (the pick_similar family); the
geometry/registration primitives may promote later. See ``../MANIFEST.md``.
"""
