"""Registration feature group (Phase 2 restructure).

* ``icp`` -- correspondence-free rigid registration of a point set to a template
  (``register_to_template``: rotation sweep + Kabsch ``_rigid_fit`` + Hungarian
  ``_match_within_gate`` + ``_icp_refine``), plus the DBSCAN docking-site
  sub-clustering (``subcluster_docking_sites``) and shape descriptors
  (``_elongation``, ``_radius_of_gyration``) it builds on. Extracted from
  ``picasso_outpost.py`` by the monolith split (S7).

`promote` candidate #3 for ``picasso.registration`` -- self-contained
numpy/scipy/sklearn, no picasso coupling. See ``../MANIFEST.md``.
"""
