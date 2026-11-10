"""Registration feature group (Phase 2 restructure).

* ``icp`` -- correspondence-free rigid registration of a point set to a
  template. ``register_to_template`` is now a thin adapter over
  ``picasso.registration.register_to_template`` (picassosr >= 1.0, reached
  through ``_picasso_api``), keeping the origami dict contract; plus the
  DBSCAN docking-site sub-clustering (``subcluster_docking_sites``) and
  shape descriptors (``_elongation``, ``_radius_of_gyration``) it builds on.

The forked algorithm was promoted upstream as
``picasso.registration.register_to_template``. See ``../MANIFEST.md``.
"""
