"""Clustering feature group (Phase 2 restructure).

* ``molint`` -- DBSCAN-based molecular-interaction (molint) analysis: barcode /
  interaction-graph / degree-of-clustering computation and plots, extracted from
  ``picasso_outpost.py`` by the monolith split (S4).

* ``dbscan_molint`` -- the molint backend subpackage (``dbscan`` / ``mask`` /
  ``output_metrics`` / ``io``), folded in from the top-level
  ``picasso_workflow/dbscan_molint/`` by the monolith-split follow-up (S4b).
  ``molint`` and ``analyse`` reach it via deferred imports.

`keep` -- coupled to the molint Confluence report format.
"""
