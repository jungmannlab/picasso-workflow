"""Clustering feature group (Phase 2 restructure).

* ``molint`` -- DBSCAN-based molecular-interaction (molint) analysis: barcode /
  interaction-graph / degree-of-clustering computation and plots, extracted from
  ``picasso_outpost.py`` by the monolith split (S4).

`keep` -- coupled to the molint Confluence report format. Uses the
``dbscan_molint/`` subpackage (``dbscan`` / ``output_metrics``) via deferred
imports; folding that subpackage in here is a later step (S4b).
"""
