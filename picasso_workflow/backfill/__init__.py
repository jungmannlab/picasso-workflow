"""WP-7 historical-archive backfill: FLAT ``_locs.hdf5`` + ``.yaml`` → registry.

Bootstrap real cohort history from the pre-automation archive, which is *flat*
picasso output — raw movies plus picasso ``*_locs.hdf5`` localization files and
their ``.yaml`` info sidecars (NOT picasso-workflow-structured runs). This is
the ``_locs.hdf5`` half of the backfill named in
:mod:`picasso_registry.backfill_liveloc` (which handles the LiveLocalization
``*_qc.json`` half).

Two design facts drive the module layout:

* **Most quality metrics — including the A1 primaries NeNA and localization
  precision — come straight from the hdf5** (:mod:`.metrics`): photons,
  background, PSF σ, localization precision (``lpx``/``lpy``) and ``n_locs`` are
  columns; density is ``n_locs`` over the FOV area from the ``.yaml``; NeNA is
  computed with picasso's ``postprocess.nena`` *through the
  ``picasso_workflow._picasso_api`` boundary* — no raw movie needed. Kinetics /
  drift / clustering require a processing pass and are OUT of scope for this
  MVP (stubbed NULL with a note). Re-localization from the raw movie is also out
  of scope.
* **The descriptor (taxon) and the good/bad gold labels are human knowledge**,
  not in the files, so they arrive via a curated CSV (:mod:`.csv_import`):
  sample taxon rows + ``qc`` / ``feedback`` rows keyed to the ingested run.

The pieces:

* :mod:`.metrics` — compute the metric vector from a loaded ``_locs.hdf5`` +
  ``.yaml`` (primaries from the hdf5; NeNA via the boundary; the processing-pass
  metrics stay absent so they persist as NULL).
* :mod:`.parser` — folder-name / ``.yaml`` convention parser → features + a
  stable content-derived ``run_id`` and the ``metadata_complete`` gate.
* :mod:`.ingest` — per-file parse → map → ``bulk_ingest``, mirroring
  :mod:`picasso_registry.backfill_liveloc` (``_already_ingested`` idempotency
  guard, ``--dry-run``, per-file error reporting), + a ``main``/argparse CLI.
* :mod:`.csv_import` — import the curated taxon + good/bad label CSV into taxon
  / ``qc`` / ``feedback`` rows.
* :mod:`.ranges` — learn per-taxon expected-metric ranges from the populated
  history, using ONLY metadata-complete records.
"""

from __future__ import annotations

from .metrics import compute_metrics
from .parser import DatasetFeatures, parse_dataset, synth_run_id
from .ranges import compute_taxon_ranges, write_ranges_to_registry

__all__ = [
    "compute_metrics",
    "parse_dataset",
    "synth_run_id",
    "DatasetFeatures",
    "compute_taxon_ranges",
    "write_ranges_to_registry",
]
