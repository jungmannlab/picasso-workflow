"""Ingest FLAT archive ``_locs.hdf5`` datasets into the registry via ``/bulk``.

Per-file: load the ``_locs.hdf5`` + ``.yaml`` (through the
``picasso_workflow._picasso_api`` boundary), parse features + a stable
content-derived ``run_id`` (:mod:`.parser`), compute the metric vector
(:mod:`.metrics` — primaries from the hdf5, NeNA via the boundary), map onto the
registry ``BulkIngest`` tables, and POST one dataset per transaction.

Mirrors :mod:`picasso_registry.backfill_liveloc`: an ``_already_ingested``
idempotency guard (no re-POST storm, 404 ⇒ not yet, any other error aborts the
sweep loudly), ``--dry-run``, and per-file error reporting that never aborts the
sweep on a single bad file.

Provenance (deliberate, stable):

* Rows land as ``analysis_run.kind = "backfill-locs"`` with
  ``recommender_version`` carrying the backfill tool tag, so hdf5-recomputed
  metrics are distinguishable from LiveLocalization-computed (``liveloc-qc``)
  and pipeline-computed metrics on the same run.
* ``--data-source`` stamps the A15/C24 acquired-vs-simulated flag (operator
  knowledge, append-only, unretrofittable).
* **No taxonomy guessing.** ``sample_taxon_id`` is left NULL; the whole feature
  bundle rides in ``experiment.extra`` / ``acquisition_run.extra`` so the
  curated-CSV descriptor mapping (:mod:`.csv_import`) attaches later without
  re-ingesting.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Callable

from picasso_workflow._picasso_api import io

from .metrics import compute_metrics, locs_signature
from .parser import DatasetFeatures, parse_dataset, synth_run_id

logger = logging.getLogger(__name__)

#: Client-side service URL / bearer token env vars (same names as
#: backfill_liveloc, so one deployment configures both backfills).
SERVICE_URL_ENV = "PAINT_REGISTRY_SERVICE_URL"
TOKEN_ENV = "PAINT_REGISTRY_TOKEN"

ANALYSIS_KIND = "backfill-locs"
BACKFILL_TAG = "wp7-backfill-locs"


class SkipFile(Exception):
    """This dataset cannot be ingested (reason in str); skip it."""


def _compact(row: dict) -> dict:
    """Drop None-valued keys: the service folds unknown keys into ``extra``;
    always-present nulls would pad rows and erase the key-presence signal."""
    return {k: v for k, v in row.items() if v is not None}


def map_dataset(
    feats: DatasetFeatures,
    metrics: dict[str, Any],
    *,
    data_source: str | None = None,
    sample_taxon_id: str | None = None,
    label_rows: dict[str, list[dict]] | None = None,
) -> dict[str, list[dict]]:
    """Map one parsed+measured dataset onto the registry ``/bulk`` tables.

    Pure function (no IO): takes the parsed :class:`DatasetFeatures` and the
    computed metric vector, returns the FK-ordered table dict for
    ``client.bulk_ingest(**tables)``. The metric keys are already registry
    ``Metrics`` column names; the ``analysis_run`` join + scope are added here.

    ``sample_taxon_id`` (when the curated CSV supplies one for this run) is set
    on the experiment **at creation** — the store is insert-only, so this is
    the only point the taxon can be attached for the ``/cohort`` join. The
    taxon node itself is created by the caller first (``csv_import.ensure_taxon``
    / a ``sample_taxonomy`` row in the batch). ``label_rows`` folds the run's
    ``qc`` / ``feedback`` verdict rows into the same transaction.
    """
    ids = feats.ids
    label_rows = label_rows or {}

    experiment = {
        "id": ids["exp"],
        "created_at": feats.created_at,
        "operator": feats.operator,
        # Taxon attached at creation when the CSV knows it (insert-only store);
        # otherwise NULL and attachable only by re-ingesting with --labels.
        "sample_taxon_id": sample_taxon_id,
        # Keep the file-derived feature bundle verbatim for later curation.
        "extra": {
            "backfill_source": "flat-locs-archive",
            "folder_tokens": feats.folder_tokens,
            "metadata_complete": feats.metadata_complete,
            "metadata_missing": feats.metadata_missing,
        },
    }

    fov = {
        "id": ids["fov"],
        "acquisition_run_id": ids["acq"],
        "exposure_ms": feats.exposure_ms,
        "frame_count": feats.frames,
        "extra": _compact(
            {"width_px": feats.width_px, "height_px": feats.height_px}
        )
        or None,
    }

    acquisition_run = {
        "id": ids["acq"],
        "experiment_id": ids["exp"],
        "microscope_id": feats.microscope,
        "started_at": feats.created_at,
        "status": "backfilled",
        "data_source": data_source,
        "raw_data_path": str(Path(feats.locs_path).parent),
        "extra": {
            "locs_path": feats.locs_path,
            "yaml_path": feats.yaml_path,
            "pixelsize_nm": feats.pixelsize_nm,
            "info": feats.raw_info,
        },
    }

    analysis_run = {
        "id": ids["ana"],
        "acquisition_run_id": ids["acq"],
        "fov_id": ids["fov"],
        "kind": ANALYSIS_KIND,
        "attempt": 1,
        "status": "done",
        "finished_at": feats.created_at,
        "recommender_version": BACKFILL_TAG,
    }

    metrics_row = {
        "analysis_run_id": ids["ana"],
        "scope": "single",
        **metrics,
    }

    artifact = {
        "analysis_run_id": ids["ana"],
        "kind": "locs-hdf5",
        "uri": feats.locs_path,
    }

    tables: dict[str, list[dict]] = {
        "experiment": [_compact(experiment)],
        "acquisition_run": [_compact(acquisition_run)],
        "fov": [_compact(fov)],
        "analysis_run": [_compact(analysis_run)],
        "metrics": [metrics_row],
        "artifact": [_compact(artifact)],
    }
    # Fold in the run's good/bad qc + feedback rows (same transaction).
    for table, rows in label_rows.items():
        if rows:
            tables[table] = rows
    return tables


def _already_ingested(client, run_id: str) -> bool:
    """True if the acquisition_run exists, False on a definite 404; any other
    failure raises — a broken deployment aborts the sweep loudly rather than
    reading as 'not yet ingested' (re-POST storm) or a bogus dry-run preview.
    (Identical contract to backfill_liveloc._already_ingested.)"""
    try:
        client.get("acquisition_run", run_id)
        return True
    except Exception as exc:
        response = getattr(exc, "response", None)
        if getattr(response, "status_code", None) == 404:
            return False
        raise


def iter_locs_files(paths: list[str]):
    """Yield ``*_locs.hdf5`` paths from files and (recursively) directories.

    Directory scans match ``*_locs.hdf5`` (the flat-archive naming); explicitly
    listed ``.hdf5`` files are taken as given. A non-existent path is a usage
    error (raised here) so a typo'd mount never reads as an empty sweep.
    """
    for entry in paths:
        p = Path(entry)
        if not p.exists():
            raise FileNotFoundError(entry)
        if p.is_dir():
            yield from sorted(set(p.rglob("*_locs.hdf5")))
        else:
            yield p


def load_locs_lenient(locs_path):
    """Load a localization file, tolerating pre-1.0 Pixelsize-less files.

    picasso 1.0 needs a pixel size to read a legacy (camera-pixel) file and
    raises :class:`picasso.io.MissingPixelsizeError` without one. Rather than
    invent a scale, load with a neutral pixel size so the localizations parse
    and strip ``Pixelsize`` from the metadata, so the dataset is flagged
    metadata-incomplete and its nm-scaled metrics are dropped downstream (see
    :mod:`picasso_workflow.backfill.metrics` / :func:`map_dataset`). Any other
    load failure propagates.
    """
    try:
        return io.load_locs(str(locs_path))
    except io.MissingPixelsizeError:
        locs, info = io.load_locs(str(locs_path), pixelsize=1.0)
        for entry in info:
            if isinstance(entry, dict):
                entry.pop("Pixelsize", None)
        return locs, info


def ingest_dataset(
    client,
    locs_path: str,
    *,
    data_source: str | None = None,
    with_nena: bool = True,
    labels: dict[str, dict[str, Any]] | None = None,
    taxon_cache: dict[str, str] | None = None,
    dry_run: bool = False,
) -> tuple[str, dict[str, list[dict]]]:
    """Load + parse + measure + map one dataset; returns ``(run_id, tables)``.

    Raises :class:`SkipFile` if the locs/yaml can't be loaded. The ``run_id`` is
    content-aware (a :func:`metrics.locs_signature` of the loaded localizations
    rides into the identity), so two distinct datasets sharing a filename +
    frame count get distinct ids. When ``labels`` (a ``{run_id: row}`` index
    from the curated CSV) has an entry for this dataset, its taxon is attached
    to the experiment at creation and its good/bad qc + feedback rows are folded
    into the same ``/bulk`` transaction — the only append-only-safe way to set
    ``experiment.sample_taxon_id``. ``taxon_cache`` dedupes taxon-node creation
    across a sweep. Under ``dry_run`` NO registry write happens here (the taxon
    node is NOT created) — the store is append-only, so a preview must not
    mutate it.
    """
    try:
        locs, info = load_locs_lenient(locs_path)
    except Exception as exc:
        raise SkipFile(f"cannot load locs/yaml ({exc})") from exc
    content_sig = locs_signature(locs)
    feats = parse_dataset(str(locs_path), info, content_sig=content_sig)
    metrics = compute_metrics(locs, info, with_nena=with_nena)

    sample_taxon_id = None
    label_rows: dict[str, list[dict]] = {}
    row = (labels or {}).get(feats.run_id)
    if row:
        # ensure_taxon POSTs a taxon node — a registry write, so it must NOT
        # run under --dry-run (the append-only store can't be un-mutated).
        if row.get("taxon") and client is not None and not dry_run:
            # Import here to avoid a hard import cycle at module load.
            from .csv_import import ensure_taxon

            sample_taxon_id = ensure_taxon(
                client, row["taxon"], row.get("taxon_path"), cache=taxon_cache
            )
        if row.get("passed") is not None:
            from .csv_import import label_bulk_rows

            label_rows = label_bulk_rows(
                row,
                analysis_run_id=feats.ids["ana"],
                acquisition_run_id=feats.run_id,
            )

    tables = map_dataset(
        feats,
        metrics,
        data_source=data_source,
        sample_taxon_id=sample_taxon_id,
        label_rows=label_rows,
    )
    return feats.run_id, tables


def ingest_paths(
    client,
    paths: list[str],
    *,
    dry_run: bool = False,
    data_source: str | None = None,
    with_nena: bool = True,
    labels: dict[str, dict[str, Any]] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Ingest every ``*_locs.hdf5`` under ``paths``; returns counts.

    One dataset = one ``/bulk`` transaction. A dataset that fails to
    load/parse/POST is reported and skipped (backfill is append-only and
    re-runnable); an unreachable/broken registry aborts loudly (via
    ``_already_ingested``). Idempotent: a dataset whose acquisition_run already
    exists is skipped, and a duplicate copy (same content-derived run_id) within
    one sweep is a quiet skip. ``labels`` (a ``{run_id: row}`` CSV index) sets
    each matched run's taxon + qc/feedback at creation.
    """
    counts = {"ingested": 0, "skipped": 0, "failed": 0}
    seen: set[str] = set()
    taxon_cache: dict[str, str] = {}
    for path in iter_locs_files(paths):
        try:
            run_id, tables = ingest_dataset(
                client,
                str(path),
                data_source=data_source,
                with_nena=with_nena,
                labels=labels,
                taxon_cache=taxon_cache,
                dry_run=dry_run,
            )
        except SkipFile as exc:
            log(f"SKIP  {path}: {exc}")
            counts["skipped"] += 1
            continue
        except Exception as exc:  # unexpected parse/compute error
            log(f"FAIL  {path}: {exc}")
            counts["failed"] += 1
            continue
        if run_id in seen:
            log(f"SKIP  {path}: duplicate copy (run {run_id})")
            counts["skipped"] += 1
            continue
        seen.add(run_id)
        if _already_ingested(client, run_id):
            log(f"SKIP  {path}: already ingested (run {run_id})")
            counts["skipped"] += 1
            continue
        if dry_run:
            log(f"DRY   {path}: would ingest as run {run_id}")
            counts["ingested"] += 1
            continue
        try:
            client.bulk_ingest(**tables)
        except Exception as exc:
            log(f"FAIL  {path}: {exc}")
            counts["failed"] += 1
            continue
        log(f"OK    {path}: run {run_id}")
        counts["ingested"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="picasso-workflow-backfill-locs",
        description=(
            "Backfill FLAT archive *_locs.hdf5 (+ .yaml) datasets into the "
            "picasso-registry (WP-7; idempotent: re-runs skip already-ingested "
            "datasets). Metrics come straight from the hdf5 (NeNA, localization "
            "precision, photons, background, density, n_locs); kinetics/drift/"
            "clustering are out of scope for this pass."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="_locs.hdf5 files and/or directories to scan recursively",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get(SERVICE_URL_ENV, "http://127.0.0.1:8000"),
        help=f"registry service URL (env {SERVICE_URL_ENV}; "
        "default http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get(TOKEN_ENV),
        help=f"write-scope bearer token (env {TOKEN_ENV}; omit on loopback)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load + map + report, POST nothing (existence checks still "
        "query the registry)",
    )
    parser.add_argument(
        "--no-nena",
        action="store_true",
        help="skip the NeNA fit (faster; nena_nm stays NULL) — the other "
        "hdf5 primaries still ingest",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="curated taxon + good/bad label CSV (keyed by run_id). Matched "
        "runs are created WITH their taxon on the experiment (insert-only "
        "store) and their qc/feedback rows in the same transaction.",
    )
    parser.add_argument(
        "--data-source",
        choices=["acquired", "simulated"],
        default=None,
        help="A15/C24 provenance stamped on every ingested run. REQUIRED "
        "knowledge, not guessable from the files: pass 'simulated' for a "
        "simulation databank (sims are default-excluded from learned cohort "
        "ranges via this flag, and it cannot be retrofitted — the store is "
        "append-only). Omitted = recorded as unknown (NULL).",
    )
    args = parser.parse_args(argv)

    if args.data_source is None:
        print(
            "note: --data-source not given — every ingested run's "
            "data_source will be recorded as unknown (NULL) permanently "
            "(append-only). Pass --data-source acquired|simulated if you "
            "know the provenance (you do)."
        )

    from picasso_registry.client import RegistryClient

    labels = None
    if args.labels:
        from .csv_import import load_label_index

        labels = load_label_index(args.labels)

    client = RegistryClient(args.url, token=args.token)
    try:
        counts = ingest_paths(
            client,
            args.paths,
            dry_run=args.dry_run,
            data_source=args.data_source,
            with_nena=not args.no_nena,
            labels=labels,
        )
    except FileNotFoundError as exc:
        parser.error(f"path does not exist: {exc}")
    except Exception as exc:
        raise SystemExit(
            f"aborted: cannot reach/query the registry at {args.url}: {exc}"
        )
    if sum(counts.values()) == 0:
        print(
            "warning: no *_locs.hdf5 files found under the given paths "
            "(pattern: *_locs.hdf5 — check the path and casing)"
        )
    print(
        "backfill-locs{}: {ingested} ingested, {skipped} skipped, "
        "{failed} failed".format(
            " (dry run)" if args.dry_run else "", **counts
        )
    )
    return 1 if counts["failed"] else 0


# Re-exported so callers that only need the stable id (idempotency checks,
# tests) don't reach into .parser.
__all__ = [
    "ingest_paths",
    "ingest_dataset",
    "map_dataset",
    "iter_locs_files",
    "synth_run_id",
    "main",
    "SkipFile",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
