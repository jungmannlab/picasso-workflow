"""Learn per-taxon expected-metric ranges from the populated history.

WP-7 final step: from the backfilled history, compute per-taxon expected ranges
for each metric so the recommender / QC can condition on "what a run of this
sample type usually looks like". **Learned ONLY from metadata-complete
records** — the untouched-default / metadata-incomplete datasets are ingested
(append-only) but kept OUT of the reference set, so a half-labeled run can't
skew a cohort's learned band.

Two gates compose:

* **taxon known** — the record's ``experiment.sample_taxon_id`` is set (attached
  via the curated CSV, :mod:`.csv_import`); unlabeled runs have no taxon to
  learn for.
* **metadata-complete** — ``experiment.extra.metadata_complete`` is True (the
  file-derivable completeness gate from :mod:`.parser`), AND, per A15/C24,
  ``acquisition_run.data_source`` is not ``"simulated"`` (sims are excluded
  from learned ranges by default).

For each (taxon, metric) with enough samples, we record a robust range
(``p05``/``p50``/``p95`` + ``n``), and optionally persist it onto the taxon's
``expected_metrics`` (the ``sample_taxonomy.expected_metrics`` column the
recommender reads via ``node_defaults``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

import numpy as np

logger = logging.getLogger(__name__)

#: Minimum metadata-complete samples a (taxon, metric) needs before a range is
#: emitted — below this the percentiles are noise.
MIN_SAMPLES = 3

#: Metrics we learn ranges for (the A1 primaries + the cheap hdf5 observables).
#: Extendable; unknown-but-present numeric metrics are ignored here.
RANGE_METRICS = (
    "nena_nm",
    "lp_x_nm",
    "lp_y_nm",
    "photons_median",
    "background",
    "psf_sigma_x",
    "psf_sigma_y",
    "density_locs_um2",
    "n_locs",
)


def _metadata_complete(experiment: Mapping[str, Any]) -> bool:
    """True iff the record is metadata-complete (from experiment.extra)."""
    extra = experiment.get("extra") or {}
    return bool(extra.get("metadata_complete"))


def _list_all(client, resource: str, page: int = 500) -> list[dict]:
    """Page through a registry list endpoint (limit/offset only, no filter)."""
    out: list[dict] = []
    offset = 0
    while True:
        batch = client.list(resource, limit=page, offset=offset)
        out.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return out


#: Default provenance tag stamped on registry range rows (mirrors ingest's
#: BACKFILL_TAG convention), so learned-range snapshots are attributable.
RANGE_SOURCE = "wp7-backfill-ranges"


def _new_ulid() -> str:
    """Mint a fresh ULID id for a range row (lazy python-ulid import).

    Reuses the backfill's python-ulid dependency (the same module
    :mod:`.parser` mints run ids with); it ships with
    ``pip install 'picasso-registry[client]'`` and is NOT a base dep of
    picasso-workflow, so the import stays inside the registry-write path.
    """
    from ulid import ULID

    return str(ULID())


def _percentile_range(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "n": int(arr.size),
    }


def collect_metric_samples(
    client,
    *,
    metrics: tuple[str, ...] = RANGE_METRICS,
) -> dict[str, dict[str, list[float]]]:
    """Gather ``{taxon_id: {metric: [values...]}}`` from the registry.

    Only metadata-complete, non-simulated, taxon-labeled records contribute.
    Walks acquisition_run → experiment (for the taxon + completeness gate) →
    analysis_run → metrics, using the registry's list endpoints. Kept separate
    from the range math so tests can assert the gate without touching numpy.
    """
    samples: dict[str, dict[str, list[float]]] = {}

    # The registry's list endpoint has no server-side filter (limit/offset
    # only), so pull each table once and join client-side — one pass, no N+1.
    try:
        experiments = {e["id"]: e for e in _list_all(client, "experiment")}
        acq_runs = _list_all(client, "acquisition_run")
        analyses = _list_all(client, "analysis_run")
        metric_rows = _list_all(client, "metrics")
    except Exception as exc:
        logger.warning("could not list registry tables: %s", exc)
        return samples

    # acquisition_run → taxon, keeping only metadata-complete, non-sim,
    # taxon-labeled runs (the two gates compose here).
    run_taxon: dict[str, str] = {}
    for acq in acq_runs:
        if acq.get("data_source") == "simulated":
            continue  # A15/C24: sims excluded from learned ranges
        exp = experiments.get(acq.get("experiment_id"))
        if exp is None:
            continue
        taxon_id = exp.get("sample_taxon_id")
        if not taxon_id or not _metadata_complete(exp):
            continue  # unlabeled or metadata-incomplete → excluded
        run_id = acq.get("id")
        if run_id:
            run_taxon[run_id] = taxon_id

    # analysis_run → its run's taxon (only for kept runs).
    ana_taxon: dict[str, str] = {}
    for ana in analyses:
        taxon_id = run_taxon.get(ana.get("acquisition_run_id"))
        if taxon_id and ana.get("id"):
            ana_taxon[ana["id"]] = taxon_id

    for row in metric_rows:
        taxon_id = ana_taxon.get(row.get("analysis_run_id"))
        if not taxon_id:
            continue
        for metric in metrics:
            val = row.get(metric)
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fval):
                continue
            samples.setdefault(taxon_id, {}).setdefault(metric, []).append(
                fval
            )
    return samples


def write_ranges_to_registry(
    client,
    ranges: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    source: str = RANGE_SOURCE,
    log: Callable[[str], None] = print,
) -> int:
    """POST one ``cohort_metric_range`` row per (taxon, metric) learned range.

    The registry is append-only (no PATCH of ``sample_taxonomy.expected_metrics``
    — see :func:`compute_taxon_ranges`), so the learned ranges are persisted as
    their own append-only snapshots: one row per (taxon_id, metric), carrying
    ``p05``/``p50``/``p95``/``n`` + ``source``, with a freshly minted ULID id.
    Re-running inserts NEW rows; consumers read the LATEST per (taxon_id,
    metric) by ``created_at`` / insertion order.

    **Best-effort / non-blocking** (same posture as the rest of the backfill):
    a failing write is logged and swallowed — never raised — so a transient
    registry hiccup can't lose the JSON sidecar or abort the run. Returns the
    count of rows successfully written.
    """
    written = 0
    for taxon_id, by_metric in ranges.items():
        for metric, stats in by_metric.items():
            try:
                client.create(
                    "cohort_metric_range",
                    id=_new_ulid(),
                    taxon_id=taxon_id,
                    metric=metric,
                    p05=stats.get("p05"),
                    p50=stats.get("p50"),
                    p95=stats.get("p95"),
                    n=stats.get("n"),
                    source=source,
                )
                written += 1
            except Exception as exc:  # best-effort: log + continue
                logger.warning(
                    "could not write range for (%s, %s): %s",
                    taxon_id,
                    metric,
                    exc,
                )
    log(f"OK    wrote {written} cohort_metric_range rows to the registry")
    return written


def compute_taxon_ranges(
    client,
    *,
    metrics: tuple[str, ...] = RANGE_METRICS,
    min_samples: int = MIN_SAMPLES,
    out_path: str | None = None,
    registry_client=None,
    source: str = RANGE_SOURCE,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-taxon expected-metric ranges from metadata-complete history.

    Returns ``{taxon_id: {metric: {p05, p50, p95, n}}}`` for every (taxon,
    metric) with at least ``min_samples`` metadata-complete samples — the
    primary deliverable, consumed by the recommender / QC.

    **Not persisted back onto ``sample_taxonomy.expected_metrics``:** the
    registry is append-only (insert-only, no PATCH), and the taxon node is
    created before its history exists, so its ``expected_metrics`` can't be
    updated in place afterwards. Instead the ranges are returned and, with
    ``out_path``, written to a JSON sidecar the recommender reads — the ranges
    are *learned per cohort* (A1/C19), re-derived from the backfill rather than
    frozen into the taxonomy row.

    With ``registry_client`` the ranges are ALSO persisted queryably in the
    registry, as append-only ``cohort_metric_range`` snapshots (one row per
    (taxon, metric); see :func:`write_ranges_to_registry`). That write is
    best-effort — a registry hiccup never aborts the run or loses the JSON
    sidecar. The sidecar stays the fallback; the two paths are independent.
    """
    samples = collect_metric_samples(client, metrics=metrics)
    ranges: dict[str, dict[str, dict[str, float]]] = {}
    for taxon_id, by_metric in samples.items():
        taxon_ranges: dict[str, dict[str, float]] = {}
        for metric, values in by_metric.items():
            if len(values) < min_samples:
                continue
            taxon_ranges[metric] = _percentile_range(values)
        if taxon_ranges:
            ranges[taxon_id] = taxon_ranges

    if out_path:
        import json

        with open(out_path, "w") as fh:
            json.dump(ranges, fh, indent=2)
        log(f"OK    wrote {len(ranges)} taxon ranges → {out_path}")

    if registry_client is not None:
        write_ranges_to_registry(
            registry_client, ranges, source=source, log=log
        )
    return ranges


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import os

    from .ingest import SERVICE_URL_ENV, TOKEN_ENV

    parser = argparse.ArgumentParser(
        prog="picasso-workflow-backfill-ranges",
        description=(
            "Learn per-taxon expected-metric ranges from the backfilled "
            "history (metadata-complete, non-simulated, taxon-labeled records "
            "only). Prints the ranges JSON; --out also writes a sidecar the "
            "recommender reads (the append-only taxonomy row is not updated)."
        ),
    )
    parser.add_argument(
        "--url",
        default=os.environ.get(SERVICE_URL_ENV, "http://127.0.0.1:8000"),
    )
    parser.add_argument("--token", default=os.environ.get(TOKEN_ENV))
    parser.add_argument(
        "--min-samples",
        type=int,
        default=MIN_SAMPLES,
        help=f"min metadata-complete samples per range (default {MIN_SAMPLES})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write the learned ranges to this JSON sidecar (the recommender "
        "reads it; the append-only taxonomy row is not updated)",
    )
    parser.add_argument(
        "--to-registry",
        action="store_true",
        help="also persist the learned ranges to the registry as append-only "
        "cohort_metric_range rows (one per taxon+metric; best-effort — a write "
        "failure is logged, not fatal). The --out JSON sidecar stays the "
        "fallback. Writes back to the same --url registry it reads from.",
    )
    args = parser.parse_args(argv)

    from picasso_registry.client import RegistryClient

    client = RegistryClient(args.url, token=args.token)
    ranges = compute_taxon_ranges(
        client,
        min_samples=args.min_samples,
        out_path=args.out,
        registry_client=client if args.to_registry else None,
    )
    print(json.dumps(ranges, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
