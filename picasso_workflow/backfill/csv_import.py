"""Curated taxon + good/bad label CSV for the flat-archive backfill.

The sample *descriptor* (taxon) and the manually-curated good/bad gold labels
are human knowledge — not in the archive files — so a reviewer supplies them in
a CSV, one row per dataset, keyed by the dataset's ``run_id`` (the stable,
content-derived id :mod:`.parser` mints; see ``synth_run_id``).

**Two append-only facts about the registry shape the split here:**

* The run's **taxon** is set on ``experiment.sample_taxon_id``, which the
  ``/cohort`` join and the learned ranges read. The store is **insert-only**
  (no PATCH/PUT — re-POSTing an id 409s), so the taxon can only be attached
  **when the experiment row is first created**. Therefore taxon labeling is fed
  to the **ingest** (``ingest --labels CSV``), which creates each experiment
  *with* its taxon in the same ``/bulk`` transaction (see :mod:`.ingest`).
* The **qc** / **feedback** rows are genuinely append-only, so they can be
  added to an already-ingested run at any time. That is what the standalone
  :func:`import_labels` CLI does (and it also creates the taxon *nodes* so they
  exist, but it will NOT silently fail trying to retrofit a taxon onto an
  already-created experiment — it warns instead).

CSV columns (header-driven; extras ignored, missing-but-optional stay absent):

====================  ====================================================
``run_id``            REQUIRED — the dataset's stable run id (matches ingest)
``taxon`` / ``taxon_name``  taxon node name (the descriptor label)
``taxon_path``        optional materialized-path hint for the node
``label``             good / bad / pass / fail / 1 / 0 (the gold verdict)
``curator`` / ``decided_by``  who labeled it
``reason`` / ``note``  free-text justification
====================  ====================================================
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

#: label strings → qc.passed / feedback.operator_decision
_GOOD = {"good", "pass", "passed", "ok", "yes", "true", "1"}
_BAD = {"bad", "fail", "failed", "reject", "no", "false", "0"}


def _first(row: Mapping[str, Any], *keys: str) -> str | None:
    """First non-empty value among ``keys`` (case-insensitive header)."""
    lower = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        val = lower.get(key.lower())
        if val is not None and str(val).strip() != "":
            return str(val).strip()
    return None


def _label_bool(label: str | None) -> bool | None:
    if label is None:
        return None
    tok = label.strip().lower()
    if tok in _GOOD:
        return True
    if tok in _BAD:
        return False
    return None


def read_label_rows(csv_path: str) -> list[dict[str, Any]]:
    """Parse the label CSV into normalized row dicts (pure IO/parse).

    Each dict has keys ``run_id, taxon, taxon_path, label, passed, curator,
    reason``. Unit-testable without a registry.
    """
    rows: list[dict[str, Any]] = []
    # utf-8-sig strips an Excel UTF-8 BOM: without it the first header becomes
    # "﻿run_id", the required run_id never matches, and EVERY row is
    # dropped as run-id-less — silently voiding the whole curated label set.
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as fh:
        for raw in csv.DictReader(fh):
            label = _first(raw, "label", "verdict", "quality", "good_bad")
            rows.append(
                {
                    "run_id": _first(
                        raw, "run_id", "acquisition_run_id", "run"
                    ),
                    "taxon": _first(
                        raw, "taxon", "taxon_name", "sample_taxon"
                    ),
                    "taxon_path": _first(raw, "taxon_path", "path"),
                    "label": label,
                    "passed": _label_bool(label),
                    "curator": _first(
                        raw, "curator", "decided_by", "operator"
                    ),
                    "reason": _first(raw, "reason", "note", "comment"),
                }
            )
    return rows


def load_label_index(csv_path: str) -> dict[str, dict[str, Any]]:
    """``{run_id: normalized_row}`` for the ingest to look up a run's labels.

    A later row with the same ``run_id`` wins (last curation decision);
    run-id-less rows are dropped (they can't be keyed to a dataset).
    """
    index: dict[str, dict[str, Any]] = {}
    for row in read_label_rows(csv_path):
        if row["run_id"]:
            index[row["run_id"]] = row
    return index


def _list_all(client, resource: str, page: int = 500) -> list[dict]:
    """Page through a registry list endpoint (it only takes limit/offset).

    The generic ``GET /<resource>`` has no server-side column filter, so the
    backfill lists the whole table and filters client-side. Paging avoids the
    silent 100-row default truncation.
    """
    out: list[dict] = []
    offset = 0
    while True:
        batch = client.list(resource, limit=page, offset=offset)
        out.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return out


def ensure_taxon(
    client,
    name: str,
    path: str | None = None,
    *,
    cache: dict[str, str] | None = None,
) -> str:
    """Create-or-reuse a taxon node by name; return its id.

    Taxonomy is append-only but the *node* is reused by name across runs, so
    this dedupes: an existing ``sample_taxonomy`` row with the same name is
    reused, otherwise one is created. ``cache`` dedupes within a single sweep.
    """
    if cache is not None and name in cache:
        return cache[name]
    try:
        existing = [
            t
            for t in _list_all(client, "sample_taxonomy")
            if t.get("name") == name
        ]
    except Exception:
        existing = []
    if existing:
        taxon_id = existing[0].get("id")
    else:
        taxon_id = client.add_taxon(name=name, path=path).get("id")
    if cache is not None:
        cache[name] = taxon_id
    return taxon_id


def label_bulk_rows(
    row: Mapping[str, Any], *, analysis_run_id: str, acquisition_run_id: str
) -> dict[str, list[dict]]:
    """Build the ``qc`` / ``feedback`` ``/bulk`` rows for one labeled run.

    Pure: no registry. Returns the (possibly empty) ``qc`` and ``feedback``
    lists for the dataset's good/bad verdict, keyed to its analysis_run. A row
    with no verdict yields no qc/feedback (taxon-only labeling).
    """
    if row.get("passed") is None:
        return {}
    passed = bool(row["passed"])
    qc = {
        "analysis_run_id": analysis_run_id,
        "passed": passed,
        "decided_by": row.get("curator"),
        "reason": row.get("reason"),
    }
    feedback = {
        "acquisition_run_id": acquisition_run_id,
        "analysis_run_id": analysis_run_id,
        "operator_decision": "good" if passed else "bad",
        "used_in_final": passed,
        "note": row.get("reason"),
    }
    return {"qc": [qc], "feedback": [feedback]}


def _run_refs(client, run_id: str) -> tuple[str | None, str | None, bool]:
    """``(experiment_id, analysis_run_id, taxon_already_set)`` for a run.

    ``(None, None, False)`` if the acquisition_run is absent. The analysis_run
    is this run's backfill analysis_run (filtered client-side — the list
    endpoint has no server-side filter).
    """
    try:
        acq = client.get("acquisition_run", run_id)
    except Exception:
        return None, None, False
    exp_id = acq.get("experiment_id")
    taxon_set = False
    if exp_id:
        try:
            taxon_set = bool(
                client.get("experiment", exp_id).get("sample_taxon_id")
            )
        except Exception:
            taxon_set = False
    ana_id = None
    analyses = [
        a
        for a in _list_all(client, "analysis_run")
        if a.get("acquisition_run_id") == run_id
    ]
    if analyses:
        ana_id = analyses[0].get("id")
    return exp_id, ana_id, taxon_set


def import_labels(
    client,
    csv_path: str,
    *,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Append qc/feedback labels (and taxon nodes) for already-ingested runs.

    Does only the **append-only** work: creates the taxon *nodes* and writes
    qc + feedback rows keyed to each run's analysis_run. It does NOT retrofit
    ``experiment.sample_taxon_id`` — the store is insert-only, so taxon↔run
    linkage must be set at ingest (``ingest --labels``); a run whose experiment
    has no taxon yet is reported so the operator re-ingests with ``--labels``.

    Returns counts ``{taxa, qc, feedback, skipped, taxon_unlinked}``.
    """
    counts = {
        "taxa": 0,
        "qc": 0,
        "feedback": 0,
        "skipped": 0,
        "taxon_unlinked": 0,
    }
    taxon_cache: dict[str, str] = {}
    seen_taxa: set[str] = set()
    for row in read_label_rows(csv_path):
        run_id = row["run_id"]
        if not run_id:
            log("SKIP  row: no run_id")
            counts["skipped"] += 1
            continue
        if row["taxon"] is None and row["label"] is None:
            log(f"SKIP  {run_id}: no taxon and no label")
            counts["skipped"] += 1
            continue
        exp_id, ana_id, taxon_set = _run_refs(client, run_id)
        if exp_id is None and ana_id is None:
            log(f"SKIP  {run_id}: not ingested (no acquisition_run)")
            counts["skipped"] += 1
            continue

        if dry_run:
            log(
                f"DRY   {run_id}: taxon={row['taxon']!r} "
                f"label={row['label']!r}"
            )
            continue

        # Taxon node (append-only), but linkage to the experiment can't be
        # retrofitted — warn if the experiment was ingested without it.
        if row["taxon"]:
            ensure_taxon(
                client, row["taxon"], row["taxon_path"], cache=taxon_cache
            )
            if row["taxon"] not in seen_taxa:
                counts["taxa"] += 1
                seen_taxa.add(row["taxon"])
            if not taxon_set:
                log(
                    f"WARN  {run_id}: taxon node created but NOT linked to the "
                    "experiment (store is insert-only). Re-ingest this run "
                    "with `ingest --labels <csv>` to set sample_taxon_id."
                )
                counts["taxon_unlinked"] += 1

        bulk = label_bulk_rows(
            row, analysis_run_id=ana_id or "", acquisition_run_id=run_id
        )
        if bulk and ana_id:
            client.bulk_ingest(**bulk)
            counts["qc"] += len(bulk.get("qc", []))
            counts["feedback"] += len(bulk.get("feedback", []))
        log(f"OK    {run_id}")
    return counts


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    from .ingest import SERVICE_URL_ENV, TOKEN_ENV

    parser = argparse.ArgumentParser(
        prog="picasso-workflow-backfill-labels",
        description=(
            "Append qc + feedback labels (and create taxon nodes) for "
            "already-ingested flat-archive runs from a curated CSV keyed by "
            "run_id. NOTE: taxon↔experiment linkage is set at ingest "
            "(`ingest --labels`), not here — the store is append-only."
        ),
    )
    parser.add_argument("csv", help="curated label CSV path")
    parser.add_argument(
        "--url",
        default=os.environ.get(SERVICE_URL_ENV, "http://127.0.0.1:8000"),
    )
    parser.add_argument("--token", default=os.environ.get(TOKEN_ENV))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse + report, write nothing",
    )
    args = parser.parse_args(argv)

    from picasso_registry.client import RegistryClient

    client = RegistryClient(args.url, token=args.token)
    counts = import_labels(client, args.csv, dry_run=args.dry_run)
    print(
        "labels{}: {taxa} taxa, {qc} qc, {feedback} feedback, "
        "{skipped} skipped, {taxon_unlinked} taxon-unlinked".format(
            " (dry run)" if args.dry_run else "", **counts
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
