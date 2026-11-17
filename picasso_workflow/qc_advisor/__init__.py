#!/usr/bin/env python3
"""
picasso_workflow.qc_advisor -- shared, LLM-free, deterministic QC advisor +
Materials & Methods for the DNA-PAINT full-automation stack (WP-ADVISOR).

ONE rule engine, imported directly (C39: no ``paint-util`` leaf), by the live
GUI (WP-GUI), the dashboard (WP-DASH), the deterministic notifier, and later the
agent's ``run_rule_checks`` (WP-14):

    from picasso_workflow.qc_advisor import diagnose, focus, Finding
    findings = diagnose(metrics)            # full sweep, prioritised
    findings = focus("nena", metrics)       # only what drives NeNA

The public findings contract WP-GUI consumes is :class:`Finding` (a dataclass
with ``metric / severity / message / cause / action / value / source / detail``
and ``.as_dict()``). Ported from LiveLocalization V0.8.
"""

from picasso_workflow.qc_advisor.diagnose import (
    FOCUS_MAP,
    METRIC_KEYS,
    STAT_METRICS,
    THRESHOLDS,
    center_edge_trend,
    cohort_stats,
    db_anomalies,
    diagnose,
    focus,
    frc_trend,
    summary,
)
from picasso_workflow.qc_advisor.filter_advisor import (
    estimate_min_net_gradient,
    filter_keep_mask,
    filter_preview_counts,
    filter_suggestions,
    suggest_clustering,
    write_clustering_to_qc,
    write_filter_suggestions_to_qc,
    write_min_net_gradient_to_qc,
)
from picasso_workflow.qc_advisor.findings import (
    SEV_ORDER,
    SEVERITIES,
    Finding,
    sort_findings,
)
from picasso_workflow.qc_advisor.materials_methods import (
    materials_and_methods,
)
from picasso_workflow.qc_advisor.notifier import (
    EVENT_SEVERITY,
    NotifierEngine,
    Notification,
    RegistryEvent,
    Subscription,
    classify_query,
)

__all__ = [
    # findings contract
    "Finding",
    "SEVERITIES",
    "SEV_ORDER",
    "sort_findings",
    # diagnostics
    "diagnose",
    "focus",
    "summary",
    "frc_trend",
    "center_edge_trend",
    "cohort_stats",
    "db_anomalies",
    "THRESHOLDS",
    "METRIC_KEYS",
    "FOCUS_MAP",
    "STAT_METRICS",
    # filter / detection / clustering advisor
    "filter_suggestions",
    "filter_keep_mask",
    "filter_preview_counts",
    "estimate_min_net_gradient",
    "suggest_clustering",
    "write_filter_suggestions_to_qc",
    "write_clustering_to_qc",
    "write_min_net_gradient_to_qc",
    # materials & methods
    "materials_and_methods",
    # notifier (C35, deterministic half)
    "NotifierEngine",
    "RegistryEvent",
    "Subscription",
    "Notification",
    "EVENT_SEVERITY",
    "classify_query",
]
