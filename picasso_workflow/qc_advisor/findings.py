#!/usr/bin/env python3
"""
findings.py -- the structured Finding contract shared across the QC advisor.

A ``Finding`` is the single explainable unit every advisor function emits: a
severity, a human message, the likely cause, a concrete action, and (where a
number drove it) the observed value plus its source. This is the STABLE public
contract that downstream consumers -- the live GUI (WP-GUI), the dashboard, the
notifier, and later the agent's ``run_rule_checks`` (WP-14) -- render. Keep it
explicit and additive: consumers may rely on every field below.

Ported from LiveLocalization V0.8 ``qc_advisor.Finding`` and extended with a
``source`` tag (which advisor produced the finding) and ``as_dict`` for
serialisation across the streaming API / registry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

# Severity ordering: lower sorts first (most urgent on top). "ok" is a positive
# finding kept for completeness; "info" is neutral context.
SEVERITIES = ("bad", "warn", "info", "ok")
SEV_ORDER: Dict[str, int] = {s: i for i, s in enumerate(SEVERITIES)}

# Glyphs used by the plain-text renderer (kept from V0.8).
_SEV_TAG = {"bad": "●", "warn": "▲", "info": "•", "ok": "✓"}


@dataclass
class Finding:
    """One prioritised QC finding.

    Attributes
    ----------
    metric : str
        The metric / topic key the finding is about (e.g. ``"nena"``,
        ``"photons_per_loc"``, ``"frc_trend"``). Not necessarily a raw qc.json
        key -- some findings summarise a trend or a cross-check.
    severity : str
        One of :data:`SEVERITIES` -- ``"bad" | "warn" | "info" | "ok"``.
    message : str
        What was observed, in plain language.
    cause : str
        The likely cause (may be empty for ``ok``/``info``).
    action : str
        A concrete, actionable suggestion (may be empty).
    value : float | None
        The observed numeric value that drove the finding, when there is one.
    source : str
        Which advisor produced it -- ``"diagnose"``, ``"frc_trend"``,
        ``"center_edge_trend"``, ``"db_anomalies"``, ``"filter"``,
        ``"clustering"``, ``"min_net_gradient"``, ``"notifier"``. Lets a
        consumer group / filter by producer without re-parsing the message.
    detail : dict
        Optional machine-readable payload (e.g. the FRC-trend fit, the filter
        threshold set, or the kept/removed preview counts) that a rich GUI can
        render beyond the text. Never required to be present.
    """

    metric: str
    severity: str
    message: str
    cause: str = ""
    action: str = ""
    value: Optional[float] = None
    source: str = "diagnose"
    detail: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEV_ORDER:
            raise ValueError(
                f"severity must be one of {SEVERITIES}, got {self.severity!r}"
            )

    @property
    def sort_key(self) -> tuple:
        """(severity-rank, metric) -- the canonical prioritised ordering."""
        return (SEV_ORDER.get(self.severity, len(SEVERITIES)), self.metric)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-serialisable dict (for the streaming API / registry / GUI)."""
        return asdict(self)

    def __str__(self) -> str:
        tag = _SEV_TAG.get(self.severity, "•")
        s = f"{tag} [{self.metric}] {self.message}"
        if self.cause:
            s += f"\n     ↳ likely: {self.cause}"
        if self.action:
            s += f"\n     ↳ try: {self.action}"
        return s


def sort_findings(findings: Iterable[Finding]) -> List[Finding]:
    """Return the findings sorted into the canonical prioritised order."""
    return sorted(findings, key=lambda f: f.sort_key)
