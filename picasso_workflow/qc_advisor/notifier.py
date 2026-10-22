#!/usr/bin/env python3
"""
notifier.py -- C35 deterministic, LLM-free lifecycle notifier + query surface.

Re-homes LiveLocalization V0.8's status/quality push logic as a transport-
agnostic, ``run_id``-keyed engine driven by REGISTRY events -- so ONE surface
spans BOTH the live phase AND the cluster-analysis phase (hand-off / started /
done / failed, drift-QC converged?, final metrics vs cohort). The engine only
DECIDES what to push and how to answer a query; the delivery transport (Slack,
web, ...) is injected as a ``sink`` callable, so nothing here talks to Slack and
Slack never runs on the cluster.

DETERMINISTIC HALF ONLY (C35): no LLM narration (that is WP-14) and no
interactive-actuation buttons (later + gated). Anti-noise is required and built
in: per-user/per-run subscription, per-event severity thresholds, and ONE thread
per ``run_id`` with dedup -- never a firehose.

Design
------
* :class:`NotifierEngine` is fed :class:`RegistryEvent` objects (typically
  translated from the ``analysis_run`` / ``metrics`` / ``artifact`` records
  picasso-workflow already writes). For each event it computes zero or more
  :class:`Notification` messages, one per subscribed user, filtered by that
  user's severity threshold and deduped within the run's thread.
* :meth:`answer_query` handles read-only ``status`` / ``quality`` questions
  ("how's my run/analysis?", "quality"), returning a registry-derived answer.
* ``sink(notification)`` is called for each message to deliver (optional). The
  engine returns the notifications too, so tests need no sink.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from picasso_workflow.qc_advisor.diagnose import db_anomalies, diagnose
from picasso_workflow.qc_advisor.findings import SEV_ORDER, Finding

# Lifecycle event kinds the engine understands, in phase order. The severity of
# each is fixed here (anti-noise: routine progress is low, failures are high).
# live phase -> cluster/analysis phase.
EVENT_SEVERITY: Dict[str, str] = {
    "acquisition_started": "info",
    "qc_update": "info",  # a periodic live metrics update
    "early_abort": "warn",  # live QC tripped an early abort
    "acquisition_done": "info",
    "handoff": "info",  # localizations shipped to the cluster
    "analysis_started": "info",
    "drift_converged": "info",
    "drift_not_converged": "warn",
    "analysis_done": "ok",
    "analysis_failed": "bad",
}

# Events that always deliver regardless of a user's severity threshold, because
# they are terminal / actionable: a failure, an early abort, or a finished run
# (a completion is delivered even though "ok" ranks below "info"). Everything
# else is gated by the subscriber's threshold.
_ALWAYS_DELIVER = {"analysis_failed", "early_abort", "analysis_done"}


@dataclass
class RegistryEvent:
    """A normalised lifecycle event derived from a registry record.

    ``kind`` is one of :data:`EVENT_SEVERITY`. ``metrics`` carries the latest
    metric dict (qc.json keys) when the event has one; ``cohort_stats`` the
    matched-cohort stats for a metrics-vs-cohort comparison; ``detail`` any extra
    payload (drift ptp, artifact/thumbnail pointer, error text)."""

    run_id: str
    kind: str
    metrics: Optional[Dict[str, Any]] = None
    cohort_stats: Optional[Dict[str, Dict[str, float]]] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    seq: Optional[int] = None  # monotonically increasing per run, if known


@dataclass
class Subscription:
    """A user's interest in a run. ``min_severity`` gates routine events; a
    user only hears events at or above it (plus the always-deliver terminal
    ones)."""

    user: str
    run_id: str
    min_severity: str = "info"


@dataclass
class Notification:
    """One message to deliver. ``thread_key`` is the per-run thread so a
    transport groups a run's messages into ONE thread."""

    run_id: str
    user: str
    kind: str
    severity: str
    message: str
    thread_key: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "user": self.user,
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "thread_key": self.thread_key,
            "detail": self.detail,
        }


def _sev_at_least(sev: str, floor: str) -> bool:
    """True if ``sev`` is at least as urgent as ``floor`` (bad > warn > info)."""
    return SEV_ORDER.get(sev, 99) <= SEV_ORDER.get(floor, 99)


def _worst_finding_severity(findings: List[Finding]) -> Optional[str]:
    if not findings:
        return None
    return min(
        (f.severity for f in findings), key=lambda s: SEV_ORDER.get(s, 99)
    )


class NotifierEngine:
    """Deterministic lifecycle notifier. Stateless except for the per-run thread
    keys, the subscription table, and a dedup ledger (so re-fed events don't
    re-notify)."""

    # sentinel "user" for a fallback/broadcast notification (a critical with no
    # per-run subscribers). Distinct from any real user id.
    FALLBACK_USER = "*"

    def __init__(
        self,
        sink: Optional[Callable[[Notification], Any]] = None,
        fallback_sink: Optional[Callable[[Notification], Any]] = None,
    ):
        """``sink`` delivers per-subscriber notifications. ``fallback_sink``
        (defaulting to ``sink``) receives always-deliver criticals for a run
        with ZERO subscribers, so an ``analysis_failed`` on the cluster before /
        after anyone subscribed still reaches someone (e.g. an ops channel) and
        is never silently dropped."""
        self._sink = sink
        self._fallback_sink = (
            fallback_sink if fallback_sink is not None else sink
        )
        self._subs: Dict[str, List[Subscription]] = {}
        # dedup ledger: run_id -> set of already-emitted (user, dedup_key)
        self._seen: Dict[str, set] = {}
        # last-known metrics / cohort per run, for status/quality queries.
        self._last: Dict[str, RegistryEvent] = {}

    # ── subscription management ──────────────────────────────────────────────
    def subscribe(
        self, user: str, run_id: str, min_severity: str = "info"
    ) -> Subscription:
        if min_severity not in SEV_ORDER:
            raise ValueError(f"unknown severity {min_severity!r}")
        sub = Subscription(user=user, run_id=run_id, min_severity=min_severity)
        subs = self._subs.setdefault(run_id, [])
        # replace an existing subscription for the same user
        subs[:] = [s for s in subs if s.user != user]
        subs.append(sub)
        return sub

    def unsubscribe(self, user: str, run_id: str) -> None:
        subs = self._subs.get(run_id, [])
        subs[:] = [s for s in subs if s.user != user]

    def subscribers(self, run_id: str) -> List[Subscription]:
        return list(self._subs.get(run_id, []))

    @staticmethod
    def thread_key(run_id: str) -> str:
        """ONE thread per run_id (anti-noise)."""
        return f"run::{run_id}"

    # ── event handling ───────────────────────────────────────────────────────
    def handle_event(self, event: RegistryEvent) -> List[Notification]:
        """Process one registry event: build per-subscriber notifications,
        filtered by severity + deduped, deliver via the sink, and return them.
        """
        if event.kind not in EVENT_SEVERITY:
            raise ValueError(f"unknown event kind {event.kind!r}")
        self._last[event.run_id] = event
        severity = self._event_severity(event)
        message, detail = self._render_event(event, severity)
        dedup_key = self._dedup_key(event)
        seen = self._seen.setdefault(event.run_id, set())

        out: List[Notification] = []
        subs = self._subs.get(event.run_id, [])
        for sub in subs:
            gate = event.kind in _ALWAYS_DELIVER or _sev_at_least(
                severity, sub.min_severity
            )
            if not gate:
                continue
            key = (sub.user, dedup_key)
            if key in seen:
                continue
            seen.add(key)
            note = self._make_notification(
                event, sub.user, severity, message, detail
            )
            out.append(note)
            if self._sink is not None:
                self._sink(note)

        # Fallback/broadcast: an always-deliver critical must reach SOMEONE even
        # with ZERO per-run subscribers (e.g. it fails on the cluster before or
        # after anyone subscribed). Gate on the absence of SUBSCRIBERS, not on an
        # empty `out` -- if subscribers exist, an empty `out` just means the event
        # was already delivered to them and deduped on replay, which must NOT
        # trigger a broadcast. Deduped under the sentinel FALLBACK_USER too, so a
        # replay with no subscribers doesn't re-broadcast either.
        if event.kind in _ALWAYS_DELIVER and not subs:
            key = (self.FALLBACK_USER, dedup_key)
            if key not in seen:
                seen.add(key)
                note = self._make_notification(
                    event, self.FALLBACK_USER, severity, message, detail
                )
                out.append(note)
                if self._fallback_sink is not None:
                    self._fallback_sink(note)
        return out

    def _make_notification(
        self,
        event: RegistryEvent,
        user: str,
        severity: str,
        message: str,
        detail: Dict[str, Any],
    ) -> Notification:
        return Notification(
            run_id=event.run_id,
            user=user,
            kind=event.kind,
            severity=severity,
            message=message,
            thread_key=self.thread_key(event.run_id),
            detail=detail,
        )

    def _event_severity(self, event: RegistryEvent) -> str:
        """The severity to attach to this event. Metric-bearing events escalate
        to the worst finding severity so a bad live run is flagged as bad."""
        base = EVENT_SEVERITY[event.kind]
        if event.kind in ("qc_update", "acquisition_done", "analysis_done"):
            worst = self._worst_metric_severity(event)
            if worst is not None and SEV_ORDER.get(worst, 99) < SEV_ORDER.get(
                base, 99
            ):
                return worst
        return base

    def _worst_metric_severity(self, event: RegistryEvent) -> Optional[str]:
        findings: List[Finding] = []
        if event.metrics:
            findings += diagnose(event.metrics)
        if event.metrics and event.cohort_stats:
            findings += db_anomalies(event.metrics, event.cohort_stats)
        return _worst_finding_severity(findings)

    def _dedup_key(self, event: RegistryEvent) -> str:
        """A stable key so the SAME event fed twice notifies once. Terminal /
        one-shot kinds (done, failed, handoff, ...) dedup on the kind alone -- one
        per run. Periodic ``qc_update`` events must NOT collapse to the kind, or
        every update after the first (including a WORSENING one) is swallowed:
          * with a ``seq`` -> key on the sequence number (each update distinct);
          * without a ``seq`` -> key on the escalated severity + a rounded metric
            fingerprint, so an update that changes the picture (e.g. NeNA 3->9 nm,
            info->bad) is delivered while an identical repeat is still deduped.
        """
        if event.kind != "qc_update":
            return event.kind
        if event.seq is not None:
            return f"qc_update#{event.seq}"
        severity = self._event_severity(event)
        fp = self._metrics_fingerprint(event.metrics)
        return f"qc_update@{severity}|{fp}"

    @staticmethod
    def _metrics_fingerprint(metrics: Optional[Dict[str, Any]]) -> str:
        """A stable, order-independent fingerprint of the metric values that
        drive severity, rounded so noise doesn't defeat dedup but a real change
        does. Empty when no metrics."""
        if not metrics:
            return ""
        parts = []
        for k in sorted(metrics):
            v = metrics[k]
            try:
                parts.append(f"{k}={round(float(v), 3)}")
            except (TypeError, ValueError):
                continue
        return ",".join(parts)

    def _render_event(
        self, event: RegistryEvent, severity: str
    ) -> tuple[str, Dict[str, Any]]:
        """Deterministic message text (+ machine detail) for an event."""
        k = event.kind
        d = dict(event.detail or {})
        if k == "acquisition_started":
            return (f"Acquisition started for run {event.run_id}.", d)
        if k == "qc_update":
            return (self._metrics_line("Live QC", event), d)
        if k == "early_abort":
            reason = d.get("reason", "live QC threshold tripped")
            return (
                f"Early-abort recommended for run {event.run_id}: {reason}.",
                d,
            )
        if k == "acquisition_done":
            return (self._metrics_line("Acquisition complete", event), d)
        if k == "handoff":
            return (
                f"Localizations handed off to the cluster for run "
                f"{event.run_id}.",
                d,
            )
        if k == "analysis_started":
            return (
                f"Cluster analysis started for run {event.run_id}.",
                d,
            )
        if k == "drift_converged":
            ptp = d.get("drift_ptp_nm")
            extra = f" (residual {ptp:.0f} nm p-p)" if _isnum(ptp) else ""
            return (
                f"Drift correction converged for run {event.run_id}{extra}.",
                d,
            )
        if k == "drift_not_converged":
            return (
                f"Drift correction did NOT converge for run {event.run_id} — "
                f"check fiducials / stability.",
                d,
            )
        if k == "analysis_done":
            base = self._metrics_line("Analysis complete", event)
            if d.get("thumbnail"):
                base += " Reconstruction thumbnail attached."
            return (base, d)
        if k == "analysis_failed":
            err = d.get("error", "unknown error")
            return (
                f"Analysis FAILED for run {event.run_id}: {err}.",
                d,
            )
        return (f"{k} for run {event.run_id}.", d)

    def _metrics_line(self, prefix: str, event: RegistryEvent) -> str:
        m = event.metrics or {}
        bits = []
        for key, label, fmt in (
            ("nena_zoom_nm", "NeNA", "{:.1f} nm"),
            ("frc_resolution_nm", "FRC", "{:.1f} nm"),
            ("photons_per_loc", "photons", "{:.0f}"),
            ("localizations", "locs", "{:.0f}"),
        ):
            v = m.get(key)
            if _isnum(v):
                bits.append(f"{label} {fmt.format(float(v))}")
        tail = f" — {', '.join(bits)}" if bits else ""
        return f"{prefix} for run {event.run_id}{tail}."

    # ── read-only query surface (status / quality) ───────────────────────────
    def answer_query(
        self, run_id: str, text: str, user: Optional[str] = None
    ) -> Optional[str]:
        """Answer a read-only ``status`` / ``quality`` question about a run,
        derived from the last event seen for it. Returns the answer text, or
        ``None`` if the text isn't a recognised query. Read-only: never mutates
        state, never actuates. Actuation buttons are out of scope (C35)."""
        intent = classify_query(text)
        if intent is None:
            return None
        last = self._last.get(run_id)
        if last is None:
            return f"No status yet for run {run_id}."
        if intent == "status":
            return self._status_reply(run_id, last)
        if intent == "quality":
            return self._quality_reply(run_id, last)
        return None

    def _status_reply(self, run_id: str, last: RegistryEvent) -> str:
        sev = self._event_severity(last)
        msg, _ = self._render_event(last, sev)
        phase = _phase_of(last.kind)
        return f"[{phase}] {msg}"

    def _quality_reply(self, run_id: str, last: RegistryEvent) -> str:
        if not last.metrics:
            return f"No metrics recorded yet for run {run_id}."
        findings: List[Finding] = diagnose(last.metrics)
        if last.cohort_stats:
            findings = findings + db_anomalies(last.metrics, last.cohort_stats)
        findings = sorted(findings, key=lambda f: f.sort_key)
        if not findings:
            return f"No quality issues found for run {run_id}."
        top = findings[:3]
        lines = [f"Quality for run {run_id}:"]
        lines += [str(f) for f in top]
        return "\n".join(lines)


# ── query intent classification (deterministic keyword rules from V0.8) ────────
_QC_RE = re.compile(r"(?<![a-z])qc(?![a-z])")


def classify_query(text: str) -> Optional[str]:
    """Map free-text to a query intent: ``"status"`` | ``"quality"`` | ``None``.

    Mirrors V0.8's keyword rules: 'quality' or a standalone 'qc' -> quality;
    'status' (or 'how's my run/analysis') -> status. Quality takes precedence
    when both appear. Deterministic; no LLM."""
    t = (text or "").lower()
    if "quality" in t or _QC_RE.search(t):
        return "quality"
    if (
        "status" in t
        or "how's my" in t
        or "hows my" in t
        or "how is my" in t
        or "progress" in t
    ):
        return "status"
    return None


def _phase_of(kind: str) -> str:
    """Human phase label for an event kind (live vs cluster)."""
    live = {
        "acquisition_started",
        "qc_update",
        "early_abort",
        "acquisition_done",
    }
    return "live" if kind in live else "cluster"


def _isnum(v: Any) -> bool:
    try:
        float(v)
        return v is not None
    except (TypeError, ValueError):
        return False
