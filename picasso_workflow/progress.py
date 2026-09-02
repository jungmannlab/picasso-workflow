#!/usr/bin/env python
"""
Module Name: progress.py
Author: Heinrich Grabmayr
Initial Date: September 1, 2026
Description: Progress tracking for picasso-workflow runs.

A single emitter (:class:`ProgressManager`) is updated by the workflow
runners at module boundaries and, for long picasso calls, from within a
module. It fans every update out to a list of *sinks* -- small callables
that render the current progress state to a surface (a ``progress.json``
file, the log, ...). The runner stays ignorant of which surfaces are
attached; the GUI cluster monitor, a local run and the console all consume
the same state.

The progress state is a plain JSON-serializable dict so that any consumer
(the GUI polling over SSH, another process, a browser) can read it without
importing this package. Its shape::

    {
        "kind": "single" | "aggregation",
        "report_name": str | None,
        "state": "pending" | "running" | "done" | "failed" | "aborted",
        "rank": int, "size": int,
        "started": iso8601 | None, "updated": iso8601 | None,
        "current": int | None,          # index of the active module/dataset
        "total": int | None,            # number of modules (single only)
        "modules": [
            {"i": int, "name": str,
             "status": "pending"|"running"|"done"|"failed"|"skipped",
             "fraction": float | None,  # intra-module progress, 0..1
             "msg": str | None,
             "elapsed": float | None,   # seconds, set on completion
             "error": str | None},
            ...
        ],
        "datasets": [                    # aggregation only
            {"i": int, "tag": str, "state": ...},
            ...
        ],
    }
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

from loguru import logger

# --- module/run status constants --------------------------------------------
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"
ABORTED = "aborted"

# canonical filenames written into a run's result folder
PROGRESS_FILENAME = "progress.json"
ABORT_FILENAME = "abort.flag"


def _now_iso() -> str:
    """Return the current local time as an ISO-8601 string (second res.)."""
    return datetime.now().isoformat(timespec="seconds")


class JsonProgressSink:
    """Write the progress state to a JSON file, atomically.

    The file is written to a temporary path and then ``os.replace``-d into
    place so a concurrent reader (e.g. the GUI polling over SSH) never sees a
    half-written file.

    Parameters
    ----------
    path : str
        Destination path of the progress JSON file.
    """

    def __init__(self, path: str):
        self.path = path

    def __call__(self, state: dict) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, self.path)


class LogProgressSink:
    """Emit a concise progress line to the loguru logger.

    Useful on the cluster, where the SLURM ``.log`` file is the only surface;
    a monitor can ``grep`` for ``[progress]``.
    """

    def __call__(self, state: dict) -> None:
        cur = state.get("current")
        total = state.get("total")
        modules = state.get("modules") or []
        name = ""
        frac = None
        if cur is not None and 0 <= cur < len(modules):
            name = modules[cur].get("name", "")
            frac = modules[cur].get("fraction")
        pct = ""
        if isinstance(frac, (int, float)):
            pct = f" {frac * 100:.0f}%"
        pos = "-" if cur is None else f"{cur + 1}/{total}"
        logger.info(
            f"[progress] {state.get('state')} module {pos} {name}{pct}"
        )


def default_sinks(result_folder: str, rank: int = 0) -> list:
    """Build the default sink list for a run.

    Rank 0 writes the shared ``progress.json``; worker ranks write a
    per-rank file so they never race on the same path.

    Parameters
    ----------
    result_folder : str
        The run's result folder.
    rank : int, optional
        SLURM task rank. Default 0.

    Returns
    -------
    list
        A list of sink callables.
    """
    fname = PROGRESS_FILENAME if rank == 0 else f"progress.rank{rank}.json"
    return [
        JsonProgressSink(os.path.join(result_folder, fname)),
        LogProgressSink(),
    ]


class ProgressManager:
    """Track and broadcast the progress of a workflow run.

    The runner calls the ``module_*`` / ``dataset_*`` methods at the right
    points; each updates the in-memory state and (unless throttled) notifies
    every sink with the full state snapshot.

    Parameters
    ----------
    result_folder : str
        The run's result folder; the default JSON sink writes here.
    kind : str, optional
        ``"single"`` or ``"aggregation"``. Default ``"single"``.
    rank, size : int, optional
        SLURM task identity, recorded in the state and used to pick the JSON
        filename. Defaults 0 / 1.
    report_name : str, optional
        Human-readable run name, recorded in the state.
    sinks : list, optional
        Sink callables. If None, :func:`default_sinks` is used.
    throttle : float, optional
        Minimum seconds between two *intra-module* (fraction-only) emits, to
        avoid hammering the JSON file per movie frame. Status transitions
        always emit. Default 1.0.
    """

    def __init__(
        self,
        result_folder: str,
        kind: str = "single",
        *,
        rank: int = 0,
        size: int = 1,
        report_name: str | None = None,
        sinks: list | None = None,
        throttle: float = 1.0,
    ):
        self.result_folder = result_folder
        self.rank = rank
        self.size = size
        self._throttle = throttle
        self._module_t0: dict[int, float] = {}
        self._last_progress_emit = 0.0
        self._state = {
            "kind": kind,
            "report_name": report_name,
            "state": PENDING,
            "rank": rank,
            "size": size,
            "started": None,
            "updated": None,
            "current": None,
            "total": None,
            "modules": [],
            "datasets": [],
        }
        if sinks is None:
            sinks = default_sinks(result_folder, rank=rank)
        self.sinks = list(sinks)

    # -- broadcasting --------------------------------------------------------

    @property
    def state(self) -> dict:
        """The current progress state dict (live reference)."""
        return self._state

    def emit(self) -> None:
        """Push the current state to every sink (failures are non-fatal)."""
        self._state["updated"] = _now_iso()
        for sink in self.sinks:
            try:
                sink(self._state)
            except Exception as e:  # a broken sink must never abort a run
                logger.debug(f"progress sink {sink!r} failed: {e}")

    # -- single-dataset (module) tracking ------------------------------------

    def start(self, module_names: list) -> None:
        """Begin a single-dataset run over ``module_names``."""
        self._state["state"] = RUNNING
        self._state["started"] = _now_iso()
        self._state["total"] = len(module_names)
        self._state["modules"] = [
            {
                "i": i,
                "name": name,
                "status": PENDING,
                "fraction": None,
                "msg": None,
                "elapsed": None,
                "error": None,
            }
            for i, name in enumerate(module_names)
        ]
        self.emit()

    def module_skipped(self, i: int) -> None:
        """Mark module ``i`` as skipped (already succeeded previously)."""
        m = self._get(i)
        if m is not None:
            m["status"] = SKIPPED
            m["fraction"] = 1.0
        self._state["current"] = i
        self.emit()

    def module_start(self, i: int) -> None:
        """Mark module ``i`` as running and start its timer."""
        now = time.perf_counter()
        self._module_t0[i] = now
        # start the intra-module throttle window here so the first few
        # per-frame ticks are coalesced rather than each writing the file.
        self._last_progress_emit = now
        m = self._get(i)
        if m is not None:
            m["status"] = RUNNING
            m["fraction"] = 0.0
        self._state["current"] = i
        self.emit()

    def module_progress(
        self, i: int, fraction: float | None, msg: str | None = None
    ) -> None:
        """Report intra-module progress (throttled).

        Parameters
        ----------
        i : int
            Module index.
        fraction : float or None
            Completion fraction in ``[0, 1]`` (clamped). None leaves it.
        msg : str, optional
            Short status message (e.g. ``"frame 21000/50000"``).
        """
        m = self._get(i)
        if m is None:
            return
        if fraction is not None:
            try:
                m["fraction"] = max(0.0, min(1.0, float(fraction)))
            except (TypeError, ValueError):
                pass
        changed = False
        if msg is not None and msg != m.get("msg"):
            m["msg"] = str(msg)
            changed = True
        now = time.perf_counter()
        if changed or (now - self._last_progress_emit) >= self._throttle:
            self._last_progress_emit = now
            self.emit()

    def module_end(
        self, i: int, status: str, error: str | None = None
    ) -> None:
        """Mark module ``i`` as finished with ``status`` and record timing."""
        m = self._get(i)
        t0 = self._module_t0.pop(i, None)
        if m is not None:
            m["status"] = status
            if status == DONE:
                m["fraction"] = 1.0
            if t0 is not None:
                m["elapsed"] = round(time.perf_counter() - t0, 3)
            if error is not None:
                m["error"] = str(error)[:2000]
        self.emit()

    # -- aggregation (dataset) tracking --------------------------------------

    def datasets_init(self, tags: list) -> None:
        """Initialise the per-dataset list for an aggregation run."""
        self._state["datasets"] = [
            {"i": i, "tag": tag, "state": PENDING}
            for i, tag in enumerate(tags)
        ]
        self.emit()

    def dataset_update(self, i: int, state: str) -> None:
        """Set the state of single dataset ``i`` in an aggregation run."""
        for d in self._state["datasets"]:
            if d["i"] == i:
                d["state"] = state
                break
        self._state["current"] = i
        self.emit()

    def mark_running(self) -> None:
        """Mark the overall run as running (used by aggregation)."""
        self._state["state"] = RUNNING
        if self._state["started"] is None:
            self._state["started"] = _now_iso()
        self.emit()

    def finish(self, state: str) -> None:
        """Set the overall run state (``done`` / ``failed`` / ``aborted``)."""
        self._state["state"] = state
        self.emit()

    # -- helpers -------------------------------------------------------------

    def _get(self, i: int) -> dict | None:
        for m in self._state["modules"]:
            if m["i"] == i:
                return m
        return None


class PicassoProgressProxy:
    """Adapt picasso's two progress conventions onto a fraction callback.

    picasso long-running calls report progress in one of two ways:

    * a plain callable receiving an absolute count -- ``progress_callback``
      of :func:`picasso.localize.identify` / ``fit``, the
      ``segmentation_callback`` / ``rcc_callback`` of
      :func:`picasso.postprocess.undrift`;
    * an object with the ``ProgressDialog`` interface (``set_value``,
      ``setMaximum``, ``zero_progress``, ...) -- the ``progress`` argument of
      :func:`picasso.clusterer.cluster` and :func:`picasso.aim.aim`.

    This proxy satisfies *both*: it is callable and it implements the dialog
    interface. Every count is divided by ``total`` and forwarded to ``cb`` as
    a fraction in ``[0, 1]``. All methods swallow their own errors so a
    progress hiccup can never crash the analysis.

    Parameters
    ----------
    cb : callable
        ``cb(fraction, msg)`` -- typically
        ``ProgressManager.module_progress`` bound to a module index.
    total : int, optional
        The count that corresponds to 100 %. May be updated later via
        :meth:`setMaximum`.
    phase : str, optional
        A label forwarded as ``msg`` (e.g. ``"identify"``).
    """

    def __init__(self, cb, total: int | None = None, phase: str | None = None):
        self._cb = cb
        self._max = total or 0
        self._phase = phase

    def _report(self, value) -> None:
        try:
            if hasattr(value, "__len__") and not isinstance(
                value, (str, bytes)
            ):
                value = value[0] if len(value) else 0
            frac = (value / self._max) if self._max else None
            self._cb(frac, self._phase)
        except Exception:
            pass

    # plain-callable convention (identify / fit / undrift callbacks)
    def __call__(self, value=0, *args, **kwargs) -> None:
        self._report(value)

    # ProgressDialog convention (cluster / aim)
    def set_value(self, value=0, *args, **kwargs) -> None:
        self._report(value)

    def setMaximum(self, maximum, *args, **kwargs) -> None:
        if maximum:
            self._max = maximum

    def set_maximum(self, maximum, *args, **kwargs) -> None:
        self.setMaximum(maximum)

    def maximum(self) -> int:
        return self._max

    def zero_progress(self, description=None, *args, **kwargs) -> None:
        if description:
            self._phase = str(description)

    def init(self, *args, **kwargs) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        pass

    def close(self, *args, **kwargs) -> None:
        pass

    def closeEvent(self, *args, **kwargs) -> None:
        pass

    def setLabelText(self, *args, **kwargs) -> None:
        pass

    def play_sound_notification(self, *args, **kwargs) -> None:
        pass

    def get_iterator(self, start=0, end=100):
        return range(start, end)


# --- consumer-side helpers (used by the GUI / any reader) -------------------


def overall_fraction(state: dict) -> float:
    """Compute an overall 0..1 completion fraction from a progress state.

    Completed and skipped modules count as full; a running module counts by
    its intra-module fraction. Returns 0.0 for an empty/None state.
    """
    if not state:
        return 0.0
    modules = state.get("modules") or []
    if not modules:
        # aggregation runs track datasets rather than modules
        datasets = state.get("datasets") or []
        if not datasets:
            return 0.0
        done = sum(1 for d in datasets if d.get("state") in (DONE, SKIPPED))
        return done / len(datasets)
    total = 0.0
    for m in modules:
        if m.get("status") in (DONE, SKIPPED):
            total += 1.0
        elif m.get("status") == RUNNING:
            total += m.get("fraction") or 0.0
    return total / len(modules)


def read_progress(folder: str) -> dict | None:
    """Read ``progress.json`` directly under ``folder`` (or None)."""
    path = os.path.join(folder, PROGRESS_FILENAME)
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_latest_progress(folder: str) -> dict | None:
    """Read the most recently modified ``progress.json`` in ``folder``'s tree.

    A run writes its ``progress.json`` into a timestamped subfolder whose name
    is only known at runtime, so we locate it by walking the tree and picking
    the newest file. Returns None if none is found or it cannot be parsed.
    """
    latest_path = None
    latest_mtime = -1.0
    for root, _dirs, files in os.walk(folder):
        if PROGRESS_FILENAME in files:
            p = os.path.join(root, PROGRESS_FILENAME)
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = p
    if latest_path is None:
        return None
    try:
        with open(latest_path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_all_progress(folder: str) -> list:
    """Read every ``progress.json`` in ``folder``'s tree.

    An aggregation run produces several progress files -- the top-level
    aggregation state plus one per single-dataset stage and one for the
    aggregation stage -- so a monitor that wants to show all stages needs
    them all, not just the newest. Returns a list of parsed states (empty if
    none found); order follows :func:`os.walk`.

    Parameters
    ----------
    folder : str
        The run's results folder (walked recursively).

    Returns
    -------
    list of dict
    """
    states = []
    for root, _dirs, files in os.walk(folder):
        if PROGRESS_FILENAME in files:
            try:
                with open(os.path.join(root, PROGRESS_FILENAME)) as f:
                    states.append(json.load(f))
            except (OSError, ValueError):
                continue
    return states


def stage_name(state: dict) -> str:
    """A short human label for a stage, from its ``report_name``.

    Strips the trailing ``_yymmdd-HHMM`` run postfix. Returns ``"(run)"`` if
    there is no report name.
    """
    import re

    name = (state or {}).get("report_name") or ""
    name = re.sub(r"_\d{6}-\d{4}$", "", name)
    return name or "(run)"


def is_aggregation_stage(state: dict) -> bool:
    """Whether ``state`` is the final aggregation stage (not a single dataset).

    The aggregation-stage single workflow is named ``..._aggregation``; the
    per-dataset singles are named ``..._sgl_NN[_tag]``.
    """
    return stage_name(state).endswith("_aggregation")


def dataset_index(state: dict) -> int | None:
    """The dataset index parsed from a single stage's ``_sgl_NN`` name, or None.

    Used to align a single-dataset stage with its slot in the top-level
    aggregation ``datasets`` list.
    """
    import re

    m = re.search(r"_sgl_(\d+)", (state or {}).get("report_name") or "")
    return int(m.group(1)) if m else None


# --- cooperative abort ------------------------------------------------------


def request_abort(folder: str) -> None:
    """Ask a running workflow to stop by dropping an abort flag file.

    The runner checks :func:`abort_requested` between modules and via the
    picasso ``abort_callback`` inside long modules, so this yields a graceful
    stop at the next checkpoint (complementary to ``scancel``).
    """
    try:
        open(os.path.join(folder, ABORT_FILENAME), "w").close()
    except OSError as e:
        logger.warning(f"could not write abort flag in {folder}: {e}")


def abort_requested(folder: str) -> bool:
    """Whether an abort flag file exists in ``folder``."""
    return os.path.exists(os.path.join(folder, ABORT_FILENAME))


def clear_abort(folder: str) -> None:
    """Remove a stale abort flag from ``folder`` (no-op if absent)."""
    try:
        os.remove(os.path.join(folder, ABORT_FILENAME))
    except OSError:
        pass
