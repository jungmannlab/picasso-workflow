#!/usr/bin/env python
"""Thin, best-effort client wrapper for the picasso-registry service.

picasso-registry is the provenance/metrics database for the DNA-PAINT
automation stack (FastAPI + SQLAlchemy; everything joined by ``run_id``). A
run logs **per-module** metrics + artifact pointers there as it executes, so
QC, the recommender and search can query the corpus later.

Design constraints this wrapper exists to enforce
-------------------------------------------------
* **Best-effort / non-blocking.** A registry outage (service down, network
  error, bad response) must *never* break an analysis run. Every call here
  swallows and logs its exception and returns ``None`` instead of raising
  into the recipe.
* **Injectable.** The real client (``picasso_registry.RegistryClient``, over
  ``requests``) and the in-memory test double
  (``picasso_registry.testing.MockRegistryClient``) share one method surface
  (``_BaseRegistry``: ``log_metrics`` / ``log_analysis`` / ``create`` ...), so
  tests pass a lightweight double in and keep unit tests hermetic -- no
  network, no FastAPI, no running service.
* **Keyed by ``run_id``.** Every metrics row and artifact pointer carries the
  analysis-run id so the append-only store can join them back to the run.

The wrapper is deliberately tiny: it owns the "never raise" policy and the
``run_id`` plumbing, and delegates the actual HTTP/DB work to the injected
client. It does not try to mirror the full registry API.
"""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger


class RegistryLogger:
    """Best-effort, non-blocking per-module logger to picasso-registry.

    Wraps an injected registry *client* (anything exposing the
    ``picasso_registry`` ``_BaseRegistry`` surface -- the real
    ``RegistryClient`` or the test ``MockRegistryClient``) and routes
    per-module metrics + artifact pointers to it, keyed by ``run_id``.

    Every public method catches and logs any exception from the underlying
    client and returns ``None`` instead of propagating it, so a registry
    problem can never abort a run.

    Parameters
    ----------
    run_id : str
        The acquisition/analysis run id every record is keyed by (the
        ``analysis_run_id`` the registry joins on). Falsy ``run_id`` disables
        logging (``enabled`` is False) -- a run without a registry run id
        simply does not log, rather than guessing one.
    client : object, optional
        A registry client (real or mock). If omitted, ``client_factory`` is
        called lazily on first use to build one; if neither is given logging
        is disabled.
    client_factory : callable, optional
        A zero-argument callable returning a client, used to defer (and make
        failure-tolerant) construction of the real HTTP client. Any error it
        raises disables logging for the run.
    """

    def __init__(
        self,
        run_id: str | None,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.run_id = run_id or None
        self._client = client
        self._client_factory = client_factory
        self._client_resolved = client is not None
        # A run with no run_id (or no way to get a client) cannot log; record
        # that up front so call sites can cheaply skip the work.
        self.enabled = bool(self.run_id) and (
            client is not None or client_factory is not None
        )

    # -- client resolution --------------------------------------------------
    def _get_client(self) -> Any | None:
        """Return the registry client, building it once if a factory was given.

        Best-effort: a factory that raises disables logging for the rest of
        the run (``enabled`` flips False) instead of raising into the caller.
        """
        if self._client_resolved:
            return self._client
        self._client_resolved = True
        try:
            self._client = self._client_factory()
        except Exception as e:  # never let client construction break a run
            logger.warning(
                f"registry: could not construct client ({e!r}); "
                "per-module logging disabled for this run."
            )
            self._client = None
            self.enabled = False
        return self._client

    # -- logging surface ----------------------------------------------------
    def log_module_metrics(
        self,
        module: str,
        index: int,
        metrics: dict | None = None,
        *,
        scope: str = "single",
    ) -> dict | None:
        """POST one metrics record for a module (best-effort, non-blocking).

        Exactly one record per module call, keyed by ``analysis_run_id`` =
        ``run_id``. The ``module`` name and its workflow ``index`` ride along
        (the ``Metrics`` schema allows extra fields, folded into its ``extra``
        JSON column), together with whatever numeric metrics the module
        produced. Returns the server's response dict, or ``None`` if logging
        is disabled or the POST failed.

        Parameters
        ----------
        module : str
            The module name (e.g. ``"localize"``).
        index : int
            The module's position in the workflow.
        metrics : dict, optional
            Numeric metric fields to record (schema keys like ``n_locs`` /
            ``nena_nm``; unknown keys ride through into ``extra``).
        scope : str, optional
            ``"single"`` or ``"aggregation"``. Default ``"single"``.
        """
        if not self.enabled:
            return None
        client = self._get_client()
        if client is None:
            return None
        payload: dict[str, Any] = {
            "analysis_run_id": self.run_id,
            "module": module,
            "module_index": index,
            "scope": scope,
        }
        if metrics:
            payload.update(metrics)
        try:
            return client.log_metrics(**payload)
        except Exception as e:  # registry must never break a run
            logger.warning(
                f"registry: failed to log metrics for "
                f"[{index:02d}] {module} (run {self.run_id}): {e!r}"
            )
            return None

    def log_artifact(
        self,
        module: str,
        kind: str,
        uri: str,
        *,
        checksum: str | None = None,
        size_bytes: int | None = None,
    ) -> dict | None:
        """POST one artifact pointer for a module (best-effort, non-blocking).

        Records where a produced artifact lives (a locs HDF5, a render PNG, a
        report) keyed by ``analysis_run_id`` = ``run_id``, so the store holds
        a pointer, not the bytes. Returns the server's response, or ``None``
        if disabled or the POST failed.

        Parameters
        ----------
        module : str
            The producing module name (recorded into the artifact's ``extra``).
        kind : str
            Artifact kind (e.g. ``"locs_hdf5"``, ``"render_png"``).
        uri : str
            Where the artifact lives (a path or URL).
        checksum, size_bytes : optional
            Integrity/size metadata when available.
        """
        if not self.enabled:
            return None
        client = self._get_client()
        if client is None:
            return None
        payload: dict[str, Any] = {
            "analysis_run_id": self.run_id,
            "kind": kind,
            "uri": uri,
            "module": module,
        }
        if checksum is not None:
            payload["checksum"] = checksum
        if size_bytes is not None:
            payload["size_bytes"] = size_bytes
        try:
            return client.create("artifact", **payload)
        except Exception as e:  # registry must never break a run
            logger.warning(
                f"registry: failed to log artifact {kind!r} for "
                f"{module} (run {self.run_id}): {e!r}"
            )
            return None


def default_client_factory(
    base_url: str, token: str | None = None
) -> Callable[[], Any]:
    """Return a factory that builds the real ``RegistryClient`` lazily.

    Kept separate (and lazy) so importing this module never imports the HTTP
    stack, and so a run only pays for / risks client construction if a
    ``base_url`` was actually configured.

    Parameters
    ----------
    base_url : str
        The registry service base URL.
    token : str, optional
        An optional static bearer token (ADR 001 / C18).

    Returns
    -------
    callable
        A zero-argument callable returning a ``RegistryClient``.
    """

    def _factory() -> Any:
        from picasso_registry.client import RegistryClient

        return RegistryClient(base_url=base_url, token=token)

    return _factory


def registry_logger_from_config(
    analysis_config: dict | None,
) -> RegistryLogger:
    """Build a :class:`RegistryLogger` from a runner's ``analysis_config``.

    Reads an optional ``registry`` sub-config:

    - ``run_id`` -- the analysis-run id every record is keyed by. Without it
      logging is disabled (nothing to join on).
    - ``client`` -- a pre-built client object (real or mock). Tests inject
      their in-memory double here, keeping the unit suite hermetic.
    - ``base_url`` (+ optional ``token``) -- used to lazily build the real
      HTTP client when no ``client`` object is supplied.

    An absent/empty ``registry`` config yields a disabled logger (the default
    hermetic path): modules run exactly as before and nothing is POSTed.

    Parameters
    ----------
    analysis_config : dict or None
        The runner's analysis configuration.

    Returns
    -------
    RegistryLogger
    """
    cfg = (analysis_config or {}).get("registry") or {}
    run_id = cfg.get("run_id")
    client = cfg.get("client")
    factory = None
    if client is None and cfg.get("base_url"):
        factory = default_client_factory(cfg["base_url"], cfg.get("token"))
    return RegistryLogger(run_id, client=client, client_factory=factory)
