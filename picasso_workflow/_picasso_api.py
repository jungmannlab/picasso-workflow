#!/usr/bin/env python
"""Single declared boundary between picasso-workflow and picasso.

picasso-workflow builds on the ``picassosr`` library, but only touches it at a
small number of stable seams (``io``, ``localize``, ``registration`` ...).  This
module is the one place that records *which* parts of picasso we rely on, so the
coupling is visible and testable instead of scattered across the package.

Two things live here:

* **The contract manifest** (:data:`PUBLIC_SUBMODULES`, :data:`TOP_LEVEL_ATTRS`,
  :data:`SUBMODULE_ATTRS`) -- the public picasso surface picasso-workflow uses.
  ``tests/test_picasso_contract.py`` asserts every entry still exists in the
  installed picassosr, so an upstream rename or removal fails loudly in CI
  instead of silently at runtime on the cluster.
* **Guarded accessors** for the handful of *private* picasso symbols we are
  currently forced to reach into (leading underscore, no public alternative
  yet).  Each raises a clear, actionable error naming the upstream symbol if it
  disappears, and is the single call site to update when picasso exposes a
  public API.  :data:`INTERNAL_SYMBOLS` tracks them so the set can shrink as
  they are upstreamed.

New picasso usage should be recorded here (and in the contract test) rather than
imported ad hoc elsewhere, so the boundary stays a single reviewable surface.
"""

from __future__ import annotations

# --- Contract manifest: the public picasso surface we depend on -------------

#: picasso submodules imported (as whole modules) somewhere in
#: picasso-workflow.  The contract test imports each one; a missing or renamed
#: submodule fails there.
PUBLIC_SUBMODULES = (
    "io",
    "localize",
    "render",
    "lib",
    "postprocess",
    "imageprocess",
    "spinna",
    "aim",
    "clusterer",
    "g5m",
    "zfit",
    "registration",
    "scmos",
    "spline",
    "masking",
    "plugins",
)

#: Attributes picasso-workflow reads off the top-level ``picasso`` package.
TOP_LEVEL_ATTRS = ("CONFIG", "__version__")

#: Attributes picasso-workflow reads off a specific submodule, beyond the
#: module merely being importable.
SUBMODULE_ATTRS = {
    "registration": ("tform",),
}

#: Picasso functions whose *parameter names* picasso-workflow passes (by
#: keyword or position).  A silent upstream rename of one of these would break
#: the call at runtime; the contract test asserts each listed parameter still
#: exists in the installed picassosr.  Keys are ``<submodule>.<function>``.
SIGNATURE_CONTRACT = {
    "io.save_locs": ("path", "locs", "info"),
    "io.load_locs": ("path",),
    "io.load_movie": ("path",),
    "postprocess.undrift": ("locs", "info", "segmentation"),
}

#: Private picasso symbols picasso-workflow currently depends on, mapped to
#: their upstream location.  These are the fragile seams -- leading underscore,
#: no public API yet.  Each is reached only through an accessor below; track
#: upstreaming so this set can shrink.
INTERNAL_SYMBOLS = {
    "_FIT_METHOD_MAP": "picasso.__main__._FIT_METHOD_MAP",
    "_spinna_batch_analysis": "picasso.__main__._spinna_batch_analysis",
}


def _missing(symbol: str) -> RuntimeError:
    """Build the error raised when a required private symbol is absent."""
    return RuntimeError(
        f"picasso-workflow requires {symbol}, which is not present in the "
        "installed picassosr. This is a private picasso symbol the workflow "
        "reaches into, so a picasso upgrade may have moved or renamed it. Pin "
        "picassosr to a compatible version (see pyproject.toml) or update the "
        "accessor in picasso_workflow/_picasso_api.py."
    )


def get_fit_method_map() -> dict:
    """Return picasso's ``_FIT_METHOD_MAP`` (fit-method name -> internal key).

    Guards the private ``picasso.__main__._FIT_METHOD_MAP`` that the
    localisation core uses to translate a configured fit-method string.
    """
    try:
        from picasso.__main__ import _FIT_METHOD_MAP
    except ImportError as exc:
        raise _missing("picasso.__main__._FIT_METHOD_MAP") from exc
    return _FIT_METHOD_MAP


def get_spinna_batch_analysis():
    """Return picasso's private ``_spinna_batch_analysis`` callable.

    Guards ``picasso.__main__._spinna_batch_analysis`` used by the SPINNA
    aggregation path in :mod:`picasso_workflow.picasso_outpost`.
    """
    try:
        from picasso.__main__ import _spinna_batch_analysis
    except ImportError as exc:
        raise _missing("picasso.__main__._spinna_batch_analysis") from exc
    return _spinna_batch_analysis
