#!/usr/bin/env python
"""GUI parameter/result schemas and docstrings for the picasso-set modules.

One entry per picasso-set module, keyed by module name:

* :data:`PICASSO_SET_PARAMS` -- ``(parameters_spec, results_spec)`` tuples in
  the exact dict shape the GUI's ``ModuleDescriptor`` methods return (see
  ``gui.py``), consumed via ``ModuleSpec.params``.
* :data:`PICASSO_SET_SUMMARIES` -- one-line summaries, reused by the
  ``modulespec`` registry entries so there is a single source of truth.
* :func:`docstring` -- renders the GUI hover/help docstring from the above.

Parameter names and defaults replicate the picasso CLI
(``picasso/__main__.py``) / library signatures 1:1; any deliberate deviation
is called out in the parameter description.

This module is dependency-free on purpose (plain dicts only): it is imported
by :mod:`picasso_workflow.modulespec`, which must stay importable without the
analysis or GUI stacks.

Author: Heinrich Grabmayr
Initial date: October 5, 2026
"""

from __future__ import annotations

# Result keys every module gets from the module decorator.
_RESULTS_COMMON: dict = {
    "folder": {
        "type": "str",
        "description": "Output folder for module results",
    },
    "start time": {
        "type": "str",
        "description": "Module execution start timestamp",
    },
    "end time": {
        "type": "str",
        "description": "Module execution end timestamp",
    },
    "duration": {
        "type": "float",
        "description": "Module execution duration in seconds",
        "min": 0.0,
    },
}


def _results(**extra) -> dict:
    """Return a results_spec: the common decorator keys plus ``extra``."""
    spec = dict(_RESULTS_COMMON)
    spec.update(extra)
    return spec


PICASSO_SET_SUMMARIES: dict[str, str] = {
    "picasso_density": (
        "Compute the local localization density "
        "(native picasso CLI: density)."
    ),
}


PICASSO_SET_PARAMS: dict[str, tuple[dict, dict]] = {
    "picasso_density": (
        {
            "radius": {
                "type": "float",
                "description": (
                    "Maximum distance (camera pixels) for localizations "
                    "to count as local (CLI: radius)"
                ),
                "min": 0.0,
                "required": True,
            },
        },
        _results(
            nlocs={
                "type": "int",
                "description": "Number of localizations processed",
                "min": 0,
            },
            filepath_locs_density={
                "type": "str",
                "description": (
                    "Saved density-annotated localizations "
                    "(mirrors the CLI's _density.hdf5 output)"
                ),
            },
        ),
    ),
}


def docstring(name: str) -> str:
    """Render the GUI docstring for a picasso-set module.

    Parameters
    ----------
    name : str
        The picasso-set module name (e.g. ``"picasso_density"``).

    Returns
    -------
    str
        A NumPy-style docstring built from the summary and parameter spec.
    """
    parameters_spec, _ = PICASSO_SET_PARAMS[name]
    lines = [
        PICASSO_SET_SUMMARIES[name],
        "",
        "Parameters",
        "----------",
    ]
    if not parameters_spec:
        lines.append("(none)")
    for pname, pspec in parameters_spec.items():
        required = "required" if pspec.get("required") else "optional"
        default = pspec.get("default")
        default_txt = "" if default is None else f", default: {default!r}"
        lines.append(f"{pname} : {pspec.get('type', 'any')}")
        lines.append(f"    {pspec.get('description', '')}")
        lines.append(f"    ({required}{default_txt})")
    return "\n".join(lines)
