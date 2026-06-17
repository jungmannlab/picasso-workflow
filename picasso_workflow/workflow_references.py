#!/usr/bin/env python
"""
Module Name: workflow_references.py
Author: Heinrich Grabmayr
Initial Date: June 17, 2026
Description:
    Helpers to keep ``$get_prior_result`` back-references consistent when the
    module order of a workflow changes.

    Workflow modules can reference the result of an earlier module by an
    index-prefixed locator, e.g.::

        ("$get_prior_result", "results, 09_find_gold, fp")

    The ``09_`` is the *position* of the referenced module. Inserting,
    removing or moving a module renumbers the positions, so a locator like
    ``09_find_gold`` silently points at the wrong (or a non-existent) module.

    A single ``$`` references a module in the *same* workflow; a double ``$$``
    references a single-dataset-workflow module from the aggregation workflow
    (``all_results, single_dataset, ...``). ``$get_previous_module_result`` is
    position-relative and therefore never needs rewriting.

    This module is intentionally dependency-free (only the standard library)
    so it can be imported by the GUI and unit-tested without pulling in
    ``picasso``.
"""

from __future__ import annotations

import re

# A locator segment that names a module by position, e.g. "09_find_gold".
# The index is zero-padded to (at least) two digits by the writers.
_MODULE_TOKEN_RE = re.compile(r"^(\d{2,})_(\S+)$")

SINGLE = "single"
AGGREGATION = "aggregation"


def _command_sign_and_name(cmd):
    """Return ``(sign_count, command_name)`` for a command string.

    ``sign_count`` is the number of leading ``$`` (1 for same-workflow, 2 for
    a cross-workflow reference); ``command_name`` is the command without the
    sign or any trailing arithmetic (e.g. ``"get_prior_result"``). Returns
    ``(0, "")`` for anything that is not a command string.

    Parameters
    ----------
    cmd : object
        The candidate command (only ``str`` starting with ``$`` qualifies).

    Returns
    -------
    tuple of (int, str)
        The sign count and the bare command name.
    """
    if not isinstance(cmd, str) or not cmd.startswith("$"):
        return 0, ""
    stripped = cmd.lstrip("$")
    sign = len(cmd) - len(stripped)
    name = stripped.split(" ")[0]
    return sign, name


def _is_prior_result(value):
    """Return ``(sign, locator)`` if ``value`` is a get_prior_result command.

    Parameters
    ----------
    value : object
        A candidate parameter value (a command tuple/list).

    Returns
    -------
    tuple of (int, str) or None
        ``(sign, locator)`` for a ``get_prior_result`` command with a string
        locator, otherwise ``None``.
    """
    if (
        isinstance(value, (tuple, list))
        and len(value) >= 2
        and isinstance(value[1], str)
    ):
        sign, name = _command_sign_and_name(value[0])
        if name == "get_prior_result":
            return sign, value[1]
    return None


def _remap_locator(locator, index_map):
    """Rewrite module-position tokens in a comma-separated locator.

    Parameters
    ----------
    locator : str
        A locator such as ``"results, 09_find_gold, fp"``.
    index_map : dict of int to int
        Mapping of old module index to new module index. Tokens whose index
        is absent (or maps to itself) are left unchanged.

    Returns
    -------
    tuple of (str, bool)
        The (possibly) rewritten locator and whether anything changed.
    """
    segments = [s.strip() for s in locator.split(",")]
    changed = False
    for k, seg in enumerate(segments):
        m = _MODULE_TOKEN_RE.match(seg)
        if not m:
            continue
        idx = int(m.group(1))
        new_idx = index_map.get(idx)
        if new_idx is not None and new_idx != idx:
            segments[k] = f"{new_idx:02d}_{m.group(2)}"
            changed = True
    return (", ".join(segments), True) if changed else (locator, False)


def _transform(value, sign_to_match, index_map, changes):
    """Recursively rewrite matching get_prior_result locators in ``value``.

    Parameters
    ----------
    value : object
        A parameter value (possibly a nested tuple/list/dict of commands).
    sign_to_match : int
        Only rewrite references whose sign equals this (1 or 2).
    index_map : dict of int to int
        Old-to-new module index mapping.
    changes : list
        Accumulator of ``(old_locator, new_locator)`` pairs that were changed.

    Returns
    -------
    object
        A value of the same shape with matching locators rewritten.
    """
    prior = _is_prior_result(value)
    if prior is not None:
        sign, locator = prior
        items = list(value)
        if sign == sign_to_match:
            new_loc, changed = _remap_locator(locator, index_map)
            if changed:
                items[1] = new_loc
                changes.append((locator, new_loc))
        # Recurse into any further (rare) arguments.
        for j in range(2, len(items)):
            items[j] = _transform(items[j], sign_to_match, index_map, changes)
        return type(value)(items)
    if isinstance(value, (tuple, list)):
        return type(value)(
            _transform(v, sign_to_match, index_map, changes) for v in value
        )
    if isinstance(value, dict):
        return {
            k: _transform(v, sign_to_match, index_map, changes)
            for k, v in value.items()
        }
    return value


def remap_references(modules, sign, index_map):
    """Rewrite get_prior_result locators of one sign across modules.

    Parameters
    ----------
    modules : list of (str, object)
        ``(module_name, parameters)`` tuples.
    sign : int
        Which references to rewrite: 1 (same-workflow) or 2 (cross-workflow).
    index_map : dict of int to int
        Old-to-new module index mapping.

    Returns
    -------
    tuple of (list, list)
        The new modules list and the list of ``(old_locator, new_locator)``
        changes that were made.
    """
    new_modules = []
    changes = []
    for name, params in modules:
        new_modules.append(
            (name, _transform(params, sign, index_map, changes))
        )
    return new_modules, changes


def update_after_index_change(
    single_modules, aggregation_modules, scope, index_map
):
    """Remap references after a workflow's module indices changed.

    When the *single* workflow changes, both its own (``$``) references and
    the aggregation workflow's cross-references (``$$``) into it are updated.
    When the *aggregation* workflow changes, only its own (``$``) references
    are updated.

    Parameters
    ----------
    single_modules, aggregation_modules : list of (str, object)
        The two workflows' module lists.
    scope : str
        Which workflow changed: :data:`SINGLE` or :data:`AGGREGATION`.
    index_map : dict of int to int
        Old-to-new module index mapping for the changed workflow.

    Returns
    -------
    tuple of (list, list, list)
        ``(new_single_modules, new_aggregation_modules, changes)``.
    """
    changes = []
    if scope == SINGLE:
        single_modules, c1 = remap_references(single_modules, 1, index_map)
        aggregation_modules, c2 = remap_references(
            aggregation_modules, 2, index_map
        )
        changes = c1 + c2
    elif scope == AGGREGATION:
        aggregation_modules, c1 = remap_references(
            aggregation_modules, 1, index_map
        )
        changes = c1
    return single_modules, aggregation_modules, changes


def insertion_index_map(insert_idx, old_len):
    """Index map for inserting one module at ``insert_idx``.

    Every module previously at ``insert_idx`` or later shifts up by one.
    """
    return {i: i + 1 for i in range(insert_idx, old_len)}


def deletion_index_map(deleted_idx, old_len):
    """Index map for deleting the module at ``deleted_idx``.

    Modules after the deleted one shift down by one. The deleted index is
    intentionally absent from the map, so references *to it* are left
    untouched and surface in :func:`validate_references`.
    """
    return {i: i - 1 for i in range(deleted_idx + 1, old_len)}


def move_index_map(from_idx, to_idx, length):
    """Index map for moving a module from ``from_idx`` to ``to_idx``."""
    order = list(range(length))
    order.insert(to_idx, order.pop(from_idx))
    return {old: new for new, old in enumerate(order)}


def _collect(value, out):
    """Collect ``(sign, locator)`` for every get_prior_result in ``value``."""
    prior = _is_prior_result(value)
    if prior is not None:
        out.append(prior)
        for j in range(2, len(value)):
            _collect(value[j], out)
        return
    if isinstance(value, (tuple, list)):
        for v in value:
            _collect(v, out)
    elif isinstance(value, dict):
        for v in value.values():
            _collect(v, out)


def validate_references(single_modules, aggregation_modules):
    """Check that every get_prior_result reference resolves to its module.

    Parameters
    ----------
    single_modules, aggregation_modules : list of (str, object)
        The two workflows' ``(module_name, parameters)`` lists.

    Returns
    -------
    list of str
        One human-readable message per broken reference. Empty when all
        references resolve.
    """
    single_names = [m[0] for m in single_modules]
    aggregation_names = [m[0] for m in aggregation_modules]
    errors = []

    def check(modules, own_names, own_label):
        for midx, (mname, params) in enumerate(modules):
            refs = []
            _collect(params, refs)
            for sign, locator in refs:
                for seg in (s.strip() for s in locator.split(",")):
                    m = _MODULE_TOKEN_RE.match(seg)
                    if not m:
                        continue
                    tidx, tname = int(m.group(1)), m.group(2)
                    if sign >= 2:
                        target_names, target_label = single_names, "single"
                    else:
                        target_names, target_label = own_names, own_label
                    if tidx >= len(target_names):
                        errors.append(
                            f"{own_label} module {midx:02d}_{mname}: "
                            f"reference '{locator}' points to {seg}, but the "
                            f"{target_label} workflow has no module at index "
                            f"{tidx:02d}."
                        )
                    elif target_names[tidx] != tname:
                        errors.append(
                            f"{own_label} module {midx:02d}_{mname}: "
                            f"reference '{locator}' points to {seg}, but "
                            f"{target_label} module {tidx:02d} is "
                            f"'{target_names[tidx]}'."
                        )

    check(single_modules, single_names, "single")
    check(aggregation_modules, aggregation_names, "aggregation")
    return errors
