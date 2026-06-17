#!/usr/bin/env python
"""Tests for picasso_workflow.workflow_references.

Covers index-token remapping of ``$get_prior_result`` locators on
insertion/deletion/move, the same/cross-workflow sign handling, and the
consistency validator. Dependency-free (no picasso import).
"""

from picasso_workflow.workflow_references import (
    SINGLE,
    AGGREGATION,
    remap_references,
    update_after_index_change,
    insertion_index_map,
    deletion_index_map,
    move_index_map,
    validate_references,
)


def _prior(locator, sign=1):
    return ("$" * sign + "get_prior_result", locator)


# --------------------------------------------------------------------------
# locator remapping
# --------------------------------------------------------------------------


def test_insertion_shifts_following_reference():
    modules = [
        ("load", {}),
        ("find_gold", {}),
        ("use_gold", {"fp": _prior("results, 01_find_gold, fp")}),
    ]
    # insert before index 1 -> find_gold moves 01 -> 02
    imap = insertion_index_map(1, len(modules))
    new, changes = remap_references(modules, 1, imap)
    assert new[2][1]["fp"] == _prior("results, 02_find_gold, fp")
    assert changes == [
        ("results, 01_find_gold, fp", "results, 02_find_gold, fp")
    ]


def test_reference_before_insertion_point_unchanged():
    modules = [
        ("find_gold", {}),
        ("use_gold", {"fp": _prior("results, 00_find_gold, fp")}),
    ]
    # insert at index 1 (after find_gold); 00_find_gold must stay 00
    imap = insertion_index_map(1, len(modules))
    new, changes = remap_references(modules, 1, imap)
    assert new[1][1]["fp"] == _prior("results, 00_find_gold, fp")
    assert changes == []


def test_deletion_shifts_following_reference_down():
    modules = [
        ("a", {}),
        ("b", {}),
        ("find_gold", {}),
        ("use", {"fp": _prior("results, 02_find_gold, fp")}),
    ]
    imap = deletion_index_map(0, len(modules))  # delete index 0
    new, _ = remap_references(modules, 1, imap)
    assert new[3][1]["fp"] == _prior("results, 01_find_gold, fp")


def test_move_remaps_reference():
    modules = [
        ("find_gold", {}),
        ("mid", {}),
        ("use", {"fp": _prior("results, 00_find_gold, fp")}),
    ]
    # move find_gold from 0 to 1
    imap = move_index_map(0, 1, len(modules))
    new, _ = remap_references(modules, 1, imap)
    assert new[2][1]["fp"] == _prior("results, 01_find_gold, fp")


# --------------------------------------------------------------------------
# sign / cross-workflow handling
# --------------------------------------------------------------------------


def test_cross_workflow_double_sign_updated_on_single_change():
    single = [("identify", {}), ("save_single_dataset", {})]
    agg = [
        (
            "load_to_aggregate",
            {
                "fp": _prior(
                    "all_results, single_dataset, $$all, "
                    "01_save_single_dataset, filepath",
                    sign=2,
                )
            },
        )
    ]
    # insert a single-workflow module at index 0 -> save moves 01 -> 02
    imap = insertion_index_map(0, len(single))
    new_single, new_agg, changes = update_after_index_change(
        single, agg, SINGLE, imap
    )
    assert "02_save_single_dataset" in new_agg[0][1]["fp"][1]
    assert len(changes) == 1


def test_single_sign_not_touched_by_cross_remap():
    # an aggregation-internal ($) ref must not move when single changes
    agg = [("a", {}), ("b", {"fp": _prior("results, 00_a, x", sign=1)})]
    single = [("s", {})]
    imap = insertion_index_map(0, len(single))
    _, new_agg, changes = update_after_index_change(single, agg, SINGLE, imap)
    assert new_agg[1][1]["fp"] == _prior("results, 00_a, x", sign=1)
    assert changes == []


def test_aggregation_internal_reference_remapped_on_agg_change():
    agg = [
        ("a", {}),
        ("b", {}),
        ("c", {"fp": _prior("results, 01_b, x", sign=1)}),
    ]
    imap = insertion_index_map(1, len(agg))  # insert before b
    _, new_agg, _ = update_after_index_change([], agg, AGGREGATION, imap)
    assert new_agg[2][1]["fp"] == _prior("results, 02_b, x", sign=1)


# --------------------------------------------------------------------------
# nesting
# --------------------------------------------------------------------------


def test_nested_command_remapped():
    modules = [
        ("nn", {}),
        (
            "use",
            {
                "v": (
                    "$sum *0.8",
                    _prior("results, 00_nn, density"),
                    _prior("results, 00_nn, density2"),
                )
            },
        ),
    ]
    imap = insertion_index_map(0, len(modules))
    new, changes = remap_references(modules, 1, imap)
    inner = new[1][1]["v"]
    assert inner[1] == _prior("results, 01_nn, density")
    assert inner[2] == _prior("results, 01_nn, density2")
    assert len(changes) == 2


def test_previous_module_result_never_rewritten():
    modules = [
        ("a", {}),
        ("b", {"fp": ("$get_previous_module_result", "filepaths")}),
    ]
    imap = insertion_index_map(0, len(modules))
    new, changes = remap_references(modules, 1, imap)
    assert new[1][1]["fp"] == ("$get_previous_module_result", "filepaths")
    assert changes == []


def test_non_reference_tuples_left_alone():
    modules = [("a", {"size": (160, 160), "name": ("x", "y")})]
    imap = insertion_index_map(0, 1)
    new, changes = remap_references(modules, 1, imap)
    assert new[0][1]["size"] == (160, 160)
    assert changes == []


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_validate_passes_on_consistent_workflow():
    single = [
        ("load", {}),
        ("find_gold", {}),
        ("use", {"fp": _prior("results, 01_find_gold, fp")}),
    ]
    assert validate_references(single, []) == []


def test_validate_flags_wrong_name():
    single = [
        ("load", {}),
        ("find_gold", {}),
        ("use", {"fp": _prior("results, 00_find_gold, fp")}),  # 00 is load
    ]
    errors = validate_references(single, [])
    assert len(errors) == 1
    assert "00_find_gold" in errors[0]
    assert "load" in errors[0]


def test_validate_flags_out_of_range():
    single = [
        ("find_gold", {}),
        ("use", {"fp": _prior("results, 05_find_gold, fp")}),
    ]
    errors = validate_references(single, [])
    assert len(errors) == 1
    assert "no module at index" in errors[0]


def test_validate_flags_broken_cross_reference():
    single = [("identify", {}), ("save_single_dataset", {})]
    agg = [
        (
            "load_to_aggregate",
            {
                "fp": _prior(
                    "all_results, single_dataset, $$all, "
                    "00_save_single_dataset, filepath",
                    sign=2,
                )
            },
        )
    ]
    # 00 is 'identify', not 'save_single_dataset'
    errors = validate_references(single, agg)
    assert len(errors) == 1
    assert "aggregation module" in errors[0]
    assert "single module 00" in errors[0]
