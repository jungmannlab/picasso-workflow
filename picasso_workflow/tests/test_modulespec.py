#!/usr/bin/env python
"""
Module Name: test_modulespec.py
Author: Heinrich Grabmayr
Initial Date: June 15, 2026
Description: Reconcile MODULE_REGISTRY against the AbstractModuleCollection
    contract and check capability-vocabulary closure.
"""

import inspect
import unittest

from picasso_workflow.modulespec import (
    CAPABILITIES,
    MODULE_REGISTRY,
    ModuleSpec,
    PicassoRelation,
    Scope,
    validate_workflow,
)
from picasso_workflow.util import AbstractModuleCollection


def _contract_module_names():
    """The module names the runner dispatches on.

    Mirrors the filter applied in :meth:`WorkflowRunner.run` exactly, so this
    test tracks whatever the runner considers a module.
    """
    members = inspect.getmembers(AbstractModuleCollection)
    return {
        name
        for name, obj in members
        if (inspect.isfunction(obj) or inspect.ismethod(obj))
        and not name.startswith("__")
    }


class TestModuleSpecRegistry(unittest.TestCase):
    def test_every_module_has_a_spec_and_no_orphans(self):
        contract = _contract_module_names()
        specced = set(MODULE_REGISTRY)
        missing = contract - specced  # contract module without a ModuleSpec
        orphan = specced - contract  # spec for a non-existent module
        self.assertEqual(
            set(), missing, f"modules without a ModuleSpec: {sorted(missing)}"
        )
        self.assertEqual(
            set(), orphan, f"specs with no matching module: {sorted(orphan)}"
        )

    def test_registry_key_matches_spec_name(self):
        for key, spec in MODULE_REGISTRY.items():
            self.assertEqual(key, spec.name)

    def test_vocabulary_closure(self):
        for spec in MODULE_REGISTRY.values():
            used = spec.requires | spec.provides | spec.optional
            unknown = used - CAPABILITIES
            self.assertEqual(
                set(),
                unknown,
                f"{spec.name}: tokens outside vocabulary: {sorted(unknown)}",
            )

    def test_scopes_non_empty_and_valid(self):
        for spec in MODULE_REGISTRY.values():
            self.assertTrue(spec.scopes, f"{spec.name}: empty scopes")
            for s in spec.scopes:
                self.assertIsInstance(s, Scope)

    def test_wraps_modules_record_a_symbol(self):
        for spec in MODULE_REGISTRY.values():
            if spec.relation is PicassoRelation.WRAPS:
                self.assertTrue(
                    spec.picasso_symbol,
                    f"{spec.name}: WRAPS module must record a picasso_symbol",
                )


class TestModuleSpecValidation(unittest.TestCase):
    def test_unknown_token_rejected(self):
        with self.assertRaises(ValueError):
            ModuleSpec(name="x", requires=frozenset({"not_a_capability"}))

    def test_empty_scopes_rejected(self):
        with self.assertRaises(ValueError):
            ModuleSpec(name="x", scopes=frozenset())

    def test_wraps_without_symbol_rejected(self):
        with self.assertRaises(ValueError):
            ModuleSpec(name="x", relation=PicassoRelation.WRAPS)


class TestValidateWorkflow(unittest.TestCase):
    def test_golden_single_workflow_passes(self):
        # Mirrors the structure of standard_singledataset_workflows.minimal.
        steps = [
            ("load_dataset_movie", {}),
            ("identify", {}),
            ("localize", {}),
            ("undrift_rcc", {}),
            ("save_single_dataset", {}),
        ]
        self.assertEqual([], validate_workflow(steps, Scope.SINGLE))

    def test_golden_aggregation_workflow_passes(self):
        steps = [
            ("load_datasets_to_aggregate", {}),
            ("align_channels", {}),
            ("save_datasets_aggregated", {}),
        ]
        self.assertEqual([], validate_workflow(steps, Scope.AGGREGATION))

    def test_accepts_scope_as_string(self):
        steps = [("load_dataset_movie", {}), ("identify", {})]
        self.assertEqual([], validate_workflow(steps, "single"))

    def test_unknown_module_reported(self):
        errors = validate_workflow([("not_a_module", {})], Scope.SINGLE)
        self.assertEqual(1, len(errors))
        self.assertIn("unknown module", errors[0])

    def test_render_before_localize_reports_missing_input(self):
        steps = [("load_dataset_movie", {}), ("render", {})]
        errors = validate_workflow(steps, Scope.SINGLE)
        self.assertTrue(any("missing required inputs" in e for e in errors))
        self.assertTrue(any("locs_undrifted" in e for e in errors))

    def test_aggregation_analysis_workflow_validates(self):
        # Both-scope analysis modules require locs_undrifted; the aggregation
        # loader supplies it (saved single-dataset results are undrifted).
        steps = [
            ("load_datasets_to_aggregate", {}),
            ("dbscan", {}),
            ("spinna", {}),
            ("render", {}),
        ]
        self.assertEqual([], validate_workflow(steps, Scope.AGGREGATION))

    def test_load_localizations_then_cluster_validates(self):
        # A loaded locs file is assumed drift-corrected, so analysis requiring
        # locs_undrifted validates after a plain load.
        steps = [("load_dataset_localizations", {}), ("dbscan", {})]
        self.assertEqual([], validate_workflow(steps, Scope.SINGLE))

    def test_cluster_on_drifted_movie_load_still_flagged(self):
        # Loading a *movie* and localizing yields only 'locs'; analysis that
        # needs drift correction must still flag the missing undrift step.
        steps = [
            ("load_dataset_movie", {}),
            ("identify", {}),
            ("localize", {}),
            ("dbscan", {}),
        ]
        errors = validate_workflow(steps, Scope.SINGLE)
        self.assertTrue(
            any("locs_undrifted" in e for e in errors),
            f"expected missing locs_undrifted, got {errors}",
        )

    def test_aggregation_only_module_in_single_scope_reported(self):
        steps = [("load_datasets_to_aggregate", {})]
        errors = validate_workflow(steps, Scope.SINGLE)
        self.assertTrue(any("not valid in single" in e for e in errors))

    def test_single_only_module_in_aggregation_scope_reported(self):
        # undrift_rcc is single-only; flagged in aggregation scope.
        steps = [("undrift_rcc", {})]
        errors = validate_workflow(steps, Scope.AGGREGATION)
        self.assertTrue(any("not valid in aggregation" in e for e in errors))

    def test_after_constraint_violation_reported(self):
        registry = {
            "a": ModuleSpec(name="a"),
            "b": ModuleSpec(name="b", after=frozenset({"a"})),
        }
        errors = validate_workflow(
            [("b", {})], Scope.SINGLE, registry=registry
        )
        self.assertTrue(any("must come after 'a'" in e for e in errors))
        # Correct order is clean.
        self.assertEqual(
            [],
            validate_workflow(
                [("a", {}), ("b", {})], Scope.SINGLE, registry=registry
            ),
        )

    def test_optional_input_not_required(self):
        # export_brightfield only optionally consumes raw_movie.
        errors = validate_workflow([("export_brightfield", {})], Scope.SINGLE)
        self.assertEqual([], errors)

    def test_every_module_validates_alone_in_one_of_its_scopes(self):
        # Sanity check that the registry's own requires/provides are internally
        # consistent: each module preceded by producers of all its requires
        # passes in a scope it declares.
        producers = {}
        for spec in MODULE_REGISTRY.values():
            for token in spec.provides:
                producers.setdefault(token, spec.name)
        for spec in MODULE_REGISTRY.values():
            scope = next(iter(spec.scopes))
            prereqs = [
                (producers[t], {})
                for t in sorted(spec.requires)
                if t in producers
            ]
            steps = prereqs + [(spec.name, {})]
            errors = validate_workflow(steps, scope)
            # The module under test is the last step; allow scope mismatches
            # among borrowed producers, but assert it has no missing-input
            # error of its own (keyed by its step index, not name).
            own_prefix = f"[{len(steps) - 1}]"
            own = [
                e
                for e in errors
                if e.startswith(own_prefix) and "missing required inputs" in e
            ]
            self.assertEqual(
                [],
                own,
                f"{spec.name}: unsatisfiable requires "
                f"{sorted(spec.requires)}",
            )


if __name__ == "__main__":
    unittest.main()
