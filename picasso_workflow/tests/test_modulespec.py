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
    LOCS_STATE_CAPABILITIES,
    MEMORY_ONLY_CAPABILITIES,
    MODULE_REGISTRY,
    ModuleSpec,
    PicassoRelation,
    RUNTIME_PARAMETER_WRITE_BACKS,
    Scope,
    SINGLE_LOCS_CAPABILITIES,
    restart_conflicts,
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
        # picasso-set modules live outside the AbstractModuleCollection
        # contract (they are reconciled against the mixins in
        # test_picasso_set.py), so the no-orphans check covers natives only.
        specced = {
            name
            for name, spec in MODULE_REGISTRY.items()
            if spec.module_set == "native"
        }
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

    def test_unknown_module_set_rejected(self):
        with self.assertRaises(ValueError):
            ModuleSpec(name="x", module_set="other")

    def test_picasso_set_requires_prefix(self):
        with self.assertRaises(ValueError):
            ModuleSpec(
                name="density",
                module_set="picasso",
                relation=PicassoRelation.WRAPS,
                picasso_symbol="picasso.postprocess.compute_local_density",
            )

    def test_picasso_set_requires_wraps(self):
        with self.assertRaises(ValueError):
            ModuleSpec(name="picasso_density", module_set="picasso")


class TestPicassoSetSpecs(unittest.TestCase):
    """Registry-level invariants of the picasso-set entries."""

    def _picasso_specs(self):
        return [
            spec
            for spec in MODULE_REGISTRY.values()
            if spec.module_set == "picasso"
        ]

    def test_picasso_set_present(self):
        self.assertTrue(self._picasso_specs())

    def test_names_prefixed_and_wrapping(self):
        for spec in self._picasso_specs():
            self.assertTrue(spec.name.startswith("picasso_"), spec.name)
            self.assertIs(spec.relation, PicassoRelation.WRAPS, spec.name)
            self.assertTrue(spec.picasso_symbol, spec.name)

    def test_params_schema_shape(self):
        # params carries the GUI (parameters_spec, results_spec) pair; every
        # leaf must at least name its widget type.
        for spec in self._picasso_specs():
            self.assertIsInstance(spec.params, tuple, spec.name)
            self.assertEqual(2, len(spec.params), spec.name)
            parameters_spec, results_spec = spec.params
            for section in (parameters_spec, results_spec):
                self.assertIsInstance(section, dict, spec.name)
                for key, leaf in section.items():
                    self.assertIn(
                        "type", leaf, f"{spec.name}.{key}: missing type"
                    )

    def test_native_specs_unaffected(self):
        for spec in MODULE_REGISTRY.values():
            if spec.module_set == "native":
                self.assertFalse(spec.name.startswith("picasso_"), spec.name)


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

    def test_golden_picasso_set_workflow_passes(self):
        # picasso-set modules validate like natives and can mix with them.
        steps = [
            ("load_dataset_localizations", {}),
            ("picasso_density", {"radius": 1.5}),
        ]
        self.assertEqual([], validate_workflow(steps, Scope.SINGLE))

    def test_golden_picasso_set_pipeline_passes(self):
        # the full picasso-set core chain validates end-to-end
        steps = [
            ("load_dataset_movie", {}),
            ("picasso_localize", {}),
            ("picasso_undrift_rcc", {}),
            ("picasso_dbscan", {"radius": 0.5, "density": 4}),
            ("picasso_nneighbor", {"files": "clusters.hdf5"}),
            ("picasso_render", {}),
        ]
        self.assertEqual([], validate_workflow(steps, Scope.SINGLE))

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

    def test_branch_type_missing_or_unknown_rejected(self):
        # Missing branch_type is rejected (would KeyError at run time).
        errors = validate_workflow(
            [("branch", {"n_branches": 2, "branch_modules": []})],
            Scope.AGGREGATION,
        )
        self.assertTrue(any("branch_type must be" in e for e in errors))
        # An unknown/removed branch_type (e.g. "screen") is rejected.
        errors = validate_workflow(
            [("branch", {"branch_type": "screen", "branch_modules": []})],
            Scope.AGGREGATION,
        )
        self.assertTrue(any("branch_type must be" in e for e in errors))

    def test_branch_runtime_requires_branch_over(self):
        errors = validate_workflow(
            [("branch", {"branch_type": "runtime", "branch_modules": []})],
            Scope.AGGREGATION,
        )
        self.assertTrue(any("requires 'branch_over'" in e for e in errors))
        # With branch_over present it validates (no branch_type error).
        errors = validate_workflow(
            [
                (
                    "branch",
                    {
                        "branch_type": "runtime",
                        "branch_over": [1, 2],
                        "branch_modules": [],
                    },
                )
            ],
            Scope.AGGREGATION,
        )
        self.assertFalse(any("branch" in e for e in errors))

    def test_branch_explicit_accepts_zero_n_branches(self):
        # n_branches=0 is present (not None), so it must not trip the
        # "requires n_branches" error (truthiness bug guard).
        errors = validate_workflow(
            [
                (
                    "branch",
                    {
                        "branch_type": "explicit",
                        "n_branches": 0,
                        "branch_modules": [],
                    },
                )
            ],
            Scope.AGGREGATION,
        )
        self.assertFalse(any("explicit branch requires" in e for e in errors))

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


class TestRestartConflicts(unittest.TestCase):
    """restart_conflicts: in-memory state lost by a mid-workflow restart."""

    STEPS = [
        ("load_dataset_movie", {}),
        ("identify", {}),
        ("localize", {}),
        ("undrift_rcc", {}),
    ]

    def test_capability_sets_are_subsets_of_vocabulary(self):
        self.assertEqual(set(), MEMORY_ONLY_CAPABILITIES - CAPABILITIES)
        self.assertEqual(set(), LOCS_STATE_CAPABILITIES - CAPABILITIES)

    def test_write_back_map_names_registered_modules(self):
        self.assertEqual(
            set(),
            set(RUNTIME_PARAMETER_WRITE_BACKS) - set(MODULE_REGISTRY),
        )

    def test_restart_at_zero_never_conflicts(self):
        hard, soft = restart_conflicts(self.STEPS, 0)
        self.assertEqual([], hard)
        self.assertEqual([], soft)

    def test_memory_only_conflict_is_hard(self):
        # restarting at localize: identifications (from identify at [1]) and
        # raw_movie (from load at [0]) are memory-only and lost.
        hard, soft = restart_conflicts(self.STEPS, 2)
        self.assertTrue(any("identifications" in msg for msg in hard))
        self.assertTrue(any("raw_movie" in msg for msg in hard))

    def test_conflicts_after_restart_module_are_also_hard(self):
        # restarting at identify: localize (later in the re-run) also misses
        # raw_movie -- it WILL execute in this run, so it is hard too.
        hard, soft = restart_conflicts(self.STEPS, 1)
        self.assertTrue(any("[1] identify" in msg for msg in hard))
        self.assertTrue(any("[2] localize" in msg for msg in hard))

    def test_producer_inside_rerun_range_is_fine(self):
        # restarting at identify: identifications for localize are
        # re-produced by identify inside the re-run range.
        hard, soft = restart_conflicts(self.STEPS, 1)
        self.assertFalse(any("identifications" in msg for msg in hard + soft))

    def test_lost_locs_state_is_hard_unless_restored(self):
        # undrift_rcc requires 'locs' produced by localize at [2]. Without a
        # restore, restarting at [3] is not viable; with the single-locs
        # state restored (checkpoint), it is.
        hard, _ = restart_conflicts(self.STEPS, 3)
        self.assertTrue(any("'locs'" in msg for msg in hard))
        hard, _ = restart_conflicts(
            self.STEPS,
            3,
            lost_capabilities=(
                MEMORY_ONLY_CAPABILITIES
                | (LOCS_STATE_CAPABILITIES - SINGLE_LOCS_CAPABILITIES)
            ),
        )
        self.assertEqual([], hard)

    def test_unknown_module_is_soft(self):
        steps = self.STEPS + [("not_a_module", {})]
        hard, soft = restart_conflicts(steps, 4)
        self.assertEqual([], hard)
        self.assertTrue(any("cannot verify" in msg for msg in soft))

    def test_optional_memory_only_is_soft(self):
        steps = [
            ("load_picassoconfig", {}),
            ("load_dataset_movie", {}),
            ("identify", {}),
        ]
        # restart at load_dataset_movie: identify's optional picasso_config
        # (from [0]) is lost, but only advisory; raw_movie is re-produced.
        hard, soft = restart_conflicts(steps, 1)
        self.assertEqual([], hard)
        self.assertTrue(any("picasso_config" in msg for msg in soft))

    def test_branch_submodule_requirements_are_seen(self):
        # the branch step itself requires nothing memory-only, but its
        # sub-workflow contains localize, which needs raw_movie and
        # identifications from the trunk -- lost at the restart point.
        steps = [
            ("load_dataset_movie", {}),
            ("identify", {}),
            ("localize", {}),
            ("save_single_dataset", {}),
            (
                "branch",
                {"branch_modules": [("localize", {})]},
            ),
        ]
        hard, soft = restart_conflicts(steps, 4)
        self.assertTrue(any("[4] branch" in msg for msg in hard))
        self.assertTrue(any("raw_movie" in msg for msg in hard))
        # a branch whose sub-modules are self-sufficient is fine
        steps[4] = ("branch", {"branch_modules": [("dummy_module", {})]})
        hard, soft = restart_conflicts(
            steps,
            4,
            lost_capabilities=MEMORY_ONLY_CAPABILITIES,
        )
        self.assertEqual([], hard)


if __name__ == "__main__":
    unittest.main()
