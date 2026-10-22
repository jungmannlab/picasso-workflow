#!/usr/bin/env python
"""Tests for the sample-aware workflow recommender (WP-10).

The four layers are tested in isolation and end-to-end:

* Layer 0 (``select_modules``): goal/conditions pick a template; unknown goal
  falls back to a sane default; coarse descriptor tags are honoured.
* Layer 3 (``topological_order`` + whole-spec): the emitted workflow validates
  against ``validate_workflow`` for its scope — "recommended specs validate".
* Layer 1 (``resolve_choice_points``): with seeded cohort history the better
  variant (lower NeNA) is chosen; with no history the template default is kept.
* Taxon fallback: seeding only a PARENT taxon's runs still yields a non-empty
  cohort (tree-distance), so the choice point is still resolved.
* End-to-end (``recommend_workflow``): valid, sensible YAML on a localize assay
  and a channel-align assay.
* Persistence + operator-edit feedback round-trip through the registry.

The registry-backed cases use the in-memory ``picasso_registry.testing``
double and are guarded by ``@requires_registry`` so they skip on the hosted
wheel-only CI (which has no picasso-registry client stack), exactly like
``test_backfill.py`` / ``test_cluster_submit.py``. The pure-Python layer tests
have no such dependency and always run.
"""

import unittest

import yaml

from picasso_workflow import recommender as rec
from picasso_workflow.modulespec import Scope, _step_name, validate_workflow

# picasso-registry client stack is NOT a base dep of picasso-workflow (the
# package stays wheel-only / hermetic and imports headlessly without it), so the
# registry-backed tests skip on the hosted CI runner and run in the dev
# container. Mirrors test_backfill.py's guard.
try:
    from picasso_registry.testing import mock_registry as _mock_registry

    _ = _mock_registry
    HAS_REGISTRY = True
except Exception:
    HAS_REGISTRY = False

requires_registry = unittest.skipUnless(
    HAS_REGISTRY,
    "picasso-registry client stack (picasso-registry[client]) not installed "
    "(hosted wheel-only CI)",
)


# ---------------------------------------------------------------------------
# Seeding helper for the registry-backed tests.
# ---------------------------------------------------------------------------
def _seed_cohort(reg, *, taxon_id, runs, modality="TIRF", dim="2D", buf="B+"):
    """Seed ``runs`` = [(module, nena_nm), ...] under ``taxon_id``.

    One experiment/acquisition/analysis per run, each with a single metrics row
    whose ``module`` (recorded into ``extra``) and ``nena_nm`` drive the
    choice-point ranking.
    """
    for i, (module, nena) in enumerate(runs):
        eid, aid, anid = (
            f"e_{taxon_id}_{i}",
            f"a_{taxon_id}_{i}",
            (f"an_{taxon_id}_{i}"),
        )
        reg.log_experiment(
            id=eid,
            sample_taxon_id=taxon_id,
            acquisition_modality=modality,
            dimensionality=dim,
            buffer=buf,
        )
        reg.log_acquisition(id=aid, experiment_id=eid, data_source="acquired")
        reg.log_analysis(id=anid, acquisition_run_id=aid, kind="localize")
        reg.log_metrics(
            analysis_run_id=anid,
            module=module,
            module_index=3,
            nena_nm=nena,
            scope="single",
        )


# ---------------------------------------------------------------------------
# Layer 0 — template select (pure, no registry).
# ---------------------------------------------------------------------------
class TestSelectModules(unittest.TestCase):
    def test_localize_goal_picks_single_template(self):
        wf, template = rec.select_modules("localize", {})
        self.assertIsNotNone(template)
        self.assertEqual(template.scope, Scope.SINGLE)
        self.assertEqual(template.question_type, "localize")
        names = [_step_name(s) for s in wf]
        self.assertIn("localize", names)
        self.assertIn("load_dataset_movie", names)

    def test_channel_align_goal_picks_aggregation_template(self):
        wf, template = rec.select_modules("channel-align", {})
        self.assertIsNotNone(template)
        self.assertEqual(template.scope, Scope.AGGREGATION)
        names = [_step_name(s) for s in wf]
        self.assertIn("align_channels", names)

    def test_unknown_goal_falls_back_to_default_template(self):
        # No template has question_type "what-even-is-this" -> fallback.
        wf, template = rec.select_modules("what-even-is-this", {})
        self.assertIsNotNone(template)
        self.assertEqual(template.template_id, "minimal")
        self.assertTrue(wf)

    def test_coarse_descriptor_tag_mismatch_is_filtered(self):
        # A declared tag that contradicts the conditions must not match; the
        # shipped templates carry no tags, so a contradicting condition still
        # falls back cleanly rather than erroring.
        wf, template = rec.select_modules("localize", {"sample_type": "Cells"})
        self.assertIsNotNone(template)
        self.assertTrue(wf)


# ---------------------------------------------------------------------------
# Layer 3 — recommended specs validate (pure, no registry).
# ---------------------------------------------------------------------------
class TestTopologicalOrderAndValidation(unittest.TestCase):
    def test_single_template_spec_validates(self):
        wf, _ = rec.select_modules("localize", {})
        ordered = rec.topological_order(wf, Scope.SINGLE)
        self.assertEqual(validate_workflow(ordered, Scope.SINGLE), [])

    def test_aggregation_template_spec_validates(self):
        # An aggregation workflow is two scopes: the per-dataset SINGLE phase
        # then the AGGREGATION phase. topological_order validates each in its
        # own scope; assert the split halves validate independently too.
        wf, _ = rec.select_modules("channel-align", {})
        ordered = rec.topological_order(wf, Scope.AGGREGATION)
        single_phase, agg_phase = rec._split_aggregation(ordered)
        self.assertEqual(validate_workflow(single_phase, Scope.SINGLE), [])
        self.assertEqual(validate_workflow(agg_phase, Scope.AGGREGATION), [])

    def test_order_reorders_a_perturbed_workflow(self):
        # Put undrift before localize; the DAG sort must repair the order so
        # locs is provided before locs_undrifted is consumed.
        perturbed = [
            ("load_dataset_movie", {}),
            ("undrift_rcc", {}),
            ("identify", {}),
            ("localize", {}),
            ("save_single_dataset", {}),
        ]
        ordered = rec.topological_order(perturbed, Scope.SINGLE)
        names = [_step_name(s) for s in ordered]
        self.assertLess(names.index("localize"), names.index("undrift_rcc"))
        self.assertEqual(validate_workflow(ordered, Scope.SINGLE), [])

    def test_unsatisfiable_graph_raises(self):
        # dbscan requires locs_undrifted that nothing provides -> no valid order.
        with self.assertRaises(ValueError):
            rec.topological_order([("dbscan", {})], Scope.SINGLE)


# ---------------------------------------------------------------------------
# Layer 1 — choice points over seeded cohort history (registry-backed).
# ---------------------------------------------------------------------------
class TestChoicePointsAndFallback(unittest.TestCase):
    @requires_registry
    def test_seeded_history_picks_the_better_variant(self):
        from picasso_registry.testing import mock_registry

        with mock_registry() as reg:
            reg.add_taxon(id="synthetic", name="Synthetic")
            reg.add_taxon(id="origami", name="Origami", parent_id="synthetic")
            # undrift_aim has lower (better) NeNA than undrift_rcc. Seed >=
            # MIN_COHORT_SAMPLES (3) aim runs so the swap clears the
            # noisy-single-sample guard.
            _seed_cohort(
                reg,
                taxon_id="origami",
                runs=[
                    ("undrift_aim", 5.0),
                    ("undrift_aim", 5.5),
                    ("undrift_aim", 5.2),
                    ("undrift_rcc", 9.0),
                ],
            )
            wf, _ = rec.select_modules("localize", {})
            # template default is undrift_rcc.
            self.assertIn("undrift_rcc", [_step_name(s) for s in wf])
            out, decisions = rec.resolve_choice_points(
                wf, {"taxon_id": "origami"}, reg
            )
            names = [_step_name(s) for s in out]
            self.assertIn("undrift_aim", names)
            self.assertNotIn("undrift_rcc", names)
            undrift = next(d for d in decisions if d["role"] == "undrift")
            self.assertEqual(undrift["chosen"], "undrift_aim")
            self.assertEqual(undrift["source"], "cohort-history")

    @requires_registry
    def test_no_history_keeps_template_default(self):
        from picasso_registry.testing import mock_registry

        with mock_registry() as reg:
            reg.add_taxon(id="synthetic", name="Synthetic")
            reg.add_taxon(id="origami", name="Origami", parent_id="synthetic")
            wf, _ = rec.select_modules("localize", {})
            out, decisions = rec.resolve_choice_points(
                wf, {"taxon_id": "origami"}, reg
            )
            self.assertIn("undrift_rcc", [_step_name(s) for s in out])
            undrift = next(d for d in decisions if d["role"] == "undrift")
            self.assertEqual(undrift["chosen"], "undrift_rcc")
            self.assertEqual(undrift["source"], "template-default")

    @requires_registry
    def test_taxon_fallback_non_empty_cohort_from_parent(self):
        from picasso_registry.testing import mock_registry

        with mock_registry() as reg:
            reg.add_taxon(id="synthetic", name="Synthetic")
            reg.add_taxon(id="origami", name="Origami", parent_id="synthetic")
            # Exact child "beads" is EMPTY; seed only the sibling/parent chain.
            reg.add_taxon(id="beads", name="Beads", parent_id="synthetic")
            _seed_cohort(
                reg,
                taxon_id="origami",
                runs=[
                    ("undrift_aim", 5.0),
                    ("undrift_aim", 5.2),
                    ("undrift_aim", 5.1),
                ],
            )
            # Query at the PARENT node: the cohort falls back up the tree and is
            # non-empty, so the choice point still resolves to the winner.
            cohort = reg.cohort("synthetic")
            self.assertTrue(cohort)  # non-empty via tree-distance
            wf, _ = rec.select_modules("localize", {})
            out, decisions = rec.resolve_choice_points(
                wf, {"taxon_id": "synthetic"}, reg
            )
            self.assertIn("undrift_aim", [_step_name(s) for s in out])


# ---------------------------------------------------------------------------
# Layer 2 — parameters from node defaults + learned ranges (registry-backed).
# ---------------------------------------------------------------------------
class TestRecommendParameters(unittest.TestCase):
    @requires_registry
    def test_node_default_overrides_template_param(self):
        from picasso_registry.testing import mock_registry

        with mock_registry() as reg:
            reg.add_taxon(id="synthetic", name="Synthetic")
            reg.add_taxon(
                id="origami",
                name="Origami",
                parent_id="synthetic",
                defaults={"localize": {"box_size": 9}},
            )
            wf = [("localize", {"box_size": 7, "fit_method": "lsq"})]
            out, decisions = rec.recommend_parameters(
                wf, {"taxon_id": "origami"}, reg
            )
            self.assertEqual(out[0][1]["box_size"], 9)
            override = next(
                d for d in decisions if d.get("param") == "box_size"
            )
            self.assertEqual(override["value"], 9)
            self.assertTrue(override["source"].startswith("node_defaults"))

    @requires_registry
    def test_learned_ranges_recorded_as_hints(self):
        from picasso_registry.testing import mock_registry

        with mock_registry() as reg:
            reg.add_taxon(id="synthetic", name="Synthetic")
            reg.add_taxon(id="origami", name="Origami", parent_id="synthetic")
            reg.log_cohort_range(
                id="r1",
                taxon_id="origami",
                metric="density_locs_um2",
                p05=1.0,
                p50=2.0,
                p95=3.0,
                n=10,
                source="wp7",
            )
            wf = [("localize", {"box_size": 7})]
            _, decisions = rec.recommend_parameters(
                wf, {"taxon_id": "origami"}, reg
            )
            hint = next(
                d for d in decisions if d.get("hint") == "learned-cohort-range"
            )
            self.assertEqual(hint["metric"], "density_locs_um2")
            self.assertEqual(hint["p50"], 2.0)

    @requires_registry
    def test_latest_range_row_wins(self):
        from picasso_registry.testing import mock_registry

        with mock_registry() as reg:
            reg.add_taxon(id="synthetic", name="Synthetic")
            reg.add_taxon(id="origami", name="Origami", parent_id="synthetic")
            # Two append-only snapshots for the same (taxon, metric); the later
            # (newer id) must win.
            reg.log_cohort_range(
                id="r1",
                taxon_id="origami",
                metric="nena_nm",
                p50=8.0,
                n=3,
            )
            reg.log_cohort_range(
                id="r2",
                taxon_id="origami",
                metric="nena_nm",
                p50=6.0,
                n=12,
            )
            latest = rec._latest_cohort_ranges(reg, "origami")
            self.assertEqual(latest["nena_nm"]["p50"], 6.0)
            self.assertEqual(latest["nena_nm"]["n"], 12)


# ---------------------------------------------------------------------------
# End-to-end — valid, sensible YAML on known assays (no registry needed).
# ---------------------------------------------------------------------------
class TestRecommendWorkflowEndToEnd(unittest.TestCase):
    def test_localize_assay_emits_valid_ordered_yaml(self):
        r = rec.recommend_workflow({"sample_type": "Synthetic"}, "localize")
        self.assertEqual(r.scope, Scope.SINGLE)
        self.assertTrue(r.rationale["validated"])
        self.assertEqual(validate_workflow(r.workflow, Scope.SINGLE), [])
        # YAML round-trips to the same module sequence.
        doc = yaml.safe_load(r.to_yaml())
        names = [step["module"] for step in doc["workflow"]]
        self.assertEqual(names, [_step_name(s) for s in r.workflow])
        self.assertEqual(doc["recommender_version"], rec.RECOMMENDER_VERSION)

    def test_channel_align_assay_emits_valid_ordered_yaml(self):
        r = rec.recommend_workflow(
            {"sample_type": "Synthetic"}, "channel-align"
        )
        self.assertEqual(r.scope, Scope.AGGREGATION)
        self.assertTrue(r.rationale["validated"])
        # Each scope phase validates in its own scope.
        single_phase, agg_phase = rec._split_aggregation(r.workflow)
        self.assertEqual(validate_workflow(single_phase, Scope.SINGLE), [])
        self.assertEqual(validate_workflow(agg_phase, Scope.AGGREGATION), [])
        names = [_step_name(s) for s in r.workflow]
        self.assertIn("align_channels", names)

    def test_no_registry_still_returns_valid_spec(self):
        # db_client=None disables Layers 1-2; template defaults are kept and the
        # spec still validates (advisory, never raises).
        r = rec.recommend_workflow(
            {"taxon_id": "origami"}, "localize", db_client=None
        )
        self.assertEqual(validate_workflow(r.workflow, Scope.SINGLE), [])

    @requires_registry
    def test_end_to_end_with_history_and_persistence(self):
        from picasso_registry.testing import mock_registry

        with mock_registry() as reg:
            reg.add_taxon(id="synthetic", name="Synthetic")
            reg.add_taxon(id="origami", name="Origami", parent_id="synthetic")
            _seed_cohort(
                reg,
                taxon_id="origami",
                runs=[
                    ("undrift_aim", 4.0),
                    ("undrift_aim", 4.2),
                    ("undrift_aim", 3.9),
                    ("undrift_rcc", 9.0),
                ],
            )
            r = rec.recommend_workflow(
                {"sample_type": "Synthetic", "taxon_id": "origami"},
                "localize",
                registry=reg,
                db_client=reg,
            )
            self.assertIn("undrift_aim", [_step_name(s) for s in r.workflow])
            self.assertEqual(validate_workflow(r.workflow, Scope.SINGLE), [])

            # Persist: recommended_spec + recommender_version land on an
            # analysis_run row.
            resp = rec.persist_recommendation(
                reg, r, acquisition_run_id="a_origami_0", taxon_id="origami"
            )
            self.assertIsNotNone(resp)
            self.assertEqual(
                resp["recommender_version"], rec.RECOMMENDER_VERSION
            )
            self.assertTrue(resp["recommended_spec"]["workflow"])

            # Operator edits a param -> feedback captures the diff.
            op_wf = [
                (
                    name,
                    (
                        {**params, "box_size": 11}
                        if name == "localize"
                        else params
                    ),
                )
                for name, params in r.workflow
            ]
            fb = rec.capture_operator_feedback(
                reg, r, op_wf, acquisition_run_id="a_origami_0"
            )
            self.assertIsNotNone(fb)
            self.assertEqual(fb["operator_decision"], "edited")
            self.assertIn(
                "localize.box_size", fb["recommender_edit"]["param_overrides"]
            )

    @requires_registry
    def test_accepted_feedback_when_unedited(self):
        from picasso_registry.testing import mock_registry

        with mock_registry() as reg:
            r = rec.recommend_workflow(
                {"sample_type": "Synthetic"}, "localize"
            )
            fb = rec.capture_operator_feedback(reg, r, list(r.workflow))
            self.assertEqual(fb["operator_decision"], "accepted")


class TestRecommenderReviewFixes(unittest.TestCase):
    """Regression tests for the code-review findings."""

    def test_invalid_scope_does_not_raise(self):
        # recommend_workflow never raises: an invalid conditions['scope'] falls
        # back to goal inference instead of a ValueError.
        r = rec.recommend_workflow({"scope": "agg"}, "localize")
        self.assertTrue(r.workflow)  # a valid default workflow, no exception

    def test_no_conditions_or_registry_does_not_raise(self):
        # The whole advisory path tolerates absent conditions/registry.
        r = rec.recommend_workflow({}, "localize")
        self.assertTrue(r.workflow)

    def test_bare_param_node_default_applies(self):
        # #5: a bare "<param>" node default applies to every module declaring it.
        wf = [("identify", {"box_size": 7}), ("localize", {"box_size": 7})]
        ovr = rec._node_default_overrides(
            {"box_size": 9}, "identify", {"box_size": 7}
        )
        self.assertEqual(ovr.get("box_size"), (9, "node_defaults[<param>]"))
        # not applied to a module that doesn't declare it.
        self.assertEqual(
            rec._node_default_overrides(
                {"foo": 1}, "identify", {"box_size": 7}
            ),
            {},
        )

    @requires_registry
    def test_single_sample_does_not_swap_the_default(self):
        # #3: one lucky cohort sample must NOT replace the template default.
        from picasso_registry.testing import mock_registry

        with mock_registry() as reg:
            reg.add_taxon(id="synthetic", name="Synthetic")
            reg.add_taxon(id="origami", name="Origami", parent_id="synthetic")
            _seed_cohort(
                reg,
                taxon_id="origami",
                runs=[("undrift_aim", 1.0)],  # a single, lucky sample
            )
            wf, _ = rec.select_modules("localize", {})
            out, decisions = rec.resolve_choice_points(
                wf, {"taxon_id": "origami"}, reg
            )
            names = [_step_name(s) for s in out]
            self.assertIn("undrift_rcc", names)  # template default kept
            self.assertNotIn("undrift_aim", names)
            undrift = next(d for d in decisions if d["role"] == "undrift")
            self.assertEqual(undrift["source"], "template-default-kept")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
