#!/usr/bin/env python
"""Tests for the offline parameter self-optimizer (WP-11).

The search logic is pure and tested here against in-test synthetic objectives
(the injected ``evaluate_fn`` seam), with the registry-backed provenance case
guarded by ``@requires_registry`` so it skips on the hosted wheel-only CI
(exactly like ``test_backfill.py`` / ``test_recommender.py``).

Verification bar (Tier 2 — scientific correctness):

* **KNOWN-OPTIMUM + held-out:** on a fixed noisy quadratic bowl with a known
  optimum, the search recovers it (within tolerance) AND beats a hand-tuned
  baseline on a HELD-OUT noise split — so it is not overfitting the search set.
* **DETERMINISM:** two seeded ``optimize`` calls produce identical proposals +
  winner.
* **PROPERTY — bounds:** over many seeds / a generated space, every proposed
  candidate is within bounds.
* **MUTATION (on the objective/scoring):** tests that FAIL if the minimize/
  maximize direction were inverted, or the "pick best" comparison flipped.
* **PROVENANCE:** the chosen params + search trace land in the registry, and a
  registry outage is swallowed (best-effort).
"""

from __future__ import annotations

import random
import unittest

from picasso_workflow import self_optimize as so

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
# A fixed synthetic objective: a noisy quadratic bowl with a known optimum.
# ---------------------------------------------------------------------------
X_STAR, Y_STAR = 0.6, -0.3  # the known optimum (inside the declared bounds).

# Two searched params on a native-like module, so (module, param) keys are real.
BOWL_SPACE = {
    ("localize", "box_size"): {
        "type": "float",
        "low": -1.0,
        "high": 1.0,
        "step": 0.1,
        "default": -1.0,  # data-driven default sits at an edge, not the optimum
    },
    ("undrift_rcc", "segmentation"): {
        "type": "float",
        "low": -1.0,
        "high": 1.0,
        "step": 0.1,
        "default": 1.0,
    },
}


def _bowl(noise_seed: int, amp: float = 0.01):
    """A ``evaluate_fn`` over :data:`BOWL_SPACE`: lower ``score`` is better.

    ``score(x, y) = (x - x*)^2 + (y - y*)^2 + seeded_noise``. The noise is keyed
    on the candidate AND ``noise_seed`` so a different ``noise_seed`` is a
    genuinely different (held-out) split, but each split is deterministic.
    """

    def _evaluate(params):
        x = params[("localize", "box_size")]
        y = params[("undrift_rcc", "segmentation")]
        rng = random.Random(f"{noise_seed}:{round(x, 6)}:{round(y, 6)}")
        noise = (rng.random() - 0.5) * amp
        return {"score": (x - X_STAR) ** 2 + (y - Y_STAR) ** 2 + noise}

    return _evaluate


# ---------------------------------------------------------------------------
# KNOWN-OPTIMUM + held-out guard.
# ---------------------------------------------------------------------------
class TestKnownOptimum(unittest.TestCase):
    def test_grid_recovers_optimum_and_beats_baseline_on_holdout(self):
        search = _bowl(noise_seed=1)
        holdout = _bowl(noise_seed=999)  # different noise split

        result = so.optimize(
            BOWL_SPACE,
            "score",
            search,
            seed=42,
            strategy="grid",
            minimize=True,
            max_evals=500,  # full 21x21 grid over the bowl
            holdout_eval=holdout,
        )

        bx = result.best_params[("localize", "box_size")]
        by = result.best_params[("undrift_rcc", "segmentation")]
        # Recovered the known optimum within one grid step.
        self.assertAlmostEqual(bx, X_STAR, delta=0.11)
        self.assertAlmostEqual(by, Y_STAR, delta=0.11)

        # Beats a hand-tuned baseline (the data-driven defaults) on the HELD-OUT
        # split — guards against overfitting the search set.
        baseline = {
            ("localize", "box_size"): -1.0,
            ("undrift_rcc", "segmentation"): 1.0,
        }
        baseline_holdout = holdout(baseline)["score"]
        winner_holdout = holdout(result.best_params)["score"]
        self.assertLess(winner_holdout, baseline_holdout)
        # The held-out score is recorded in the result + provenance.
        self.assertIsNotNone(result.holdout_score)
        self.assertIn("holdout", result.rationale)

    def test_coordinate_strategy_also_recovers_optimum(self):
        search = _bowl(noise_seed=2)
        result = so.optimize(
            BOWL_SPACE,
            "score",
            search,
            seed=7,
            strategy="coordinate",
            minimize=True,
            max_evals=200,
            rounds=4,
        )
        bx = result.best_params[("localize", "box_size")]
        by = result.best_params[("undrift_rcc", "segmentation")]
        self.assertAlmostEqual(bx, X_STAR, delta=0.11)
        self.assertAlmostEqual(by, Y_STAR, delta=0.11)

    def test_holdout_guard_prefers_the_generalizing_config(self):
        # Construct an objective where ONE search candidate wins only because of
        # a lucky search-split noise spike, but a different candidate wins the
        # held-out split. The guard must prefer the held-out winner.
        target = ("localize", "box_size")
        space = {
            target: {
                "type": "choice",
                "choices": ["a", "b"],
                "default": "a",
            }
        }

        def search(params):
            # 'a' looks great ONLY on the search split (big negative spike).
            return {"m": 0.0 if params[target] == "b" else -5.0}

        def holdout(params):
            # On held-out, 'b' is genuinely better.
            return {"m": 1.0 if params[target] == "a" else 0.0}

        result = so.optimize(
            space, "m", search, seed=3, minimize=True, holdout_eval=holdout
        )
        # Search alone would pick 'a' (-5.0); the held-out guard flips to 'b'.
        self.assertEqual(result.best_params[target], "b")
        self.assertEqual(
            result.rationale["holdout"]["preferred"], "holdout_winner"
        )


# ---------------------------------------------------------------------------
# DETERMINISM.
# ---------------------------------------------------------------------------
class TestDeterminism(unittest.TestCase):
    def test_same_seed_identical_proposals_and_winner(self):
        r1 = so.optimize(
            BOWL_SPACE,
            "score",
            _bowl(noise_seed=5),
            seed=123,
            strategy="coordinate",
            minimize=True,
            max_evals=120,
        )
        r2 = so.optimize(
            BOWL_SPACE,
            "score",
            _bowl(noise_seed=5),
            seed=123,
            strategy="coordinate",
            minimize=True,
            max_evals=120,
        )
        self.assertEqual(r1.best_params, r2.best_params)
        self.assertEqual(r1.best_score, r2.best_score)
        # The full search trace is identical too (reproducible provenance).
        self.assertEqual(r1.provenance["trace"], r2.provenance["trace"])

    def test_different_seed_coordinate_explores_differently(self):
        # The random restart is seeded, so a different seed can visit different
        # candidates (sanity: the seed actually threads into the search).
        r1 = so.optimize(
            BOWL_SPACE,
            "score",
            _bowl(noise_seed=5),
            seed=1,
            strategy="coordinate",
            minimize=True,
            max_evals=30,
            rounds=1,
        )
        r2 = so.optimize(
            BOWL_SPACE,
            "score",
            _bowl(noise_seed=5),
            seed=2,
            strategy="coordinate",
            minimize=True,
            max_evals=30,
            rounds=1,
        )
        # Both still valid; the traces are allowed to differ by seed.
        self.assertTrue(r1.provenance["trace"])
        self.assertTrue(r2.provenance["trace"])


# ---------------------------------------------------------------------------
# PROPERTY — bounds (property-style: loop seeds / generated spaces).
# ---------------------------------------------------------------------------
class TestBoundsProperty(unittest.TestCase):
    def _random_space(self, rng):
        space = {}
        n = rng.randint(1, 3)
        for i in range(n):
            kind = rng.choice(["float", "int", "choice"])
            key = (f"mod{i}", f"param{i}")
            if kind == "choice":
                space[key] = {
                    "type": "choice",
                    "choices": [rng.randint(0, 9) for _ in range(3)],
                    "default": 0,
                }
            else:
                lo = rng.uniform(-10, 0)
                hi = rng.uniform(0.1, 10)
                space[key] = {
                    "type": kind,
                    "low": lo,
                    "high": hi,
                    "step": (hi - lo) / rng.randint(2, 6),
                    "default": lo,
                }
        return space

    def test_every_proposed_candidate_is_in_bounds(self):
        for trial in range(40):
            rng = random.Random(trial)
            space = self._random_space(rng)
            objective = "m"
            seen = []

            def evaluate(params, _seen=seen, _sp=space):
                # Record + bounds-check every candidate the search proposes.
                so.assert_in_bounds(_sp, params)
                _seen.append(dict(params))
                # arbitrary deterministic score
                return {
                    objective: sum(hash(repr(v)) % 7 for v in params.values())
                }

            strat = "grid" if trial % 2 == 0 else "coordinate"
            result = so.optimize(
                space,
                objective,
                evaluate,
                seed=trial,
                strategy=strat,
                minimize=bool(trial % 2),
                max_evals=50,
            )
            self.assertTrue(seen, "search proposed no candidates")
            # The winner itself is in bounds.
            so.assert_in_bounds(space, result.best_params)

    def test_clamp_forces_out_of_bounds_into_range(self):
        spec_f = {"type": "float", "low": 0.0, "high": 1.0}
        self.assertEqual(so._clamp(spec_f, 5.0), 1.0)
        self.assertEqual(so._clamp(spec_f, -5.0), 0.0)
        spec_i = {"type": "int", "low": 0, "high": 10}
        self.assertEqual(so._clamp(spec_i, 7.6), 8)
        self.assertEqual(so._clamp(spec_i, 100), 10)
        spec_c = {"type": "choice", "choices": ["a", "b"]}
        self.assertEqual(so._clamp(spec_c, "zzz"), "a")

    def test_assert_in_bounds_rejects_out_of_range(self):
        space = {("m", "p"): {"type": "float", "low": 0.0, "high": 1.0}}
        with self.assertRaises(AssertionError):
            so.assert_in_bounds(space, {("m", "p"): 2.0})


# ---------------------------------------------------------------------------
# MUTATION — the objective direction / "pick best" comparison must bite.
# ---------------------------------------------------------------------------
class TestObjectiveDirectionBites(unittest.TestCase):
    """These FAIL under the two targeted mutations:

    * inverting minimize<->maximize in ``optimize``'s ``pick``;
    * flipping the ``min``/``max`` "pick best" selection.
    """

    def _linear_space(self):
        # score = x; the min is at low, the max is at high — opposite corners,
        # so a flipped direction lands on the wrong one and the asserts fail.
        return {
            ("m", "x"): {
                "type": "float",
                "low": 0.0,
                "high": 10.0,
                "step": 1.0,
                "default": 5.0,
            }
        }

    def _evaluate(self, params):
        return {"score": params[("m", "x")]}

    def test_minimize_picks_the_low_end(self):
        result = so.optimize(
            self._linear_space(),
            "score",
            self._evaluate,
            seed=0,
            minimize=True,
        )
        # Minimizing score=x must land at the LOW end; a flipped direction (or
        # a min->max mutation of the pick) would land at 10.0 and fail here.
        self.assertEqual(result.best_params[("m", "x")], 0.0)
        self.assertEqual(result.best_score, 0.0)

    def test_maximize_picks_the_high_end(self):
        result = so.optimize(
            self._linear_space(),
            "score",
            self._evaluate,
            seed=0,
            minimize=False,
        )
        # The mirror case: maximizing must land at the HIGH end.
        self.assertEqual(result.best_params[("m", "x")], 10.0)
        self.assertEqual(result.best_score, 10.0)

    def test_direction_inferred_from_a1_c19_table(self):
        # nena_nm is lower-is-better (A1/C19), so an un-overridden optimize on
        # it MINIMIZES — picks the low end. A mutation that dropped the
        # _LOWER_IS_BETTER lookup (defaulting to maximize) would fail here.
        space = {
            ("localize", "x"): {
                "type": "float",
                "low": 1.0,
                "high": 9.0,
                "step": 1.0,
                "default": 9.0,
            }
        }
        result = so.optimize(
            space,
            "nena_nm",
            lambda p: {"nena_nm": p[("localize", "x")]},
            seed=0,
        )
        self.assertTrue(result.minimize)
        self.assertEqual(result.best_params[("localize", "x")], 1.0)

    def test_unknown_metric_defaults_to_maximize(self):
        self.assertFalse(so.infer_minimize("photons_median"))
        self.assertTrue(so.infer_minimize("nena_nm"))


# ---------------------------------------------------------------------------
# Validation + evaluator-seam helpers (pure).
# ---------------------------------------------------------------------------
class TestValidationAndSeam(unittest.TestCase):
    def test_malformed_space_raises(self):
        with self.assertRaises(ValueError):
            so.optimize({}, "m", lambda p: {"m": 1}, seed=0)
        with self.assertRaises(ValueError):
            so.optimize(
                {("m", "p"): {"type": "float", "low": 1.0}},  # no high
                "m",
                lambda p: {"m": 1},
                seed=0,
            )
        with self.assertRaises(ValueError):
            so.optimize(
                {("m", "p"): {"type": "float", "low": 2.0, "high": 1.0}},
                "m",
                lambda p: {"m": 1},
                seed=0,
            )

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            so.optimize(
                BOWL_SPACE,
                "score",
                _bowl(1),
                seed=0,
                strategy="bayes-9000",
            )

    def test_missing_objective_metric_returns_defaults(self):
        # Every candidate lacks the objective → defaults returned, flagged.
        result = so.optimize(
            BOWL_SPACE,
            "score",
            lambda p: {"other": 1.0},
            seed=0,
            minimize=True,
        )
        self.assertEqual(result.n_evals, 0)
        self.assertEqual(
            result.best_params[("localize", "box_size")], -1.0
        )  # the data-driven default

    def test_overlay_params_writes_onto_matching_module(self):
        base = [("localize", {"box_size": 7}), ("undrift_rcc", {})]
        out = so._overlay_params(
            base,
            {
                ("localize", "box_size"): 9,
                ("undrift_rcc", "segmentation"): 500,
            },
        )
        self.assertEqual(dict(out)["localize"]["box_size"], 9)
        self.assertEqual(dict(out)["undrift_rcc"]["segmentation"], 500)
        # original base is not mutated (deep-ish copy of params).
        self.assertEqual(base[0][1]["box_size"], 7)

    def test_overlay_params_drops_unknown_module(self):
        base = [("localize", {})]
        out = so._overlay_params(base, {("nope", "p"): 1})
        self.assertEqual(out, [("localize", {})])

    def test_parallel_map_hook_is_used(self):
        calls = {"n": 0}

        def pmap(fn, cands):
            calls["n"] += 1
            return [fn(c) for c in cands]

        so.optimize(
            BOWL_SPACE,
            "score",
            _bowl(1),
            seed=0,
            strategy="grid",
            minimize=True,
            parallel_map=pmap,
        )
        self.assertGreaterEqual(calls["n"], 1)

    def test_cluster_evaluator_overlays_and_submits(self):
        submitted = {}

        def submit_fn(wf):
            submitted["wf"] = wf
            return {"nena_nm": 3.0}

        base = [("localize", {"box_size": 7})]
        ev = so.cluster_evaluator(base, submit_fn)
        out = ev({("localize", "box_size"): 9})
        self.assertEqual(out["nena_nm"], 3.0)
        self.assertEqual(dict(submitted["wf"])["localize"]["box_size"], 9)


# ---------------------------------------------------------------------------
# PROVENANCE — write-back to the in-memory registry (best-effort).
# ---------------------------------------------------------------------------
class TestProvenance(unittest.TestCase):
    @requires_registry
    def test_chosen_params_and_trace_land_in_registry(self):
        from picasso_registry.testing import mock_registry

        with mock_registry() as reg:
            result = so.optimize(
                BOWL_SPACE,
                "score",
                _bowl(1),
                seed=42,
                strategy="grid",
                minimize=True,
                registry=reg,
                run_context={"taxon_id": "origami"},
            )
            rows = reg.list("analysis_run", limit=500)
            opt_rows = [r for r in rows if r.get("kind") == "optimization"]
            self.assertEqual(len(opt_rows), 1)
            row = opt_rows[0]
            self.assertEqual(row["recommender_version"], so.OPTIMIZER_VERSION)
            spec = row["recommended_spec"]
            self.assertEqual(spec["objective"], "score")
            self.assertTrue(spec["minimize"])
            self.assertEqual(spec["seed"], 42)
            # The chosen params persisted with dotted string keys.
            self.assertIn("localize.box_size", spec["chosen_params"])
            # The full search trace landed under extra.
            self.assertTrue(row["extra"]["search_trace"])
            self.assertEqual(row["extra"]["taxon_id"], "origami")
            # Best params in the returned result match what was written.
            self.assertEqual(
                result.params_as_str_keys()["localize.box_size"],
                spec["chosen_params"]["localize.box_size"],
            )

    def test_registry_outage_is_swallowed(self):
        class _FailingRegistry:
            def log_analysis(self, **_):
                raise RuntimeError("registry down")

        # Must NOT raise; the result is returned regardless (best-effort).
        result = so.optimize(
            BOWL_SPACE,
            "score",
            _bowl(1),
            seed=0,
            strategy="grid",
            minimize=True,
            registry=_FailingRegistry(),
        )
        self.assertIsNotNone(result.best_params)

    def test_no_registry_skips_write_back(self):
        # registry=None → nothing persisted, result still returned.
        result = so.optimize(
            BOWL_SPACE, "score", _bowl(1), seed=0, minimize=True
        )
        self.assertIsNotNone(result.best_score)
        self.assertEqual(result.optimizer_version, so.OPTIMIZER_VERSION)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
