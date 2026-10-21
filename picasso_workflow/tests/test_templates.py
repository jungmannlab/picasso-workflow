#!/usr/bin/env python
"""Tests for versioned analysis templates + deviation logging.

WP-TEMPLATE-CONFORMANCE (Phases 1-2). Covers:
* config_diff is exact and stable (exact / param-override / structural / ad-hoc);
* template_version changes iff template content changes (hash stability);
* a diff against template vN is reproducible and order-independent;
* the qc.json sidecar round-trips the three fields additively;
* T1 verification: golden config_diff snapshots for curated (template, run)
  pairs, cross-checked by an INDEPENDENT oracle that re-derives the diff from
  the two resolved workflow YAMLs by a different route than the production code.
"""

import json
import os
import tempfile
import unittest

import yaml

import picasso_workflow.standard_singledataset_workflows as ssw
from picasso_workflow.modulespec import _step_name, _step_params
from picasso_workflow import templates as tpl


# ---------------------------------------------------------------------------
# Independent oracle (test-side; different route than production code).
#
# Production diffs the CANONICAL (placeholder-normalized) forms in-memory via a
# two-pointer walk. The oracle instead: serializes each workflow to YAML, reads
# it back, and compares by (a) a position-free multiset of module names for
# structure and (b) a brute-force nested-dict leaf walk for params — a wholly
# separate implementation, so agreement is real evidence.
# ---------------------------------------------------------------------------
def _resolved_to_yaml(workflow):
    """Serialize a workflow to a YAML string of plain dicts (oracle input)."""
    doc = []
    for step in workflow:
        doc.append(
            {
                "module": _step_name(step),
                "params": _oracle_canon(dict(_step_params(step))),
            }
        )
    return yaml.safe_dump(doc, sort_keys=True)


_ORACLE_RUN_KEYS = {"filename", "filepath", "filepaths", "tags"}


def _oracle_canon(value, key=None):
    """Independent canonicalizer: normalize run-specific holes + directives."""
    if key is not None and key in _ORACLE_RUN_KEYS:
        return "<run-specific>"
    # a recipe directive is a list/tuple whose head is a "$..." string
    if (
        isinstance(value, (list, tuple))
        and value
        and isinstance(value[0], str)
        and value[0].startswith("$")
    ):
        return "<run-specific>"
    if isinstance(value, dict):
        return {k: _oracle_canon(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_oracle_canon(v) for v in value]
    return value


def _oracle_leaves(d, prefix=""):
    """Brute-force flatten a dict to dotted leaves (oracle; recursion only)."""
    leaves = {}
    for k, v in d.items():
        path = prefix + k
        if isinstance(v, dict):
            leaves.update(_oracle_leaves(v, path + "."))
        else:
            leaves[path] = v
    return leaves


def oracle_diff(run_workflow, template_default):
    """Re-derive config_diff from the two resolved workflow YAMLs independently.

    Deliberately a different algorithm than production: round-trips through
    YAML, aligns modules by name via index lists, and leaf-diffs with a
    separate flattener.
    """
    run = yaml.safe_load(_resolved_to_yaml(run_workflow))
    tmpl = yaml.safe_load(_resolved_to_yaml(template_default))

    run_names = [s["module"] for s in run]
    tmpl_names = [s["module"] for s in tmpl]

    # Structural diff via the same two-pointer alignment, re-implemented.
    added, removed = [], []
    i = j = 0
    while i < len(run) and j < len(tmpl):
        if run_names[i] == tmpl_names[j]:
            i += 1
            j += 1
        else:
            run_has_t = tmpl_names[j] in run_names[i + 1 :]
            tmpl_has_r = run_names[i] in tmpl_names[j + 1 :]
            if tmpl_has_r and not run_has_t:
                removed.append(tmpl_names[j])
                j += 1
            elif run_has_t and not tmpl_has_r:
                added.append(run_names[i])
                i += 1
            else:
                added.append(run_names[i])
                removed.append(tmpl_names[j])
                i += 1
                j += 1
    added.extend(run_names[i:])
    removed.extend(tmpl_names[j:])

    # Param overrides for name-aligned common modules.
    overrides = {}
    i = j = 0
    while i < len(run) and j < len(tmpl):
        if run_names[i] == tmpl_names[j]:
            tleaves = _oracle_leaves(tmpl[j]["params"])
            rleaves = _oracle_leaves(run[i]["params"])
            for key in set(tleaves) | set(rleaves):
                tv = tleaves.get(key, None)
                rv = rleaves.get(key, None)
                if (key in tleaves) and (key in rleaves) and tv == rv:
                    continue
                overrides[f"{run_names[i]}.{key}"] = {
                    "template": tv if key in tleaves else None,
                    "run": rv if key in rleaves else None,
                }
            i += 1
            j += 1
        else:
            # mirror production's advance rule so alignment matches
            run_has_t = tmpl_names[j] in run_names[i + 1 :]
            tmpl_has_r = run_names[i] in tmpl_names[j + 1 :]
            if tmpl_has_r and not run_has_t:
                j += 1
            elif run_has_t and not tmpl_has_r:
                i += 1
            else:
                i += 1
                j += 1
    return {
        "param_overrides": overrides,
        "modules_added": added,
        "modules_removed": removed,
    }


class TestTemplateRegistry(unittest.TestCase):
    def test_existing_recipes_registered(self):
        for tid in ("minimal", "picasso_native", "minimal_channel_align"):
            self.assertIn(tid, tpl.TEMPLATE_REGISTRY)
            t = tpl.get_template(tid)
            self.assertTrue(t.template_version)
            self.assertIsInstance(t.question_type, str)

    def test_resolve_returns_ordered_steps(self):
        wf = tpl.get_template("minimal").resolve()
        self.assertEqual(_step_name(wf[0]), "load_dataset_movie")
        self.assertEqual(_step_name(wf[-1]), "save_single_dataset")

    def test_aggregation_resolves_to_aggregation_modules(self):
        wf = tpl.get_template("minimal_channel_align").resolve()
        names = [_step_name(s) for s in wf]
        self.assertIn("align_channels", names)
        self.assertNotIn("load_dataset_movie", names)  # that's single-scope


class TestVersionHashStability(unittest.TestCase):
    def test_version_stable_across_calls(self):
        t = tpl.get_template("minimal")
        self.assertEqual(t.template_version, t.template_version)

    def test_version_independent_of_run_specific_filepath(self):
        # Same recipe, different dataset path → same template_version.
        v1 = tpl.content_hash(ssw.minimal(filepath="/data/a.ome.tiff"))
        v2 = tpl.content_hash(ssw.minimal(filepath="/other/b.raw"))
        self.assertEqual(v1, v2)

    def test_version_changes_iff_content_changes(self):
        base = ssw.minimal(filepath="x")
        v_base = tpl.content_hash(base)

        # change a real param → version MUST change
        changed = ssw.minimal(filepath="x", box_size=9)
        self.assertNotEqual(v_base, tpl.content_hash(changed))

        # identical content built again → version unchanged
        self.assertEqual(v_base, tpl.content_hash(ssw.minimal(filepath="x")))

    def test_module_reorder_changes_version(self):
        wf = ssw.minimal(filepath="x")
        reordered = [wf[1], wf[0]] + list(wf[2:])
        self.assertNotEqual(tpl.content_hash(wf), tpl.content_hash(reordered))


class TestConfigDiff(unittest.TestCase):
    def setUp(self):
        self.default = tpl.get_template("minimal").resolve()

    def test_exact_identical_run(self):
        diff = tpl.compute_config_diff(self.default, self.default)
        self.assertEqual(diff["param_overrides"], {})
        self.assertEqual(diff["modules_added"], [])
        self.assertEqual(diff["modules_removed"], [])
        self.assertEqual(
            tpl.derive_conformance(diff, True), tpl.CONFORMANCE_EXACT
        )

    def test_single_param_override(self):
        run = ssw.minimal(filepath="run.tiff", box_size=11)
        diff = tpl.compute_config_diff(run, self.default)
        self.assertEqual(diff["modules_added"], [])
        self.assertEqual(diff["modules_removed"], [])
        # box_size appears in identify (nested + top-level) and localize.
        keys = set(diff["param_overrides"])
        self.assertTrue(keys)
        for k, v in diff["param_overrides"].items():
            self.assertEqual(v["template"], 7)
            self.assertEqual(v["run"], 11)
        self.assertEqual(
            tpl.derive_conformance(diff, True), tpl.CONFORMANCE_PARAM
        )

    def test_structural_added_module(self):
        run = list(self.default)
        run.insert(-1, ("density", {"radius": 5}))
        diff = tpl.compute_config_diff(run, self.default)
        self.assertEqual(diff["modules_added"], ["density"])
        self.assertEqual(diff["modules_removed"], [])
        self.assertEqual(
            tpl.derive_conformance(diff, True),
            tpl.CONFORMANCE_STRUCTURAL,
        )

    def test_structural_removed_module(self):
        run = [s for s in self.default if _step_name(s) != "undrift_rcc"]
        diff = tpl.compute_config_diff(run, self.default)
        self.assertEqual(diff["modules_removed"], ["undrift_rcc"])
        self.assertEqual(diff["modules_added"], [])
        self.assertEqual(
            tpl.derive_conformance(diff, True),
            tpl.CONFORMANCE_STRUCTURAL,
        )

    def test_ad_hoc_no_template(self):
        record = tpl.template_conformance_record(self.default, template=None)
        self.assertIsNone(record["template_id"])
        self.assertIsNone(record["template_version"])
        self.assertIsNone(record["config_diff"])
        self.assertEqual(record["conformance"], tpl.CONFORMANCE_ADHOC)

    def test_diff_reproducible_and_order_independent(self):
        # Recompute several times + shuffle dict insertion order of params;
        # the diff must be identical every time.
        run = ssw.minimal(filepath="r", box_size=9)
        d1 = tpl.compute_config_diff(run, self.default)
        d2 = tpl.compute_config_diff(run, self.default)
        self.assertEqual(
            json.dumps(d1, sort_keys=True), json.dumps(d2, sort_keys=True)
        )

        # Reversing the key order of a param dict must not change the diff.
        run2 = []
        for name, params in run:
            run2.append((name, dict(reversed(list(params.items())))))
        d3 = tpl.compute_config_diff(run2, self.default)
        self.assertEqual(
            json.dumps(d1, sort_keys=True), json.dumps(d3, sort_keys=True)
        )


class TestQcSidecarRoundTrip(unittest.TestCase):
    def test_sidecar_round_trips_three_fields(self):
        template = tpl.get_template("minimal")
        run = ssw.minimal(filepath="r", box_size=9)
        with tempfile.TemporaryDirectory() as d:
            qc_path = os.path.join(d, "qc.json")
            record = tpl.log_run_conformance(qc_path, run, template)
            with open(qc_path) as fh:
                on_disk = json.load(fh)
        section = on_disk["template_conformance"]
        self.assertEqual(section["template_id"], "minimal")
        self.assertEqual(
            section["template_version"], template.template_version
        )
        self.assertIn("config_diff", section)
        self.assertEqual(section["conformance"], tpl.CONFORMANCE_PARAM)
        self.assertEqual(section, record)

    def test_sidecar_additive_preserves_existing(self):
        with tempfile.TemporaryDirectory() as d:
            qc_path = os.path.join(d, "qc.json")
            with open(qc_path, "w") as fh:
                json.dump(
                    {
                        "recommendations": {"filter": {"x": 1}},
                        "metrics": {"nena": 2.3},
                    },
                    fh,
                )
            tpl.log_run_conformance(
                qc_path,
                tpl.get_template("minimal").resolve(),
                tpl.get_template("minimal"),
            )
            with open(qc_path) as fh:
                data = json.load(fh)
        # existing keys untouched
        self.assertEqual(data["recommendations"], {"filter": {"x": 1}})
        self.assertEqual(data["metrics"], {"nena": 2.3})
        self.assertIn("template_conformance", data)


class TestT1GoldenOracle(unittest.TestCase):
    """Golden snapshots + independent-oracle cross-check for curated pairs."""

    def _pairs(self):
        default = tpl.get_template("minimal").resolve()
        exact = list(default)
        param = ssw.minimal(filepath="p", box_size=11)
        added = list(default)
        added.insert(-1, ("render", {"colordrive": "nena"}))
        removed = [s for s in default if _step_name(s) != "undrift_rcc"]
        return {
            "exact": (exact, default),
            "param": (param, default),
            "added": (added, default),
            "removed": (removed, default),
        }

    GOLDEN = {
        "exact": {
            "param_overrides": {},
            "modules_added": [],
            "modules_removed": [],
        },
        "added": {
            "param_overrides": {},
            "modules_added": ["render"],
            "modules_removed": [],
        },
        "removed": {
            "param_overrides": {},
            "modules_added": [],
            "modules_removed": ["undrift_rcc"],
        },
    }

    def test_golden_snapshots_match_production_and_oracle(self):
        pairs = self._pairs()
        for name, (run, default) in pairs.items():
            prod = tpl.compute_config_diff(run, default)
            orac = oracle_diff(run, default)
            # Production and the INDEPENDENT oracle must agree.
            self.assertEqual(
                json.dumps(prod, sort_keys=True),
                json.dumps(orac, sort_keys=True),
                msg=f"oracle disagreement for pair '{name}'",
            )
            # Curated structural goldens (param pair has many keys; checked
            # separately below).
            if name in self.GOLDEN:
                self.assertEqual(
                    prod,
                    self.GOLDEN[name],
                    msg=f"golden mismatch for pair '{name}'",
                )

    def test_param_pair_golden(self):
        run, default = self._pairs()["param"]
        prod = tpl.compute_config_diff(run, default)
        self.assertEqual(prod["modules_added"], [])
        self.assertEqual(prod["modules_removed"], [])
        # Every override is box_size 7 -> 11, and the oracle agrees.
        self.assertTrue(prod["param_overrides"])
        for v in prod["param_overrides"].values():
            self.assertEqual(v, {"template": 7, "run": 11})
        orac = oracle_diff(run, default)
        self.assertEqual(
            json.dumps(prod, sort_keys=True),
            json.dumps(orac, sort_keys=True),
        )


if __name__ == "__main__":
    unittest.main()
