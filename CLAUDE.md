# CLAUDE.md — picasso-workflow

Standing context for Claude Code (claude.ai/code) working in this repo. Read
this first, then the design doc and the cross-repo pointers below. picasso-workflow
is one repo in the **DNA-PAINT full-automation** stack (siblings: PycroFlow,
monet, picasso, picasso-registry, picasso-agent).

## What this repo is

picasso-workflow automates and documents DNA-PAINT **analysis** workflows built
on `picassosr` (the picasso localization/clustering library). It runs a picasso
pipeline as an ordered list of named **modules** — two workflow kinds:

- **Single-dataset** — one movie is loaded, localized, drift-corrected,
  clustered, etc. (`WorkflowRunner`).
- **Aggregation / investigation** — many datasets each run the same
  single-dataset modules, then are combined by aggregation modules
  (`AggregationWorkflowRunner`). Module parameters can vary per channel via a
  `("$$map", "<column>", <default>)` command backed by a column in
  `single_dataset_tileparameters` (see the README "Per-channel parameters").

Each run documents itself to **Confluence** (`ConfluenceReporter`) and/or a
local **HTML report** (`html_reporter`). A **PyQt6 GUI** (`picasso-workflow-gui`)
builds, edits, and launches workflows, and can generate SLURM scripts to submit
them to a cluster. The importable package is `picasso_workflow/`.

See `README.md` for the full feature list, installation, per-channel parameters,
the four-tier testing strategy, the SLURM cluster runner, CI, and the release
workflow — this file is the short standing context, not a duplicate of it.

## Current branch

`feature-FullAutoS0A` — PRs target `master`. (Upstream also maintains a
`develop` branch; the release workflow merges `develop` → `master` and tags on
`master` — see README "Releasing".)

## Commands

```bash
# Install (core; pulls picassosr and the PyQt6 GUI deps). Do this in a
# python>=3.10 env (README recommends a conda env named picasso-workflow).
pip install -e .
pip install -e ".[cluster]"    # adds mpi4py for the MPI aggregation path

# For a local development picasso, install it first, then picasso-workflow:
#   cd /path/to/picasso && pip install -r requirements.txt

# Run the GUI (console script and module form are equivalent)
picasso-workflow-gui
python -m picasso_workflow.gui

# Test (a bare `pytest` is UNIT-ONLY — the `integration` mark is deselected by
# default in [tool.pytest.ini_options] addopts). See README "Testing".
pytest                         # tiers 1+2: unit + template validation (fast)
pytest -v                      # verbose, still unit-only
pytest -k <name>               # a single test by keyword
pytest -m integration          # tier 3: real picasso pipeline (needs picassosr)
pytest -m "integration and real_data"   # tier 4: real acquired data (PW_TEST_DATA_DIR)
pytest -m ""                   # clear the default filter: run everything

# Lint / format — pre-commit owns black + flake8 (both run in pre-commit's own
# isolated envs; they are NOT direct package deps, so `pre-commit` is the
# canonical path rather than a bare `black`/`flake8`).
pre-commit install             # once, to run hooks on every commit
pre-commit run --all-files     # run all hooks (trailing-ws, eof, yaml, black, flake8) now
```

## Conventions (aligned across the DNA-PAINT automation repos)

This repo already matches the aligned target — no S0A-2 migration pending here.

- **Style:** Black, line length **79** (Black owns line wrapping). Config in
  `pyproject.toml [tool.black]`.
- **Lint:** flake8 via **Flake8-pyproject**, config in
  `pyproject.toml [tool.flake8]` with `extend-ignore = E203, E501, W503` — E501
  is ignored because **Black owns line length** (it already wraps code; long
  strings/comments/HTML it can't split are intentional). `max-line-length = 88`
  there is informational only. Some experimental modules and the test-data
  fixtures are `extend-exclude`d.
- **Pre-commit:** `pre-commit install` once; hooks run trailing-whitespace,
  end-of-file-fixer, check-yaml, check-added-large-files, **black**, and
  **flake8** (Flake8-pyproject). isort / bandit / mypy are intentionally **not**
  in the pre-commit run (see `.pre-commit-config.yaml`).
- **Versioning:** **setuptools-scm** — **the tag IS the version**; there is no
  version string to edit by hand. It writes `picasso_workflow/_version.py`
  (gitignored, importable as `picasso_workflow.__version__`); fallback outside a
  git checkout is `0.3.3.dev0`. Release = merge to `master`, then
  `git tag vX.Y.Z && git push origin vX.Y.Z` (format `vMAJOR.MINOR.PATCH`).
- **Changelog on release:** the changelog is `CHANGELOG.md` at the repo root
  (Keep a Changelog, in Markdown, with `### Added` / `### Changed` / `### Fixed`
  subsections — matching monet / PycroFlow / picasso-registry). Add an entry
  under the top **`## [Unreleased]`** section in every PR; at release, promote
  `[Unreleased]` to a dated, tagged section (e.g. `## [1.2.3] - YYYY-MM-DD`). Because the version comes from git tags, the changelog
  is the human-facing record of what each tag contains.
- **Packaging:** `pyproject.toml` only (no `setup.py` / `setup.cfg`). Runtime
  deps and the `[cluster]` extra live there.
- **Tests:** write/extend tests with every change; keep every tier green.
  Picasso is fully mocked in the unit tier so it runs anywhere with no data or
  network. Adding a workflow module touches
  `util.AbstractModuleCollection`, `analyse.AutoPicasso`,
  `confluence.ConfluenceReporter`, and the matching `tests/test_*` files; if a
  snapshotted template references it, re-run `python tools/snapshot_templates.py`
  (see README "Adding a new workflow module").

## Architecture (short)

`workflow.py` holds the orchestrators (`WorkflowRunner`,
`AggregationWorkflowRunner`): each reads a module list, calls the corresponding
analysis method, and records results/failures to `WorkflowRunner.yaml`.
`analyse.py` (`AutoPicasso`) implements the actual picasso-backed modules;
`util.py` provides `AbstractModuleCollection` (the module contract),
`ParameterTiler` / `ParameterCommandExecutor` (the `$`/`$$map` per-channel
parameter machinery), and typing helpers. `standard_singledataset_workflows.py`
and `standard_aggregation_workflows.py` are the predefined recipes;
`modulespec.py` is the `ModuleSpec` annotation/validation layer;
`picasso_outpost.py` holds picasso-adjacent code not yet upstream. Reporting:
`confluence.py` (`ConfluenceReporter` / `ConfluenceInterface`) and
`html_reporter.py`. `_launcher.py` is the `picasso-workflow-gui` entry point;
`__init__.py` configures loguru logging and deep-merges `config.yaml`
(package → site → per-user). Full module map in `README.md`.

## Standing pointers

Paths so later sessions can `@`-reference them. Repo root is
`/workspaces/DNA-PAINT-FullAutomation/repositories/picasso-workflow`; the shared
workspace root is `/workspaces/DNA-PAINT-FullAutomation`.

**Live (resolve today):** the shared planning docs live in `../../planning/`
(workspace `planning/` folder); start from its document map.
- Document map / reading order: `../../planning/README.md`
- Design doc — recommendation & roadmap (strategy, prioritized initiatives #1–#9,
  work packages WP-1–WP-16, Parts I–X):
  `../../planning/DNA-PAINT_Automation-Recommendation.md`
- **Playbook** — Claude Code implementation playbook (operating model, Step 0
  foundations, style/repo alignment, gated dependency-ordered work orders):
  `../../planning/DNA-PAINT_ClaudeCode-Implementation-Playbook.md`
- **Work-order briefs** — self-contained, paste-ready briefs (S0A-1, S0A-2,
  S0B-1/2, WP-1…WP-16); this task is S0A-1:
  `../../planning/DNA-PAINT_Work-Order-Briefs.md`
- **Progress tracker** — tick-off worksheet + gates for the work orders:
  `../../planning/DNA-PAINT_Implementation-Progress-Tracker.md`
- Module-annotations reference (the `ModuleSpec` layer, data dependencies,
  capability registry):
  `../../planning/picasso-workflow_Module-Annotations_Reference.md`
- Dev-environment setup (OrbStack dev-container):
  `../../planning/DNA-PAINT_ClaudeCode-DevEnvironment.md`
- Sibling repo standing context:
  - PycroFlow (experiment orchestration): `../PycroFlow/CLAUDE.md`
  - monet (laser-power calibration/control): `../monet/CLAUDE.md`
  - picasso-registry (provenance/metrics DB; **owns the schema/API contract**):
    `../picasso-registry/CLAUDE.md`
  - picasso-agent (agentic layer): `../picasso-agent/CLAUDE.md`
  - picasso (upstream localization/clustering library — `picassosr`; no CLAUDE.md
    yet): `../picasso`
- Sibling repo roots: `../PycroFlow`, `../monet`, `../picasso`,
  `../picasso-registry`, `../picasso-agent`

**Forthcoming (planned; not yet in-tree — do not treat as resolvable):**
- Cross-repo contracts (after S0B): the picasso-registry OpenAPI spec + generated
  client and the shared schemas (metric-vector, workflow-YAML,
  `localize_frames` signature, picasso-workflow `ModuleSpec`) — these will be
  owned by picasso-registry; see `../picasso-registry/CLAUDE.md` and work orders
  S0B-1 / S0B-2 in the briefs above. picasso-workflow's `ModuleSpec`
  (`modulespec.py`) is part of that contract set.

## Notes for editing

- `.gitignore` **tracks this `CLAUDE.md`** (it is not ignored) but keeps
  `.claude/` and `CLAUDE.local.md` ignored (local settings / personal notes) —
  keep it that way.
- `picasso_workflow/_version.py` is generated by setuptools-scm and gitignored —
  never commit or hand-edit it.
- `spinna_mle.py`, `spinna_mle_2.py`, and `nn_redistribution.py` are isolated
  experimental modules (not imported by the package) and are flake8-excluded;
  don't expect them to be linted.
- **GPU fitting on the cluster** needs the `[gpu]` extra
  (`pip install -e ".[gpu]"` → `numba-cuda` + `cuda-bindings`); nothing in the
  base deps or `picassosr` pulls it, and without it numba's bundled `numba.cuda`
  **segfaults** at CUDA-context creation on CUDA-13 drivers (a native SIGSEGV,
  not a catchable error) while `nvidia-smi` looks fine. A recurring companion
  trap across these repos: a `~/.local` user-site install shadows every conda
  env on `sys.path`; batch jobs are immune (`assemble_slurm_commands` exports
  `PYTHONNOUSERSITE=1`), interactive shells / the GUI are not. Verify real env
  state with `PYTHONNOUSERSITE=1 python -c "import numba; print(numba.__file__)"`.
  See README "GPU-accelerated fitting".
