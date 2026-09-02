# Changelog

All notable changes to picasso-workflow are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions are derived from git tags by setuptools-scm, so entries are collected
under `[Unreleased]` until a tag is cut.

This file was started after v0.5.6; earlier history is in the git log.

## [Unreleased]

### Fixed

- Live-progress tree no longer collapses on every refresh: the user's
  expand/collapse of the aggregation root and stage groups is captured
  (keyed by dataset slot) before the per-poll rebuild and reapplied after, so
  investigating a stage's details is not undone on the next 15 s refresh.
- SLURM logs are now reliably captured: the generated submission creates the
  `logs/` directory before submitting (SLURM opens `--output`/`--error` at
  launch and does not create their parent, so a missing dir silently dropped
  the job's stdout/stderr — where a traceback / OOM message lands), and the
  log filenames use `%j` (job id) instead of `%A` (array-master id).
- Live-progress monitor no longer shows a previous run's progress while a
  resubmitted job is still pending. The results folder accumulates one
  subfolder per run, so polling the tree returned every past run's
  `progress.json`; the monitor now scopes the states to the current SLURM job
  id (the run token woven into every `report_name`), so a pending resubmission
  shows PENDING with an empty tree rather than the last run's completed stages.
- Generated SLURM scripts now exit with the workflow's real status: the `srun`
  step's exit code is captured (`PW_RC=$?`) and the batch script ends with
  `exit ${PW_RC:-0}`. Previously the trailing `echo` made the batch script
  exit 0 even when the `srun` step was cancelled/killed (e.g. SIGTERM, MPI
  teardown), so SLURM misreported a dead run as `COMPLETED`.
- Live-progress monitor: a job SLURM reports `COMPLETED` while the tracked
  progress never reached 100% is now flagged amber as "COMPLETED but workflow
  unfinished (stopped at <module>)" instead of a misleading green COMPLETED.
- Live-progress box layout: only the module tree grows when the box is
  resized vertically; the job-state chip and the overall progress bar keep
  their natural height (fixed vertical size policy; the tree gets the stretch).

### Added

- `load_dataset_movie`: new optional `stage_to_local` flag (GUI checkbox, off
  by default) that copies the movie to node-local scratch (`$SLURM_TMPDIR` /
  `$TMPDIR`, auto-cleaned by SLURM) before loading, so the per-frame reads of
  identify/localize are local instead of over a slow/edgy shared filesystem —
  the fix for identify/localize crawling or hanging on large movies read over
  NFS. Split OME-TIFF/MMStack series and `.raw` sidecars are staged as a set;
  best-effort, falling back to the network path on any error.
- `localize` now logs an anchor line at the start of the fit (method, spot
  count, box, multiprocess) and at the end (localization count, elapsed time,
  spots/s), and wires picasso's spot-extraction progress
  (`cut_progress_callback`) in addition to the fit `progress_callback`. The
  extraction phase — which silently dominates a large movie on a slow
  filesystem — now reports forward motion too, so a slow localize is legible
  in the SLURM log instead of looking hung.
- `zfit` gained the same fail-fast GPU guard as `localize`: when `gpu=True`
  but `numba.cuda.is_available()` is False, it raises an actionable
  `AutoPicassoError` up front (pointing at the CUDA-toolkit / `module load
  cuda` fix or `gpu=False`) instead of aborting deep inside picasso.

### Fixed

- `identify` / `localize` now fall back to picasso's own defaults for unset
  optional arguments instead of forwarding the GUI's empty/minimum sentinels.
  Previously an unset field was passed through verbatim, so `''` roi /
  frame_bounds reached picasso as empty strings and, worse,
  `temporal_median_window: 1` (a no-op median) switched the temporal-median
  background filter *on* — forcing the slow filtered read path and making
  `identify` crawl on large movies. Now roi/frame_bounds are forwarded only
  when non-empty, the temporal-median filter only when its window is >= 2
  (stride only alongside it), the Gaussian pre-filter only for sigma > 0, an
  empty `fitting_method` resolves to the default, and non-positive `eps` /
  `max_it` become None (= picasso's per-method default). The corresponding GUI
  fields use 0 as the "off / use default" sentinel (`temporal_median_window`,
  `temporal_median_stride`, `max_it` minimum lowered from 1 to 0).

### Added

- **Progress tracking** for workflow runs, overall and per module. A new
  `picasso_workflow/progress.py` provides a `ProgressManager` emitter that the
  runners drive at module boundaries; it fans updates out to sinks — an atomic
  `progress.json` written into the run's result folder (the universal, cross-
  process source of truth) and a `[progress]` log line. `WorkflowRunner`
  records per-module status (`pending`/`running`/`done`/`failed`/`skipped`)
  and timing; `AggregationWorkflowRunner` tracks per-dataset state (rank 0
  owns `progress.json`, worker ranks write `progress.rank<N>.json`).
- **Intra-module progress**: the long picasso calls (`identify`, `localize`
  fit, `smlm_clusterer`, `undrift_rcc`, `undrift_aim`) now forward their
  frame/spot/segment counts through picasso's progress callbacks, converted to
  a 0..1 fraction via `PicassoProgressProxy`. Wiring is inert unless the runner
  attaches a callback, so behaviour (and unit tests) are unchanged when
  progress tracking is off.
- **Cooperative abort**: a run stops gracefully at the next module boundary
  (and inside long picasso calls, via `abort_callback`) when an `abort.flag`
  file appears in its result folder, complementing a hard `scancel`.
- **GUI live progress monitor** (Run tab): a SLURM-state chip, an overall
  progress bar and a per-module/per-dataset tree, refreshed every **15 s**
  (chosen because each SSH round-trip is slow). It fuses two sources —
  `squeue`/`sacct` for the job's liveness and terminal cause (exit code, peak
  memory on OOM) and the run's `progress.json` for which module and how far —
  so a killed job reads as e.g. "OUT_OF_MEMORY during module 3/8 localize"
  rather than a frozen bar. `SlurmCommunicator` gains `fetch_progress()` and
  `write_abort_flag()`; **Cancel Job** now requests a graceful abort before
  `scancel`.
- For **aggregation** runs the monitor now shows *all stages* as a labelled,
  collapsible tree — an `[Aggregation]` root (datasets N/M) with one
  `[single NN] <tag>` child per dataset (each expandable to its modules) and
  a final `[aggregation stage]` node — instead of one unlabelled single
  workflow. All stages are fetched in a single SSH round-trip
  (`SlurmCommunicator.fetch_all_progress` / `progress.read_all_progress`), and
  the overall bar blends per-dataset progress with the aggregation stage.
- The monitor can now **attach to a run started in a previous GUI session**:
  the poll derives its target from the Run-tab fields rather than only from a
  same-session submission. Enter the results folder (plus the job ID and
  cluster host for a cluster run, or leave the job ID blank to watch a local
  results folder) and click *Refresh now*. The SSH connection is built lazily
  and cached per host (shared with submission).
- **Local execution** (Run tab): *Start Workflow locally* now actually runs
  the generated `start_workflow.py` as a subprocess and drives the same
  monitor from the local `progress.json` (previously a stub).

- Run tab (SLURM): new **Partition** dropdown, populated per cluster from the
  new `SlurmPartitions` config section (editable, so an unlisted partition can
  be typed). The chosen partition is emitted as `#SBATCH --partition=…` in the
  generated SLURM script. GPU nodes usually live in a dedicated partition, so
  this is required for a `--gres=gpu` request (e.g. `spline-mle-gpu`) to
  schedule. `SlurmDefault` gains `partition` (preselected default) and `gpus`
  keys.
- `localize` module: fail-fast guard for GPU fitting. When a resolved
  `fitting_method` ends in `-gpu` but `numba.cuda.is_available()` is False
  (e.g. libNVVM / CUDA toolkit missing), the module now raises an actionable
  `AutoPicassoError` up front instead of aborting deep inside picasso after a
  long spot-extraction, and points at the CUDA-toolkit / `module load cuda`
  fix or dropping the `-gpu` suffix to fit on the CPU.
- SLURM job scripts now `module load` environment modules and export
  `PYTHONNOUSERSITE=1`. This makes GPU fitting (`spline-mle-gpu`) work on
  module-based HPC systems: a CUDA module provides libNVVM and sets
  `CUDA_HOME` so numba's `cuda.is_available()` is True, and disabling
  `~/.local` site-packages stops a stray user-site install from shadowing the
  conda env. The modules are editable on the Run tab in a new **Modules**
  field (space-separated), prefilled per cluster from
  `ClusterEnvironment.<host>.Modules` (e.g. `cuda/13.0`); a blank field loads
  nothing.

- `localize` module: new optional `fitting_method` parameter exposing the
  picasso 0.11 fitting models (`gausslq` (default), `gaussmle`, the
  `-rotated` / `-spherical` Gaussian variants, `spline` experimental-PSF
  fitting, and their `-gpu` counterparts). Spline fitting additionally
  accepts a `spline_calibration` (a dict or a path to a picasso spline
  calibration file) and yields z (3D) directly, and the fitter's `eps`
  (convergence) and `max_it` (iteration cap) are now exposable. The
  resolved fit method is recorded in the results and the Confluence report.
  These parameters (`fitting_method`, `spline_calibration`,
  `camera_calibration`, `eps`, `max_it`) are also selectable in the GUI's
  `localize` module form — `spline_calibration` is where the spline-PSF
  calibration file is entered.
- `identify` module: now uses the picasso 0.11 threaded `localize.identify`
  entry point and exposes its new background-suppression and scoping
  options — `temporal_median_window` / `temporal_median_stride` (temporal
  median filter), `gaussian_filter_sigma` (spatial Gaussian pre-filter),
  and one-or-more `roi` / `frame_bounds`. `identify_parallel` toggles
  multi-core detection. Unset options fall back to the picasso defaults
  (no filtering, whole movie), so existing workflows are unaffected. These
  options are also selectable in the GUI's `identify` module form.
- `zfit` module: new optional `fitting_method` (default `auto` — inferred
  from the `"Fit method"` the `localize` module recorded, so it need not be
  set to match by hand; override with `gausslq`/`gaussmle`), used by picasso
  0.11 to compute the axial localization precision. Also new `gpu` (fit z on
  a CUDA device) and `filter` (z-fit RMSD filter) parameters, all exposed in
  the GUI. Guards against `zfit.zfit` returning no localizations instead of
  crashing on the z histogram.
- GUI: parameter forms support conditional visibility via a `visible_if`
  spec key — a field is shown only while a controlling parameter holds one of
  the listed values. Used so the `localize` `spline_calibration` field
  appears only for the `spline` fitting methods. `_load_calibration` also
  tolerates an empty-string path (treats it as unset) so leaving an optional
  calibration field blank in the GUI no longer errors.
- `localize` module: new optional `camera_calibration` parameter (a dict or
  a path to a picasso camera calibration file) enabling the picasso 0.11
  per-pixel sCMOS noise model during fitting.
- New `register_channels` aggregation module: fits a higher-degree-of-freedom
  channel transform (affine / projective / polynomial) from fiducial-bead
  movies via `picasso.registration` and warps each channel's localizations
  into the reference frame. Complements the existing translation-only
  `align_channels`. (3D localization is available via the `spline` fitting
  method above and, for astigmatism, the existing `zfit` module.) Note: the
  transform math is unit-tested with identity transforms and mocks; real
  bead-data validation belongs to the integration tier.

- CI: a hosted unit-test workflow (`.github/workflows/unit-tests-hosted.yml`,
  `ubuntu-latest`) that installs the wheel-only base, pulls the Qt runtime libs
  and runs the unit tier headless (`QT_QPA_PLATFORM=offscreen`). Intended as the
  required merge gate alongside Lint, so PRs no longer depend on the self-hosted
  Windows/SLURM runners being online.
- Confluence error reports now identify the failing module by index and
  name in a heading, list the parameters it was called with, name the
  innermost picasso-workflow stack frame, and link the module result
  folder and the preceding module's results. Previously only the
  exception message and traceback were posted, so diagnosing a failure
  meant guessing which parameter value had caused it.
- Failed modules are recorded in `WorkflowRunner.yaml` with their
  index, parameters, exception type, message and traceback.
- Per-channel parameters: module parameters can differ between channels
  of an aggregation workflow via a `("$$map", "<column>", <default>)`
  command backed by a column in `single_dataset_tileparameters`. The
  GUI shows the resolved per-channel values beneath the parameter row
  and round-trips the dataset table through the generated script.

### Changed

- Bumped the picasso pin to `picassosr>=0.11.0` (final release; resolves
  from PyPI).
- Migrated the `localize` module's gauss fitting off the deprecated
  `picasso.gausslq` API (removed/deprecated in picasso 0.11: the
  `fit_spots_gpufit` / `locs_from_fits_gpufit` GPU helpers were deleted, and
  the whole module now warns and is slated for removal in picasso 1.0). It now
  calls the high-level `picasso.localize.fit`, selecting `gausslq-gpu` vs
  `gausslq` and single- vs multi-process from `gpufit_installed` and the
  `fit_parallel` parameter. The recorded "Fit method" now reflects the actual
  method used.
- The Zeiss `.czi` reader (`aicsimageio` + `aicspylibczi` + `fsspec`)
  moved from the base dependencies to an optional `formats` extra
  (`pip install "picasso_workflow[formats]"`). aicspylibczi has no
  aarch64 wheels and source-builds via a C++/cmake toolchain, and
  aicsimageio dragged an old imagecodecs that failed to build on the
  py3.10 arm64 container, so a bare `pip install -e .` now resolves
  entirely from wheels. The `convert_zeiss_movie` module raises a clear
  ImportError pointing at the extra when the reader is absent; no other
  workflow is affected.
- `estimate_density_from_neighbordists` now validates the
  `[min_dist, max_dist]` window per neighbour order and fails with a
  message naming the parameter and the surviving counts, e.g.
  `min_dist=50, max_dist=300 leaves 0 of 530 k=4 nearest-neighbour
  distances (observed range 338.3-2.264e+04)`.

  BEHAVIOUR CHANGE: runs in which some neighbour order had no distances
  inside the window previously fitted on the remaining orders and now
  raise instead. Workflows believed to be healthy may start failing;
  widen the window or reduce the number of neighbours fitted.
- Tracebacks are posted inside a Confluence code macro, so their line
  structure is preserved. They were previously HTML-escaped without a
  wrapper and reflowed into a single unreadable paragraph.
- CI now runs a `black --check` + `flake8` lint job on a GitHub-hosted
  runner (`.github/workflows/lint.yml`), so style regressions are caught
  in CI and not only by the local pre-commit hook. Linter versions are
  pinned to match `.pre-commit-config.yaml`.

### Fixed

- The new `register_channels` module was missing its GUI descriptor, so
  opening the GUI raised `TypeError: Can't instantiate abstract class
  ModuleDescriptor with abstract method register_channels`. Added the
  `ModuleDescriptor.register_channels` descriptor and a regression test
  asserting `ModuleDescriptor` implements every abstract module — the GUI
  test `window` fixture had been silently *skipping* this class of failure
  (it treats any `Window()` construction error as a skip).
- `picasso_outpost.convert_zeiss_file` was left half-refactored by the
  earlier "aicsimageio removal" change: it still constructed an unused
  `AICSImage` and referenced an undefined `data`, so it raised
  `NameError`/`F821` and failed both `flake8` (lint gate) and its unit test.
  It now reads the `.czi` natively via picasso's `io.load_czi`
  (`data = movie[:].squeeze()`) and the dead `aicsimageio` import was removed.
  (The now-unused `[formats]` optional dependency can be dropped as a
  follow-up.)
- Pinned `atlassian-python-api>=3.41,<5`. The previously unpinned dependency
  resolved to 5.0.x, whose Confluence-client restructure (Cloud/Server
  subclasses, `get_page_by_title` `space` → `space_key`, relocated
  `create_page` / `update_page` / `attach_file` / … methods) broke
  `ConfluenceInterface` with a `TypeError`. The pin restores the API the
  reporter targets; migrating to 5.x is a separate follow-up. The live
  Confluence tests (`Test_A_ConfluenceInterface`, `Test_C_ConfluenceReporter`)
  are now `@pytest.mark.integration` so a bare `pytest` (unit tier) no longer
  runs them even when a `TEST_CONFLUENCE_TOKEN` is configured — they run under
  `pytest -m integration`, and still skip when no token is set.
- `PathParser` (the cross-machine file-path converter used by `spinna_batch`
  via `util.convert_filepath_for_machine`) raised `KeyError: 'Drivepaths'` on
  any machine whose config has no `Drivepaths` section — i.e. a fresh install /
  CI / any non-lab machine (only `config_template.yaml` ships, not a populated
  `config.yaml`). It now defaults to an empty drive map, which `convert_path`
  already treats as "no known drive root -> return the path unchanged". This
  makes the unit tier hermetic (fixes `test_analyse.py::test_modules` and
  `test_spinna_batch_single_dataset` off the lab machines) and stops the module
  from hard-crashing when no drive mapping is configured.
- An exception other than `AutoPicassoError` escaped
  `WorkflowRunner.run()` before `save()`, so the failing module left
  no trace in `WorkflowRunner.yaml`.
- A module raising `AutoPicassoError` on the first iteration of
  `run()` raised `UnboundLocalError` on `success`, masking the
  real error.
- `call_module` re-raised a `copy.copy()` of the exception, which
  drops `__traceback__`; the propagated error stopped at the re-raise
  rather than pointing at the code that failed.
- `fit_csr` used truthiness to detect optional parameters, so
  `min_dist=0`, `max_dist=0` and `bkg_fraction=0` were silently
  replaced by defaults.
- `nndistribution_from_csr` raised "zero-size array to reduction
  operation maximum" on an empty distance array instead of returning an
  empty result.
- The GUI's "Remove Dataset" button did nothing, silently, when a
  channel was selected, and its buttons were ordered inconsistently.
- Commands could not be assigned to nested (dict) sub-parameters: the
  `cmd` dialog raised `KeyError` on accept, and a nested command
  value was discarded when the workflow was reloaded.
