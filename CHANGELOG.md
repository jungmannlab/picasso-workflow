# Changelog

All notable changes to picasso-workflow are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions are derived from git tags by setuptools-scm, so entries are collected
under `[Unreleased]` until a tag is cut.

This file was started after v0.5.6; earlier history is in the git log.

## [Unreleased]

### Added

- `localize` module: new optional `fitting_method` parameter exposing the
  picasso 0.11 fitting models (`gausslq` (default), `gaussmle`, the
  `-rotated` / `-spherical` Gaussian variants, `spline` experimental-PSF
  fitting, and their `-gpu` counterparts). Spline fitting additionally
  accepts a `spline_calibration` (a dict or a path to a picasso spline
  calibration file) and yields z (3D) directly, and the fitter's `eps`
  (convergence) and `max_it` (iteration cap) are now exposable. The
  resolved fit method is recorded in the results and the Confluence report.
- `identify` module: now uses the picasso 0.11 threaded `localize.identify`
  entry point and exposes its new background-suppression and scoping
  options — `temporal_median_window` / `temporal_median_stride` (temporal
  median filter), `gaussian_filter_sigma` (spatial Gaussian pre-filter),
  and one-or-more `roi` / `frame_bounds`. `identify_parallel` toggles
  multi-core detection. Unset options fall back to the picasso defaults
  (no filtering, whole movie), so existing workflows are unaffected.

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
