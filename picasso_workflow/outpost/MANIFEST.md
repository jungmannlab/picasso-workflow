# Outpost migration manifest

Per-feature status for the Phase 2 reorganisation of picasso-workflow's
picasso-adjacent code. Source of truth for what moves where. See `README.md`
for the `keep / dedup / promote / extend-up / delete` vocabulary.

Readiness/effort: **H** = ready/clean, **M** = needs a refactor first, **L** =
WIP / not ready.

## Triage — two axes (2026-10-21)

Every outpost item sits on **two independent axes**. The
`keep/dedup/promote/extend-up/delete` statuses below encode the **destination**
axis; this section adds the **integration-health** axis for everything that
stays, and distils the resulting backlog. (Motivation: the `outpost/<feature>/`
layout made `render` *look* like an upstream candidate when it is actually
integration glue — a `keep`. The two axes keep that distinction explicit.)

**Axis 1 — Destination** (where the code belongs long-term):

- **Upstream-bound** (`promote` / `extend-up`) — genuinely novel; stage for
  picasso. Re-verified this pass: CSR k-NN stats, point-set shift kernel,
  correspondence-free ICP registration, resolution (FRC / decorrelation;
  **PPAC deferred** — never performed well, needs a perf rewrite before
  promotion, see S1), binding-events, CellMask morphology. None are
  glue-mislabeled.
- **Workflow-bound** (`keep`) — permanent glue / orchestration / reporting over
  picasso (channel-alignment, spinna wrappers, origami, dbscan-molint, render,
  Ripley's-K, report helpers). `render` moved here from the promote mindset.

**Axis 2 — Integration health** (for everything that stays — is it wired to
picasso *well*?). The **import boundary is already clean**: no direct
`import picasso` in production; every use goes through `_picasso_api`. What
remains is the keep-it-clean backlog:

| health issue | items | fix |
|---|---|---|
| **Fragile private reach-ins** — depend on picasso *private* symbols; guarded by `get_internal` + the contract test, but any upstream refactor can break them | `postprocess._pick_similar` (`pick_gold`), `postprocess._n_block_locs_at` (`pick_similar`), `localize._to_photons` (`get_spots`), `spinna._targets_from_structures` (`screen_label_uncertainty`) | upstream a **public** picasso API for each, then drop the `get_internal`. (The boundary + contract test already make the breakage *loud*, not silent — so this is hardening, not a live bug.) |
| **Duplication of picasso** — re-implements what picasso ≥ 0.12 provides | ~~**3 copies of `otsu()`**~~ **✅ deduped** → `picasso.masking.threshold_otsu` (all 3 deleted, callers rewired via `_picasso_api`, equivalence-pinned); remaining: `_undrift_from_picked` drift computation; `_solve_optimal_shifts` vs `picasso.lib.minimize_shifts`; `dbscan_molint/mask.py` precursor | equivalence-test, then rewire to upstream + delete the fork (see Dedup table). |
| **Monolith** — ~~promote-candidates and keep-glue physically interleaved in one file~~ | `picasso_outpost.py` — **DONE: split complete, file deleted (was 7579 lines; S1–S9)** | ✅ split into `outpost/<feature>/`; the last 2 report helpers moved to `outpost/reporting/histograms.py` (S9) and the file was removed. |

**Reading:** the upstream-bound set is the *promotion* backlog; the table above
is the *keep-it-clean* backlog. The monolith split (S1–S9) is **complete** —
the two kinds of code are now physically separated across `outpost/<feature>/`
modules, unblocking the promotion work.

## Done

| feature | source | status | notes |
|---|---|---|---|
| spinna_mle / spinna_mle_2 / nn_redistribution / spinna / spinna_io / spinna_main | `picasso_workflow/spinna*.py`, `nn_redistribution.py` | **delete** ✅ | Dead, never committed (gitignored local experiments); 0 imports anywhere. Removed from disk + gitignore + flake8-exclude + lint allowlist. |
| g5m (outpost copy) | `outpost_modules/g5m.py` | **delete** ✅ | Fully commented-out; live `g5m` comes from `picasso.g5m` via the boundary. |
| undrift_rsso (RSSO drift) | `outpost_modules/undrift_rsso.py` + the `undrift_rsso` workflow module | **delete** ✅ | Retired: never performant (maintainer decision). Removed the 6.5k-line module + the `undrift_rsso` method/helpers in `analyse.py`, the `confluence`/`util`/`modulespec`/`gui` entries, and its tests. Unrelated RSSO *channel-alignment* (`picasso_outpost.align_by_rsso`) is a different feature and stays. |
| ripleys_analysis (legacy Ripley) | `ripleys_analysis/` subpackage + the `ripleysk` / `ripleysk_average` workflow modules | **delete** ✅ | Retired: superseded by `ripleysk2` / `ripleysk_average2` (which use `spatial_stats/ripleys`). Removed the subpackage, the two methods + the commented `ripleysk_rafal`, and the `confluence`/`util`/`modulespec`/`gui` entries + tests. Shared helpers `_plot_ripleys_integrals` / `_find_ripleys_significant` stay (used by the `…2` pair). |

## Promote — genuinely novel, upstream candidates (ranked by readiness)

| # | feature → target module | source | status | ready | callers | blockers / notes |
|---|---|---|---|---|---|---|
| 1 | **CSR k-NN statistics** → `picasso.postprocess` | `outpost/spatial_stats/csr_knn.py` (moved, S2) | promote | **H** | analyse.py | Pure numpy/scipy, no picasso coupling. Analytic CSR model + density MLE picasso lacks. Now isolated in its own module (S2) → upstream-ready. Minor cleanup: recursion/renormalisation in `nndistribution_from_csr`. |
| 2 | **Point-set pairwise-shift kernel** → `picasso.imageprocess` | `outpost/alignment/shift_kernel.py` `_calculate_pairwise_shift` + `_find_peak_center_of_mass`, `_fit_2d_gaussian_peak`, `_calculate_adaptive_bins`, `_normalize_quality_metrics` (moved, S5) | promote | M | analyse.py (10+) | Now isolated in its own module. **Still has plot-saving mixed in** (`_calculate_pairwise_shift` calls the shift-histogram savers) — split the numeric core from plot-saving/filename params before upstreaming. Complements image-based `imageprocess.rcc`. |
| 3 | **Correspondence-free rigid registration** → `picasso.registration` | `outpost/registration/icp.py` `register_to_template`, `_rigid_fit` (Kabsch), `_match_within_gate` (Hungarian), `_icp_refine`, `_rotation_matrix` (moved, S7) | promote | **H** | origami (monolith) | Now isolated in its own module → upstream-ready. Pure numpy + `scipy.optimize.linear_sum_assignment`. Rotation-sweep + ICP template fit under unknown correspondence — complements `picasso.registration`’s matched-point affine. (The `subcluster_docking_sites`/candidate companions moved with it but are origami-coupled — the pure registration core is the upstream target.) |
| 5 | **Resolution: FRC + decorrelation** → `picasso.postprocess` | `outpost_modules/resolution_frc.py`, `resolution_decorrelation.py` | promote | **H** | analyse.py | Standard published methods, self-contained (decorrelation reuses `resolution_frc.render_image_histogram`). |
| 5b | **Resolution: PPAC** → `picasso.postprocess` | `outpost/resolution/resolution_ppac.py` (moved, S1) | ⚠️ deferred | L | analyse.py | Third resolution estimator, but **never performed well** — demoted from `promote` until rewritten (O(G²) KDTree-per-cell → displacement-vector histogram; see its module docstring / the S1 row). Not promote-ready as-is. |
| 6 | **Binding-event analysis** → picasso (new module) | `outpost/binding_events/{analysis,plotting}.py` (`Measurement`, `Plotting`) | promote | **H** | analyse.py | Cleanly paired analysis+plotting, zero picasso/sibling coupling. **Moved** into `outpost/binding_events/`; deps in the `[binding-events]` extra. |
| 7 | **CellMask morphology + density CSR** → extend `picasso.masking` | `outpost/masking/mask.py` `CellMask` (morphology: `filter_mask`, `component_masks`, `erode`/`dilate`; stats: `area`, `mean_density`; `random_points*`) | extend-up | M | analyse.py, ripleys | The cell-object layer `picasso.masking` lacks. **Moved** from `outpost_modules/` into `outpost/masking/`. Contribute the new part; dedup the overlapping thresholding (see masking/dedup below). |

## Dedup — redundant with picasso ≥ 0.12, rewire + delete the fork

| feature | source | upstream target | ready | callers | notes |
|---|---|---|---|---|---|
| `generate_N_structures` | `picasso_outpost.py` | `spinna.generate_N_structures` | ✅ done | analyse.py | **Deduped (batch 1).** Thin wrapper (silently dropped its `save` arg) removed. |
| `picked_locs` (+ `get_block_locs_at`, `locs_at`) | `outpost/picks/picks.py` (moved, S6) | `postprocess.picked_locs("Circle", pick_size=d/2)` | ✅ done | analyse.py | **Deduped (batch 2).** The forked `get_block_locs_at`/`locs_at` copies deleted. Membership + group identical; only intra-frame row order differs (see `tests/test_outpost_dedup_equivalence.py`). |
| `otsu()` threshold (×3) | `outpost/masking/mask.py`, `outpost/spatial_stats/ripleys.py`, `outpost/spatial_stats/ripley_dcatlas_analysis.py` | `picasso.masking.threshold_otsu` | ✅ done | analyse.py, ripleys | **Deduped.** The three byte-identical private `otsu()` copies were the seed for picasso's public `threshold_otsu` (same algorithm). Deleted all three; callers now use `threshold_otsu` via the `_picasso_api` boundary. Equivalence pinned (max abs diff 0 over random bimodal images, `test_otsu_threshold_matches_upstream`). NB: the upstream is `threshold_otsu` (a scalar threshold), **not** `mask_image(method="otsu")` — the callers scale the threshold (`otsu(x) * threshold`), they don't want a mask. |
| `pick_similar` | `outpost/picks/picks.py` (moved, S6) + `get_pick_similar_vals`, `pick_similar_analysis` | — | ❌ keep | analyse.py | NOT redundant: a deliberate *variant* returning `(picks, nlocs, rmsds, labels)` diagnostics, not just picks like upstream (its own comment says so). The workflow's phase-space analysis needs the extra outputs. |
| `pick_gold` | `outpost/picks/picks.py` (moved, S6) | — | ❌ keep | analyse.py | Workflow-specific gold-bead criteria (n_locs/RMSD thresholds); already calls upstream `postprocess._pick_similar`. (Boundary follow-up: that's a private reach-in → belongs behind a `_picasso_api` accessor.) |
| `_undrift_from_picked` | `outpost/picks/picks.py` (moved, S6) + `_undrift_from_picked_coordinate` | `postprocess.undrift_from_picked` (partial) | ⏳ maybe | analyse.py | Wrapper: computes drift via a custom per-coordinate routine, **applies** it, returns `(locs, info, drift)` — divergent interface from upstream (which returns just a drift DataFrame). Only the drift *computation* is a possible future dedup, gated by its own equivalence test. |
| spot extraction | `outpost/spots/extraction.py` `get_spots`, `normalize_spot`, `_cut_spots*` (moved, S3b) | — | ❌ keep | analyse.py | NOT a dedup: `_cut_spots_byrandomframe` is deliberate random-frame *sampling*, divergent from `localize.get_spots`. |
| global shift solve | `outpost/alignment/channels.py` `_solve_optimal_shifts` (moved, S5) | `picasso.lib.minimize_shifts` | ⏳ maybe | internal | Used internally (not dead); overlaps upstream but needs an equivalence check before rewiring. |

**Dedup reality (verified batches 1–3):** of the functions the triage listed, only `generate_N_structures` and `picked_locs` were *genuine* redundancies — both done. The rest of the so-called "pick/undrift fork" are **divergent workflow variants** that *use* upstream internals but return workflow-specific structures (diagnostics, applied drift, sampled spots), so they stay. The clean dedup wins are complete; remaining items (`_undrift_from_picked` drift computation, `_solve_optimal_shifts`) are uncertain and individually gated. Net leverage now shifts to **promotion** (upstreaming the novel primitives) and the **boundary follow-up** (route `pick_gold`'s private `postprocess._pick_similar` reach-in through `_picasso_api`).
| masking primitives | inline render/blur, `apply_to_locs` binning (`outpost/masking/mask.py`) | `picasso.masking.{generate_image,mask_image(method="otsu"),mask_locs}` | M | analyse.py | The private `otsu()/3` thresholding is **done** (see the `otsu()` row above — rewired to `threshold_otsu`). Remaining: consolidate the inline image/blur/binning masking onto one `CellMask`. |
| dbscan_molint mask precursor | `dbscan_molint/mask.py` | `CellMask` / `picasso.masking` | M | analyse.py | 2022 script-style precursor with known bugs + "Luciano's code" file coupling. Fold `CSR_sim_in_mask*` onto `CellMask`, delete the rest. |

## Keep — stays workflow-side

| feature | source | why |
|---|---|---|
| channel alignment orchestration | `outpost/alignment/channels.py` `align_channels`, `align_by_rcc`, `plot_shift` (moved, S5) | Workflow multi-channel wrapper (now over upstream `align_from_picked`/`rcc`). |
| spinna wrappers | `outpost/spinna/wrappers.py` (moved, S3) | Thin wrappers over `picasso.spinna` + workflow plumbing/plots. |
| dbscan-molint analysis | `outpost/clustering/molint.py` `DBSCAN_analysis*`, `degree_of_clustering`, barcode/interaction plots (moved, S4); `dbscan_molint/` subpackage (fold = S4b) | Coupled to the molint report/Confluence format. |
| origami orchestration + descriptors | `outpost/origami/origami.py` `pick_origami`, `OrigamiTemplate` builders, `structure_pattern_features`, `lattice_defect*` (moved, S8) | Workflow-shaped; geometry primitives may promote later after #3 lands. |
| kinetics simulation | `outpost/origami/origami.py` `simulate_origami_nlocs_rmsd`, `origami_phase_space_preview` (moved, S8) | WIP/experimental. |
| Zeiss import | `outpost/formats/zeiss.py` `convert_zeiss_file` (moved, S3c) | Niche .czi conversion glue (picasso `io.load_czi`; `[formats]` extra). |
| render glue | `outpost/render/render.py` | matplotlib wrapper over `picasso.render`. **Moved** from `outpost_modules/` into `outpost/render/`. |
| Ripley's K core | `outpost/spatial_stats/ripleys.py` | Ripley's K / RDF / FRC + CSR/randomized controls + plotting (Rafal). Report-coupled (`confluence.ripleysk2`). **Moved** from `outpost_modules/` into `outpost/spatial_stats/`. |
| Ripley DC-Atlas redo | `outpost/spatial_stats/ripley_dcatlas_analysis.py` | Explicit reproducible redo of a published pipeline (built on `ripleys`). **Moved** into `outpost/spatial_stats/`. |
| report plot helpers | `outpost/reporting/histograms.py` `plot_1dhist`, `plot_2dhist` (moved, S9) | Reporting. Last code out of the monolith; `picasso_outpost.py` then deleted. |

## Suggested PR sequence

1. ✅ Delete dead code + this scaffold (#38).
2. ✅ **Dedup batches 1–2** (#39): `generate_N_structures`, `picked_locs`. Equivalence-tested.
3. ✅ **Dedup batch 3** (investigation): confirmed the rest of the pick/undrift fork is NOT redundant (divergent variants — `pick_similar`, `pick_gold`, `_undrift_from_picked` stay); MANIFEST corrected. Clean dedup wins complete.
4. ✅ **Boundary follow-up** (#41): `pick_gold`'s private `postprocess._pick_similar` reach-in — and three others (`_n_block_locs_at`, `localize._to_photons`, `spinna._targets_from_structures`) — now go through `_picasso_api.get_internal`, registered in `INTERNAL_SYMBOLS` and checked by the contract test. (Durable fix = upstream a public API for each — see the Triage integration-health table.)
5. **Restructure** (incremental, one feature per PR): fold `outpost_modules/`
   into `outpost/<feature>/` and physically split `picasso_outpost.py`, flipping
   rows as each feature lands. Done so far: `resolution/` (#42),
   `binding_events/` (#45), `spatial_stats/` (#46, ripleys + DC-Atlas),
   `masking/` (#47, CellMask), `render/` (#49); `undrift_rsso` and the legacy
   `ripleys_analysis/` subpackage both retired (deleted). `outpost_modules/` now
   holds no feature modules; the `picasso_outpost.py` monolith split is
   **complete (S1–S9)** and the file has been deleted.
6. **Promote** (readiness H first): CSR k-NN stats, correspondence-free ICP registration, binding-events; then point-set shift kernel, resolution (FRC/decorrelation/PPAC), CellMask morphology.

## Monolith split — sequenced plan (`picasso_outpost.py` → `outpost/<feature>/`)

**✅ COMPLETE (S1–S9).** `picasso_outpost.py` (was **7579 lines**) has been fully
split into `outpost/<feature>/` modules and **deleted**; the last two report
helpers landed in `outpost/reporting/histograms.py` (S9). The plan below is kept
as the record of how the split was sequenced. Originally used
at **48 sites in `analyse.py`** and heavily in `tests/test_picasso_outpost.py`.
Unlike the AutoPicasso-mocked unit tests, `test_picasso_outpost.py` calls these
functions **directly**, so a repointed test gives *real* coverage of each move —
the safety net the earlier outpost slices lacked. Each PR extracts one cohesive
cluster into `outpost/<feature>/`, updates its `analyse.py` import sites +
internal cross-refs + `test_picasso_outpost.py` references, with the same
contract/test care as the earlier slices. **Promote-candidates get their own
module** so they can later be upstreamed cleanly (the triage's point). Pure
moves, no behaviour change.

This **refines the README target layout**: `drift/` is obsolete (undrift_rsso
deleted) — the shared shift kernel moves to `alignment/`; and ICP registration
gets its own `registration/` rather than living under `origami/`.

| # | cluster (lines in current file) | → target | dest | notes / deps |
|---|---|---|---|---|
| S1 ✅ | **PPAC resolution** (`resolution_ppac*`, `analyse_resolution_ppac`) | `outpost/resolution/resolution_ppac.py` | keep ⚠️ needs-work | **Done** (−512 lines from the monolith). **Flagged not promote-ready — never performed well.** Root cause: rebuilds a `KDTree(xy_shift)` for *every* grid cell → O(G²·N log N); sparse/chunked/parallel variants are band-aids. **Fix before promoting** (written up in the module docstring): the G×G map is a 2D histogram of pairwise displacement vectors — build one `KDTree`, `query_pairs` within `r_max` once, `np.histogram2d` the displacements; then drop dead `batch_size`/`use_chunking` + redundant backends, add an equivalence test, revisit promotion. |
| S2 ✅ | **CSR k-NN stats** (`nndistribution_from_csr`, `csr_cdf_for_ks_test`, `nndist_loglikelihood_csr`, `estimate_density_from_neighbordists`, `minimization_loglike`, `_check_distance_window`) | `outpost/spatial_stats/csr_knn.py` | promote | **Done** (−383 lines). Self-contained (numpy/scipy + loguru). Repointed `analyse.fit_csr` (4 calls, deferred import) + `test_04*`; removed 3 now-orphaned scipy imports (`gamma`/`factorial`/`minimize`) from the monolith. Still a genuine promote candidate. |
| S3 ✅ | **spinna wrappers** (`spinna_batch`, `screen_label_uncertainty`, `single_spinna_run`, `single_spinna_fit_le_run`, `plot_spinna_nnd`, `_plot_label_unc_scan`, `load_structures_from_dict`) | `outpost/spinna/wrappers.py` | keep | **Done** (−706 lines). Carried `spinna`/`get_internal`/`get_spinna_batch_analysis` + `os`/`glob`/`datetime` imports; removed `spinna`/`get_spinna_batch_analysis`/`glob`/`datetime` as now-orphaned from the monolith. 9 analyse call sites (top-level alias import) + ~12 `test_analyse` `@patch`/direct refs repointed to `outpost.spinna.wrappers`. |
| S3b ✅ | **spot extraction** (`get_spots`, `_cut_spots`, `_cut_spots_byrandomframe`, `normalize_spot`) | `outpost/spots/extraction.py` | keep | **Done** (−85 lines). Needs `np` + `get_internal`. analyse `_draw_sample_spots` (2 calls, deferred import). No orphaned monolith imports. |
| S3c ✅ | **Zeiss import** (`convert_zeiss_file`) | `outpost/formats/zeiss.py` | keep | **Done** (batched with S3b). Uses picasso's native `io.load_czi`/`io.save_raw` (needs `[formats]` extra). analyse `convert_zeiss_movie` + `test_analyse` `@patch` repointed. |
| S4 ✅ | **dbscan-molint** (`DBSCAN_analysis*`, `_do_dbscan_molint`, `degree_of_clustering`, barcode/interaction plots) | `outpost/clustering/molint.py` | keep | **Done** (−996 lines). Needs np/pd/plt/mlines/stats/os/yaml/util/io; **removed `mlines`/`io`/`util`/`stats` as now-orphaned** from the monolith (sklearn `DBSCAN` stays — origami uses it). No test callers. The `dbscan_molint/` subpackage is used via deferred imports and stays put — **folding it into `clustering/` is a separate step (S4b).** |
| S5 ✅ | **channel alignment + shift kernel** (`align_channels`/`align_by_*`/`plot_shift`/`shift_from_rcc` + `_calculate_pairwise_shift` & peak-fit helpers + shift-histogram plots + `_solve_optimal_shifts`) | `outpost/alignment/` (**`shift_kernel.py`** + **`channels.py`**) | keep + promote | **Done** (−1841 lines, biggest slice). Two-module split: `shift_kernel.py` = the estimation machinery (promote #2), `channels.py` = orchestration + `_solve_optimal_shifts`; one-way `channels`→`shift_kernel`. **Could NOT cleanly isolate the pure numeric core** (the kernel calls its plot savers → circular), so `shift_kernel.py` keeps the plots too — the plots-stripping refactor (row 2) precedes upstreaming. channels temporarily imports `shift_from_picked`/`sort_picked_locs` from the monolith (→ picks, S6). Removed `plt`/`os`/`render`/`imageprocess` as orphaned from the monolith. |
| S6 ✅ | **picks / fiducials** (`pick_gold`, `pick_similar*`, `picked_locs`, `index_locs`, `_undrift_from_picked*`, `shift_from_picked`, `sort_picked_locs`, `find_structures`, block/locs helpers) | `outpost/picks/picks.py` | keep | **Done** (−816 lines). Carried the numba `_n_block_locs_at = get_internal(...)` **module-global dispatcher** (njit can't call get_internal). Removed `pd`/`nb`/`postprocess`/`get_internal`/`OPTICS` as orphaned from the monolith (all 4 get_internal reach-ins now left it). Resolved S5's channels→monolith forward-import. **Origami (still in monolith, S8) calls `pick_similar`/`get_pick_similar_vals`/`picked_locs` → monolith now forward-imports them from picks** (one-way, no cycle). ~13 analyse + ~14 test refs repointed; numba-compile regression test (`test_outpost_dedup_equivalence`) green. |
| S7 ✅ | **ICP registration** (`register_to_template`, `_rigid_fit` Kabsch, `_match_within_gate` Hungarian, `_icp_refine`, `_rotation_matrix`, `subcluster_docking_sites`, `classify/accept_candidate`, `_candidate_centers_cluster_of_clusters`, `_elongation`/`_radius_of_gyration`, `_median_nn_spacing`) | `outpost/registration/icp.py` | promote | **Done** (−408 lines), **separated from origami** → upstream-ready. Pulled the shared `_median_nn_spacing` geometry helper in (register_to_template needs it); imports `get_pick_similar_vals` from picks (one-way). Removed `linear_sum_assignment`/`DBSCAN`/`get_pick_similar_vals` orphans from the monolith. Origami (monolith, S8) forward-imports the 9 names it uses. No analyse coupling; `test_picasso_outpost` (~10 sites) repointed. |
| S8 ✅ | **origami** (`OrigamiTemplate` + builders, kinetics sim, structure/lattice descriptors, `pick_origami`) | `outpost/origami/origami.py` | keep | **Done** (−1686 lines, biggest slice — drains the monolith to 56 lines). Moved as ONE module. Imports `pick_similar`/`picked_locs` from picks + the 9 registration names from `registration/icp` directly → **both monolith forward-imports (S6/S7) removed**. Removed the now-orphaned sklearn/scipy/yaml/dataclass/Counter imports from the monolith. analyse (6) + **gui (origami_phase_space_preview)** + test_analyse `@patch` + test_picasso_outpost (many template tests) repointed. |
| S9 ✅ | **report plot helpers** (`plot_1dhist`, `plot_2dhist`) + leftovers | `outpost/reporting/histograms.py` | keep | **Done** — the final slice. Needs `np` + `LogNorm` + `lib` only. analyse (2 calls) repointed via `reporting_hist` alias; stale `picasso_outpost` docstring refs in `util`/`gui`/`modulespec`/`_picasso_api` repointed to the new homes. **`picasso_outpost.py` deleted** — the monolith split is complete. |

**Order rationale:** S1–S2 are clean, self-contained extractions into folders
that already exist (lowest risk, first) — S2 is a genuine *promote*; S1 (PPAC)
is the same easy move but lands flagged needs-work, not promote-ready (see its
row). S3–S6 are cohesive *keep* clusters (S3/S3b/S3c are three *distinct*
features, each its own folder — not bundled). S7 (ICP *promote*) is extracted
**before** S8 so origami imports a clean `registration` module. S8 is the
largest/most tangled — last. S9 empties and removes the file. Each step is one
PR (the tiny S3c may batch with a neighbour; S8 may split), merged sequentially
with the same union-rebase dance as #45–#51.
