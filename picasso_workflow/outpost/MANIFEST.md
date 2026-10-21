# Outpost migration manifest

Per-feature status for the Phase 2 reorganisation of picasso-workflow's
picasso-adjacent code. Source of truth for what moves where. See `README.md`
for the `keep / dedup / promote / extend-up / delete` vocabulary.

Readiness/effort: **H** = ready/clean, **M** = needs a refactor first, **L** =
WIP / not ready.

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
| 1 | **CSR k-NN statistics** → `picasso.postprocess` | `picasso_outpost.py` `nndistribution_from_csr`, `csr_cdf_for_ks_test`, `nndist_loglikelihood_csr`, `estimate_density_from_neighbordists` (+ `minimization_loglike`, `_check_distance_window`) (2765–3147) | promote | **H** | analyse.py | Pure numpy/scipy, no picasso coupling. Analytic CSR model + density MLE picasso lacks. Minor cleanup: recursion/renormalisation in `nndistribution_from_csr`. |
| 2 | **Point-set pairwise-shift kernel** → `picasso.imageprocess` | `picasso_outpost.py` `_calculate_pairwise_shift` + `_find_peak_center_of_mass`, `_fit_2d_gaussian_peak`, `_calculate_adaptive_bins`, `_normalize_quality_metrics` (518–1361) | promote | M | analyse.py (10+) | Split numeric core from plot-saving/filename params. Complements image-based `imageprocess.rcc`. |
| 3 | **Correspondence-free rigid registration** → `picasso.registration` | `picasso_outpost.py` `register_to_template`, `_rigid_fit` (Kabsch), `_match_within_gate` (Hungarian), `_icp_refine`, `_rotation_matrix`; companion `subcluster_docking_sites` (5542–5800) | promote | **H** | analyse.py (origami) | Small, self-contained (numpy + `scipy.optimize.linear_sum_assignment`). Rotation-sweep + ICP template fit under unknown correspondence — complements `picasso.registration`’s matched-point affine. |
| 5 | **Resolution: FRC + decorrelation** → `picasso.postprocess` | `outpost_modules/resolution_frc.py`, `resolution_decorrelation.py` | promote | **H** | analyse.py | Standard published methods, self-contained (decorrelation reuses `resolution_frc.render_image_histogram`). |
| 5b | **Resolution: PPAC** → `picasso.postprocess` | `picasso_outpost.py` `resolution_ppac*`, `analyse_resolution_ppac` (7158–7662) | promote | M | analyse.py | Third, independent resolution estimator. Prune dead params (`batch_size`, `use_chunking`) + collapse to one backend first. |
| 6 | **Binding-event analysis** → picasso (new module) | `outpost/binding_events/{analysis,plotting}.py` (`Measurement`, `Plotting`) | promote | **H** | analyse.py | Cleanly paired analysis+plotting, zero picasso/sibling coupling. **Moved** into `outpost/binding_events/`; deps in the `[binding-events]` extra. |
| 7 | **CellMask morphology + density CSR** → extend `picasso.masking` | `outpost/masking/mask.py` `CellMask` (morphology: `filter_mask`, `component_masks`, `erode`/`dilate`; stats: `area`, `mean_density`; `random_points*`) | extend-up | M | analyse.py, ripleys | The cell-object layer `picasso.masking` lacks. **Moved** from `outpost_modules/` into `outpost/masking/`. Contribute the new part; dedup the overlapping thresholding (see masking/dedup below). |

## Dedup — redundant with picasso ≥ 0.12, rewire + delete the fork

| feature | source | upstream target | ready | callers | notes |
|---|---|---|---|---|---|
| `generate_N_structures` | `picasso_outpost.py` | `spinna.generate_N_structures` | ✅ done | analyse.py | **Deduped (batch 1).** Thin wrapper (silently dropped its `save` arg) removed. |
| `picked_locs` (+ `get_block_locs_at`, `locs_at`) | `picasso_outpost.py` | `postprocess.picked_locs("Circle", pick_size=d/2)` | ✅ done | analyse.py | **Deduped (batch 2).** The forked `get_block_locs_at`/`locs_at` copies deleted. Membership + group identical; only intra-frame row order differs (see `tests/test_outpost_dedup_equivalence.py`). |
| `pick_similar` | `picasso_outpost.py` (≈4594) + `get_pick_similar_vals`, `pick_similar_analysis` | — | ❌ keep | analyse.py | NOT redundant: a deliberate *variant* returning `(picks, nlocs, rmsds, labels)` diagnostics, not just picks like upstream (its own comment says so). The workflow's phase-space analysis needs the extra outputs. |
| `pick_gold` | `picasso_outpost.py` (≈4164) | — | ❌ keep | analyse.py | Workflow-specific gold-bead criteria (n_locs/RMSD thresholds); already calls upstream `postprocess._pick_similar`. (Boundary follow-up: that's a private reach-in → belongs behind a `_picasso_api` accessor.) |
| `_undrift_from_picked` | `picasso_outpost.py` (≈4370) + `_undrift_from_picked_coordinate` | `postprocess.undrift_from_picked` (partial) | ⏳ maybe | analyse.py | Wrapper: computes drift via a custom per-coordinate routine, **applies** it, returns `(locs, info, drift)` — divergent interface from upstream (which returns just a drift DataFrame). Only the drift *computation* is a possible future dedup, gated by its own equivalence test. |
| spot extraction | `picasso_outpost.py` `get_spots`, `normalize_spot`, `_cut_spots*` | — | ❌ keep | analyse.py | NOT a dedup: `_cut_spots_byrandomframe` is deliberate random-frame *sampling*, divergent from `localize.get_spots`. |
| global shift solve | `picasso_outpost.py` `_solve_optimal_shifts` | `picasso.lib.minimize_shifts` | ⏳ maybe | internal | Used internally (not dead); overlaps upstream but needs an equivalence check before rewiring. |

**Dedup reality (verified batches 1–3):** of the functions the triage listed, only `generate_N_structures` and `picked_locs` were *genuine* redundancies — both done. The rest of the so-called "pick/undrift fork" are **divergent workflow variants** that *use* upstream internals but return workflow-specific structures (diagnostics, applied drift, sampled spots), so they stay. The clean dedup wins are complete; remaining items (`_undrift_from_picked` drift computation, `_solve_optimal_shifts`) are uncertain and individually gated. Net leverage now shifts to **promotion** (upstreaming the novel primitives) and the **boundary follow-up** (route `pick_gold`'s private `postprocess._pick_similar` reach-in through `_picasso_api`).
| masking primitives | local `otsu()` (`outpost/masking/mask.py`; also in the Ripley's-K module, `outpost_modules/ripleys.py` → `spatial_stats/ripleys.py` after #46), inline render/blur, `apply_to_locs` binning | `picasso.masking.{generate_image,mask_image(method="otsu"),mask_locs}` | M | analyse.py | Preserve the PW `otsu()/3` threshold **as a parameter**, not a reason to keep a private Otsu. Consolidate the 3 masking copies onto one `CellMask`. |
| dbscan_molint mask precursor | `dbscan_molint/mask.py` | `CellMask` / `picasso.masking` | M | analyse.py | 2022 script-style precursor with known bugs + "Luciano's code" file coupling. Fold `CSR_sim_in_mask*` onto `CellMask`, delete the rest. |

## Keep — stays workflow-side

| feature | source | why |
|---|---|---|
| channel alignment orchestration | `picasso_outpost.py` `align_channels`, `align_by_rcc`, `plot_shift` | Workflow multi-channel wrapper (now over upstream `align_from_picked`/`rcc`). |
| spinna wrappers | `picasso_outpost.py` `spinna_batch`, `single_spinna_run`, `screen_label_uncertainty`, plotting (2027–2695) | Thin wrappers over `picasso.spinna` + workflow plumbing/plots. |
| dbscan-molint analysis | `picasso_outpost.py` `DBSCAN_analysis*`, `degree_of_clustering`, barcode/interaction plots (3148–4016); `dbscan_molint/` | Coupled to the molint report/Confluence format. |
| origami orchestration + descriptors | `picasso_outpost.py` `pick_origami`, `OrigamiTemplate` builders, `structure_pattern_features`, `lattice_defect*` (5024–6648) | Workflow-shaped; geometry primitives may promote later after #3 lands. |
| kinetics simulation | `picasso_outpost.py` `simulate_origami_nlocs_rmsd`, `origami_phase_space_preview` (5265–5445) | WIP/experimental. |
| Zeiss import | `picasso_outpost.py` `convert_zeiss_file` (1901) | Niche .czi conversion glue (`aicsimageio`). |
| render glue | `outpost/render/render.py` | matplotlib wrapper over `picasso.render`. **Moved** from `outpost_modules/` into `outpost/render/`. |
| Ripley's K core | `outpost/spatial_stats/ripleys.py` | Ripley's K / RDF / FRC + CSR/randomized controls + plotting (Rafal). Report-coupled (`confluence.ripleysk2`). **Moved** from `outpost_modules/` into `outpost/spatial_stats/`. |
| Ripley DC-Atlas redo | `outpost/spatial_stats/ripley_dcatlas_analysis.py` | Explicit reproducible redo of a published pipeline (built on `ripleys`). **Moved** into `outpost/spatial_stats/`. |
| report plot helpers | `picasso_outpost.py` `plot_1dhist`, `plot_2dhist`, `_save_*_histogram_plot` | Reporting. |

## Suggested PR sequence

1. ✅ Delete dead code + this scaffold (#38).
2. ✅ **Dedup batches 1–2** (#39): `generate_N_structures`, `picked_locs`. Equivalence-tested.
3. ✅ **Dedup batch 3** (investigation): confirmed the rest of the pick/undrift fork is NOT redundant (divergent variants — `pick_similar`, `pick_gold`, `_undrift_from_picked` stay); MANIFEST corrected. Clean dedup wins complete.
4. **Boundary follow-up**: route `pick_gold`'s private `postprocess._pick_similar` reach-in through a `_picasso_api` accessor (like `_FIT_METHOD_MAP`).
5. **Restructure** (incremental, one feature per PR): fold `outpost_modules/`
   into `outpost/<feature>/` and physically split `picasso_outpost.py`, flipping
   rows as each feature lands. Done so far: `resolution/` (#42),
   `binding_events/` (#45), `spatial_stats/` (#46, ripleys + DC-Atlas),
   `masking/` (#47, CellMask), `render/` (#49); `undrift_rsso` and the legacy
   `ripleys_analysis/` subpackage both retired (deleted). `outpost_modules/` now
   holds no feature modules; only the `picasso_outpost.py` monolith remains to
   split.
6. **Promote** (readiness H first): CSR k-NN stats, correspondence-free ICP registration, binding-events; then point-set shift kernel, resolution (FRC/decorrelation/PPAC), CellMask morphology.
