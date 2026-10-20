# picasso-workflow outpost — Phase 2 migration

"Outpost" is picasso-workflow's picasso-**adjacent** code: DNA-PAINT analysis
helpers that are not (yet) in the `picassosr` library. Historically it lived in
two places with no feature cohesion:

- `picasso_workflow/picasso_outpost.py` — a ~7,600-line monolith, ~110 top-level
  defs spanning 7 unrelated feature areas.
- `picasso_workflow/outpost_modules/` — a flat grab-bag of single-file modules
  from different authors/years.

Phase 2 reorganises this into **feature-focused submodules** under
`picasso_workflow/outpost/`, and — crucially — resolves each feature against
upstream picasso, which (as of **0.12**) has absorbed a large slice of what the
outpost once owned. `MANIFEST.md` is the per-feature tracker.

## Why a manifest

The migration is large, so we make it a **tracked checklist** instead of one
monumental refactor. Each feature has a status; a PR flips exactly one or two
rows and links the upstream picasso PR (if any). `grep 'status: promote'
MANIFEST.md` shows what is left to upstream; `grep 'status: dedup'` shows the
redundancy still to remove.

## Status vocabulary

| status | meaning |
|---|---|
| `keep` | Stays workflow-side permanently (report/Confluence glue, orchestration, workflow-specific policy). |
| `dedup` | Redundant with picasso ≥ 0.12. Rewire callers to the upstream API, then **delete the local fork**. Higher leverage than promotion. |
| `promote` | Genuinely novel and reusable. Propose upstreaming into picasso as a new function/module. |
| `extend-up` | Extends an existing picasso module (e.g. `CellMask` → `picasso.masking`); contribute the new part, dedup the overlapping part. |
| `delete` | Dead code. Remove outright. |

## Target layout

```
picasso_workflow/outpost/
├── MANIFEST.md              # per-feature migration tracker (source of truth)
├── README.md                # this file
├── resolution/              # FRC, image-decorrelation, point-pattern autocorr (PPAC)
├── drift/                   # RSSO drift + the shared point-set pairwise-shift kernel
├── masking/                 # CellMask morphology + density CSR (aligned to picasso.masking)
├── spatial_stats/           # analytic CSR k-NN stats + Ripley's K / CSR envelopes
├── picks/                   # pick/fiducial utilities (mostly dedup to picasso.postprocess)
├── origami/                 # DNA-origami template build + ICP registration + lattice defects
├── clustering/              # DBSCAN molecular-interaction (molint) analysis + plots
├── spinna/                  # thin wrappers over picasso.spinna + Zeiss import
├── binding_events/          # binding-event analysis + plotting (self-contained)
└── render/                  # matplotlib glue over picasso.render (workflow-side)
```

## Process for each feature

1. **Dedup first.** If picasso ≥ 0.12 already provides it, rewire `analyse.py`
   (and `picasso_set/*`) to the upstream call and delete the fork. Before
   deleting a fork that `analyse.py` calls, add an equivalence check (run fork
   vs upstream on the synthetic fixtures in `tests/conftest.py` and assert the
   same result) so a subtle behavioural difference can't slip through — the unit
   tier mocks `AutoPicasso`, so it will not catch this on its own.
2. **Then promote** the genuinely-novel remainder as an upstream picasso PR;
   once merged and released, replace the local copy with the upstream import and
   flip the row to `dedup`/`keep`.
3. **Update the MANIFEST row** in the same PR.
