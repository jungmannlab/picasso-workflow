# Cluster Apptainer image (register C10 / WP-9)

`paint.def` is the cluster counterpart of the dev-container `Dockerfile`: the
same foundation (Python 3.11 + Debian bookworm + headless-Qt libs) with the
picasso 1.0 analysis stack baked in (`picassosr` 1.0 + `picasso-workflow` +
`picasso-registry[client]`), so the undrift-first aggregation (WP-9 Phase C)
runs inside the image exactly as it does in dev. A single `.sif` also sidesteps
the inode blow-up a conda env causes on an HPC home quota.

## Build it

- **In CI:** [`.github/workflows/build-apptainer.yml`](../../.github/workflows/build-apptainer.yml)
  builds on a hosted runner (`workflow_dispatch` with a `pw_ref`, or on each
  published release) and publishes the `.sif` as a CI artifact / release asset.
- **By hand** (login node or any box with Apptainer + fakeroot):
  ```bash
  export APPTAINER_TMPDIR=$SCRATCH/apptainer-tmp     # keep scratch off home quota
  export APPTAINER_CACHEDIR=$SCRATCH/apptainer-cache
  module load apptainer/1.4.1                        # hpcl8 floor
  apptainer build --fakeroot picasso-workflow.sif deploy/apptainer/paint.def
  apptainer test picasso-workflow.sif               # imports the stack headlessly
  ```

### Bump the pins in `paint.def` (`%post`) before a build

- **`PW_REF`** — picasso-workflow git ref. Tracks `develop` (the picasso-1.0
  migration, PR #77); switch to a release tag once one is cut. CI overrides this
  per-build (the release tag / the dispatch input).
- **`REGISTRY_REF`** — keep matched to the deployed registry service's tag
  (C48 version-matching). Currently `v0.7.0`.

## Naming & deploy (C48)

Published images are **`picasso-workflow-<version>.sif`** — the filename carries
the *primary consumer's* version (`picasso_workflow.__version__`), which is the
key a run matches on (the submitter resolves its installed `__version__` →
`image_path`). Rolling CI builds are `picasso-workflow-develop-<sha>.sif` (+ a
`picasso-workflow-develop.sif` symlink maintained at pull time). The bundled
`picassosr` / registry-client versions are **not** inferable from the filename —
read them from `/opt/paint-image-freeze.txt` + `apptainer inspect` labels.

CI builds the `.sif` and pushes it to **ghcr via ORAS** (the C48-recommended
target); it does **not** auto-appear on the cluster. Pull it on the **login
node** (compute nodes have no internet) to the shared path WP-9's submit template
expects, then jobs `apptainer exec` from there:

```bash
module load apptainer/1.4.1
# rolling develop builds are tagged :develop and :develop-<sha>; releases :<version>
apptainer pull /sw/picasso-workflow-<version>.sif \
    oras://ghcr.io/jungmannlab/picasso-workflow:<version>
# (release builds are also attached as a GitHub release asset: curl -L -o … works too)
# then, matching image_path= in ClusterSubmitConfig / the Gate-5 runbook:
apptainer exec /sw/picasso-workflow-<version>.sif python -m picasso_workflow …
```

The ghcr package starts **private** — make it public (or give the login node a
read token) so pulls work.

The full reference recipe (`build.sh` + extended notes) lives in the planning
repo at `dev-environment/apptainer/`.
