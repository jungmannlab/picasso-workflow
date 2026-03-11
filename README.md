# picasso-workflow

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

master:
![master unit tests](https://img.shields.io/github/actions/workflow/status/jungmannlab/picasso-workflow/run-unittests.yml?branch=master&label=unit%20tests)
![master cluster tests](https://img.shields.io/github/actions/workflow/status/jungmannlab/picasso-workflow/run-cluster-tests.yml?branch=master&label=cluster%20tests)

develop:
![develop unit tests](https://img.shields.io/github/actions/workflow/status/jungmannlab/picasso-workflow/run-unittests.yml?branch=develop&label=unit%20tests)
![develop cluster tests](https://img.shields.io/github/actions/workflow/status/jungmannlab/picasso-workflow/run-cluster-tests.yml?branch=develop&label=cluster%20tests)
![Coveralls develop](https://img.shields.io/coverallsCoverage/github/jungmannlab/picasso-workflow?branch=develop)


A package for automated DNA-PAINT analysis workflows

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [CI / GitHub Actions](#ci--github-actions)
- [Releasing](#releasing)
- [Contributing](#contributing)
- [License](#license)

## Features

- The project aims at automating DNA-PAINT workflows, especially the analysis
via picassosr.
- There are two main types of workflow:
	- Single-dataset workflow: a single dataset is e.g. loaded, localized,
	and clustered.
	- Aggregation workflow: multiple datasets undergo a single-dataset
	workflow and are then aggregated.

## Installation

- create a new anaconda environment: `conda create -n picasso-workflow python=3.10`
- If you want to use a local development version of picasso, install that first:
	- `cd /path/to/picasso`
	- `pip install -r requirements.txt`
- Dependencies are specified in requirements.txt, install by:
	- `cd /path/to/picasso-workflow`
	- `pip install -e .`
- Should be platform independent. Tested on MacOS Sonoma and  Windows Server.

## Usage

- see examples in the folder "examples".

### One-click installers

Three installer scripts handle the full setup (find conda → create
environment → `pip install` → create shortcut/app bundle) in a single
double-click.

| Script | Platform | Who runs it |
|---|---|---|
| `tools/install_windows_personal.bat` | Windows | Any user — creates shortcut on your own desktop for testing |
| `tools/install_windows_allusers.bat` | Windows | Administrator — creates shortcut on every user's desktop |
| `tools/install_mac.command` | macOS | Any user — creates `~/Applications/picasso-workflow.app` |

**Windows**: double-click the `.bat` file.  The all-users variant
automatically requests elevation (UAC prompt).

**macOS**: double-click `install_mac.command` in Finder.  On first run,
macOS may block it — go to *System Settings → Privacy & Security* and
click *Open Anyway*, then double-click again.

After installation the GUI can also be launched from the terminal:

```bash
# terminal (any platform, environment activated):
picasso-workflow-gui

# or:
python -m picasso_workflow.gui
```

### Windows Server deployment — per-user desktop shortcut

On a shared Windows Server, placing a shortcut in
`C:\Users\Public\Desktop` makes it appear on **every** user's desktop
without GPO or per-user scripting.  The helper script
`tools\deploy_gui_shortcut.ps1` does this automatically.

**Prerequisites**

1. Install the package in the shared conda environment (once, by an
   administrator):
   ```powershell
   conda activate picasso-workflow
   pip install -e C:\path\to\picasso-workflow
   ```
   `pip install` reads the `[project.gui-scripts]` entry point in
   `pyproject.toml` and creates
   `<conda-env>\Scripts\picasso-workflow-gui.exe` — a native Windows
   executable that launches the GUI without a console window.

2. Verify it works interactively:
   ```powershell
   conda activate picasso-workflow
   picasso-workflow-gui
   ```

**Step 1 — Test as a normal user (no admin needed)**

Run without `-AllUsers` to create a shortcut on your own desktop only.
This lets you verify the install before involving an administrator:

```powershell
conda activate picasso-workflow
powershell -ExecutionPolicy Bypass -File tools\deploy_gui_shortcut.ps1
```

Double-click the shortcut that appears on your desktop.  If the GUI
opens correctly, the install is working.

**Step 2 — Deploy to all users (Administrator required)**

Once verified, ask an administrator to run the same script with
`-AllUsers` from an elevated prompt:

```powershell
# Option A — environment is already activated:
conda activate picasso-workflow
powershell -ExecutionPolicy Bypass -File tools\deploy_gui_shortcut.ps1 -AllUsers

# Option B — specify the environment path explicitly:
powershell -ExecutionPolicy Bypass -File tools\deploy_gui_shortcut.ps1 `
    -CondaEnvPath "C:\ProgramData\Anaconda3\envs\picasso-workflow" -AllUsers
```

This writes `C:\Users\Public\Desktop\picasso-workflow.lnk`, which
appears on every user's desktop.  Re-run after upgrading the package or
moving the conda environment.

**What the script does**

| Step | Action |
|---|---|
| 1 | Resolves the conda environment path (`$CONDA_PREFIX` or `-CondaEnvPath`) |
| 2 | Locates `Scripts\picasso-workflow-gui.exe` inside that environment |
| 3 | Without `-AllUsers`: creates shortcut on your personal desktop |
| 3 | With `-AllUsers`: creates shortcut in `C:\Users\Public\Desktop` |

No registry edits, no GPO, no per-user configuration needed.

### macOS deployment — single-user app bundle

On macOS the standard way to make a Python GUI launchable from Finder (or
pinnable to the Dock) is a `.app` bundle.  The helper script
`tools/deploy_gui_mac.sh` builds one and places it in `~/Applications/`.

**Prerequisites** — same as Windows: install the package in the conda
environment first:

```bash
conda activate picasso-workflow
pip install -e /path/to/picasso-workflow
picasso-workflow-gui   # verify it launches from the terminal
```

**Creating the app bundle (no sudo required)**

```bash
# With the environment already activated:
conda activate picasso-workflow
bash tools/deploy_gui_mac.sh

# Or with an explicit environment path:
CONDA_ENV_PATH=~/miniconda3/envs/picasso-workflow \
    bash tools/deploy_gui_mac.sh
```

The script creates `~/Applications/picasso-workflow.app`.  To make it
easily accessible:

- **Dock**: drag `~/Applications/picasso-workflow.app` onto the Dock
- **Desktop alias**: in Finder open `~/Applications`, then drag the app
  to `~/Desktop` while holding `Cmd+Alt`

**Icon** — the script converts `picasso_workflow/picasso-workflow.ico`
to the macOS `.icns` format automatically using `sips` and `iconutil`
(both are built into macOS).  No extra tools needed.

Re-run the script after upgrading the package or moving the conda
environment.

## Testing

The test suite is organised in four tiers. The first two tiers run without
any external dependencies and are executed by CI on every push. Tiers 3 and 4
require a working `picassosr` installation and are run explicitly before
merging to `master`. Tier 4 additionally requires access to lab network
volumes and is run on a lab machine.

### Tier 1 — Unit tests

```bash
pytest                        # run all non-integration tests
pytest -v                     # verbose output
pytest -m "not integration"   # explicitly exclude integration tests
```

Each module in `analyse.py` / `workflow.py` / `confluence.py` has a
corresponding unit-test file under `picasso_workflow/tests/`. Picasso is
fully mocked so these tests run anywhere without data or network access.

### Tier 2 — Template structural validation

```bash
pytest                        # included automatically in the normal run
```

`test_template_validation.py` imports every snapshotted `start_workflow.py`
from `picasso_workflow/tests/TestData/templates/` and asserts that every
module name referenced in the template exists in `AutoPicasso`. This catches
regressions where a module is renamed or removed while a production template
still references the old name. No picasso installation or data files are
required. When the templates directory is empty the test is silently skipped.

### Tier 3 — Integration tests

```bash
pytest -m integration
```

These tests run the **real picasso pipeline** against minimal bundled OME-TIFF
datasets (`picasso_workflow/tests/TestData/integration/`). Confluence
reporting is replaced by a `MagicMock` so no credentials or network access are
needed. The tests are skipped automatically if `picassosr` is not installed.

**What is tested:**

| Test | Description |
|---|---|
| `Test_A::test_01` | `load → identify → localize` on a single 30 px / 1k-frame stack |
| `Test_A::test_02` | same pipeline × 2 channels + `align_channels` aggregation |
| `test_03_undrift_rcc` | full pipeline including `undrift_rcc` on a 5 000-frame synthetic movie |
| `test_template_smoke[<name>]` | first safe modules of each snapshotted template, real data path substituted with bundled file |
| `Test_B::test_01` | same as `test_01` but with a live Confluence reporter (requires env vars below) |

The `test_03_undrift_rcc` test uses a **session-scoped synthetic movie** (5 000 frames, 128 × 128 px, ~20 Gaussian emitters on Poisson background) generated in `conftest.py`. It does not require any external data files.

**Confluence integration** (optional, skipped when env vars are absent):

```bash
export TEST_CONFLUENCE_URL=https://your-confluence-instance
export TEST_CONFLUENCE_USERNAME=your-username
export TEST_CONFLUENCE_TOKEN=your-api-token
export TEST_CONFLUENCE_SPACE=SPACE_KEY
export TEST_CONFLUENCE_PAGE=Parent Page Title
pytest -m integration
```

### Tier 4 — Real acquired-data tests

```bash
export PW_TEST_DATA_DIR=/Volumes/pool-miblab1/users/<you>/test-datasets
pytest -m "integration and real_data"
```

Or configure the path once in `~/.config/picasso_workflow/config.yaml`:

```yaml
TestData:
  directory: /Volumes/pool-miblab1/users/<you>/test-datasets
```

`test_real_data_integration.py` discovers real OME-TIFF acquisitions under
`PW_TEST_DATA_DIR` and runs the production pipeline against them. All tests
carry both the `integration` and `real_data` markers and are **skipped
automatically** when the path is not set or the directory is not mounted.

**What is tested:**

| Test | Description |
|---|---|
| `test_load_picassoconfig` | checks the picasso config referenced in `config.yaml` is readable |
| `test_minimal_pipeline_on_real_data` | `load → identify (auto net_gradient) → localize` on up to 3 real movies |
| `test_full_pipeline_undrift_on_real_data` | full pipeline including `undrift_rcc` and `save` on the first movie found |

### Keeping template snapshots up to date

Production workflow templates live on the lab network volumes and are listed
in `picasso_workflow/config.yaml` under `Templates:`. A snapshot of each
template's `start_workflow.py` is committed to the repository so that Tier 2
and Tier 3 template tests can run offline.

Run the snapshot script **on a machine that can access the pool volumes**
whenever a template is created or updated:

```bash
python tools/snapshot_templates.py
git add picasso_workflow/tests/TestData/templates/
git commit -m "update template snapshots"
```

The script copies only `start_workflow.py` (the workflow module list). File
lists (`src_loc.yaml`) that contain absolute paths to acquired data are
intentionally excluded from the repository.

### Running all tiers on the SLURM cluster

The scripts in `tools/cluster_tests/` let you run the full test suite as
a SLURM job chain.  Each tier is submitted as a separate job; a tier starts
only if the previous one passed (`--dependency=afterok`), so a Tier 1
failure automatically cancels Tiers 2–4 without wasting compute time.

```
submit_all.sh
    │
    ├─► [job A] tier1_2.sbatch   unit + template validation
    │         afterok:A ↓
    ├─► [job B] tier3.sbatch     integration (synthetic + bundled data)
    │         afterok:B ↓
    └─► [job C] tier4.sbatch     real acquired data (skips if not mounted)
```

#### Prerequisites

Before the first run, make sure the following are in place on the cluster:

1. **Project is checked out** (or accessible via a network path) on the
   cluster, e.g.:
   ```bash
   git clone <repo-url> ~/picasso-workflow
   ```
2. **`picasso-workflow` conda environment is installed** on the cluster.
   Follow the same steps as [Installation](#installation):
   ```bash
   conda create -n picasso-workflow python=3.10
   conda activate picasso-workflow
   cd ~/picasso-workflow
   pip install -e .
   ```
   Verify: `python -c "import picasso; import picasso_workflow; print('OK')`
3. **Module name matches** — the `.sbatch` files load
   `anaconda/3/2023.03`.  Check what is available on your cluster with
   `module avail anaconda` and edit the `module load` line if needed.
4. **Pool volumes are mounted on compute nodes** (Tier 4 only) — ask your
   cluster administrator.  Tier 4 tests skip gracefully if the directory
   is not accessible, so this is only needed for real-data coverage.

#### Submitting the test chain

SSH to the cluster login node, navigate to the project, and run
`submit_all.sh`:

```bash
ssh clusterXXX
cd ~/picasso-workflow

# Tiers 1–3 (no real data required):
tools/cluster_tests/submit_all.sh

# All four tiers — option A: set the env var for this session
export PW_TEST_DATA_DIR=/path/to/real/datasets
tools/cluster_tests/submit_all.sh

# All four tiers — option B: path already in ~/.config/picasso_workflow/config.yaml
tools/cluster_tests/submit_all.sh   # no env var needed
```

**How `PW_TEST_DATA_DIR` is resolved** (same rule locally and on the cluster):

The `network_test_data` fixture checks these sources in order, stopping at the first non-empty result:

1. `PW_TEST_DATA_DIR` environment variable
2. `TestData → directory` in `~/.config/picasso_workflow/config.yaml`
3. _(skip — no path configured)_

On most HPC clusters the home directory is NFS-mounted and shared between login nodes and compute nodes, so `~/.config/picasso_workflow/config.yaml` is the same file everywhere.  If you have already set `TestData.directory` there for local Tier 4 runs, the cluster jobs pick it up automatically without any extra env var.  The env var is only needed if you want to override the config for a specific run.

The script prints the three job IDs and a ready-made `squeue` command:

```
Project directory: /home/you/picasso-workflow
Results directory: /home/you/picasso-workflow/test-results

Submitted Tier 1+2 (unit + template):  job 12345
Submitted Tier 3  (integration):        job 12346  (depends on 12345)
Submitted Tier 4  (real data):          job 12347  (depends on 12346)

Monitor:  squeue -j 12345,12346,12347
Tail log: tail -f test-results/tier1_2_12345.log
```

#### Monitoring progress

```bash
# Live queue view (refreshes every 2 s):
watch -n 2 squeue -j 12345,12346,12347

# Tail the log of the currently running tier:
tail -f test-results/tier1_2_12345.log
```

Common SLURM job states:

| State | Meaning |
|---|---|
| `PD` | Pending — waiting in the queue or for dependency |
| `R` | Running |
| `CG` | Completing — cleaning up |
| `CD` | Completed successfully (exit 0) |
| `F` | Failed (non-zero exit — pytest reported failures) |
| `CA` | Cancelled — a dependency failed, so this tier was skipped |

If Tier 3 shows `F`, Tier 4 will show `CA` — look at the Tier 3 log to
find the failing test.

#### Reading the results

Results land in `test-results/` (gitignored):

```
test-results/
    tier1_2_12345.log   # full pytest output + SLURM bookkeeping
    tier1_2_12345.xml   # JUnit XML (machine-readable)
    tier3_12346.log
    tier3_12346.xml
    tier4_12347.log
    tier4_12347.xml
```

The last few lines of each `.log` file contain the pytest summary:

```
PASSED picasso_workflow/tests/test_z_integration.py::...
FAILED picasso_workflow/tests/test_z_integration.py::... - AssertionError
====== 5 passed, 1 failed in 23.4s ======
```

#### Resubmitting a single tier

If only one tier needs to be re-run (e.g. after a bug fix):

```bash
cd ~/picasso-workflow

# Re-run Tier 3 only:
sbatch --export=ALL,PW_PROJECT_DIR="$(pwd)" \
       tools/cluster_tests/tier3.sbatch

# Re-run Tier 4 with real data:
export PW_TEST_DATA_DIR=/path/to/real/datasets
sbatch --export=ALL,PW_PROJECT_DIR="$(pwd)" \
       tools/cluster_tests/tier4.sbatch
```

#### Adapting to a different cluster

All cluster-specific settings are at the top of each `.sbatch` file.
Things you may need to change:

| Setting | Location | Default |
|---|---|---|
| Anaconda module name | `module load …` line | `anaconda/3/2023.03` |
| Conda env name | `conda activate …` line | `picasso-workflow` |
| Memory / CPUs / time | `#SBATCH` directives | per-file defaults |
| Partition / QOS | add `#SBATCH --partition=…` | _(none — cluster default)_ |

### Adding a new workflow module

When adding a module, make sure all tiers remain green:

1. Add unit tests to `test_analyse.py` and `test_confluence.py` (mocked).
2. Re-run `pytest` — Tier 1 and Tier 2 must pass.
3. Run `pytest -m integration` — Tier 3 must pass.
4. If any snapshotted template uses the renamed/removed module, update
   `standard_singledataset_workflows.py` or `standard_aggregation_workflows.py`
   and re-run `python tools/snapshot_templates.py`.
5. On a lab machine with `PW_TEST_DATA_DIR` set, run
   `pytest -m "integration and real_data"` — Tier 4 must pass.

## CI / GitHub Actions

Two GitHub Actions workflows run automatically on every push and pull request
to `master` and `develop`.

| Workflow file | Runner | What it runs | When |
|---|---|---|---|
| `run-unittests.yml` | Windows self-hosted | `pytest` (all mocked unit tests) + coverage | every push / PR |
| `run-cluster-tests.yml` | Linux self-hosted on cluster | SLURM Tiers 1–3 (unit + template + integration) | every push / PR |
| `run-cluster-tests.yml` | Linux self-hosted on cluster | SLURM Tier 4 (real data) | push to `master` only |

### How the cluster CI workflow works

`run-cluster-tests.yml` runs on a self-hosted runner registered on the
cluster login node.  It submits individual `sbatch` jobs (the same scripts
used manually via `submit_all.sh`) and polls `squeue` until they finish,
then checks exit codes via `sacct` and uploads the JUnit XML reports as
workflow artifacts.

```
GitHub Actions runner (login node)
    │
    ├─ sbatch tier1_2.sbatch  ──► compute node  [unit + template, ≤15 min]
    │       afterok ↓
    ├─ sbatch tier3.sbatch    ──► compute node  [integration,     ≤30 min]
    │       (on push to master only)
    │       afterok ↓
    └─ sbatch tier4.sbatch    ──► compute node  [real data,       ≤12 h  ]
```

### Setting up the cluster self-hosted runner

This only needs to be done once per cluster.  Run all commands on the
**cluster login node** that has access to `sbatch`.

**1. Register the runner in GitHub**

Go to the repository → *Settings* → *Actions* → *Runners* →
*New self-hosted runner*.  Select **Linux / x64** and follow the
displayed download and configuration commands.

When the interactive `config.sh` script asks for labels, enter:

```
self-hosted,linux,cluster
```

These three labels are what `run-cluster-tests.yml` uses to select this
runner (`runs-on: [self-hosted, linux, cluster]`).

**2. Install the runner as a persistent service**

So the runner survives SSH session disconnects and cluster reboots:

```bash
cd ~/actions-runner          # or wherever you installed it
sudo ./svc.sh install        # installs a systemd service
sudo ./svc.sh start
sudo ./svc.sh status         # should show "active (running)"
```

If you do not have `sudo` on the login node, use a `screen` or `tmux`
session as a fallback:

```bash
screen -S gh-runner
cd ~/actions-runner
./run.sh
# Ctrl-A D to detach
```

**3. Verify SLURM is on the runner's PATH**

The runner process inherits the environment of the user who started it.
Check that `sbatch`, `squeue`, and `sacct` are accessible:

```bash
which sbatch squeue sacct
```

If not, add the SLURM `bin` directory to `~/.bashrc` (or `~/.profile`
for non-interactive sessions) and restart the runner service.

**4. Ensure the conda environment exists**

The `.sbatch` scripts activate the `picasso-workflow` conda environment.
Follow the [Installation](#installation) steps on the cluster if you
have not done so already, then verify:

```bash
conda activate picasso-workflow
python -c "import picasso; import picasso_workflow; print('OK')"
```

If the module name `anaconda/3/2023.03` used in the `.sbatch` files does
not exist on your cluster, edit the `module load` line in each file
(`tools/cluster_tests/tier1_2.sbatch`, `tier3.sbatch`, `tier4.sbatch`).

### Enabling Tier 4 real-data tests in CI

Tier 4 runs only on push to `master` and requires the path to the real
acquired-data directory.  Set it as a repository-level Actions variable
(not a secret — it is a plain path):

*Settings → Secrets and variables → Actions → Variables → New repository variable*

| Name | Example value |
|---|---|
| `PW_TEST_DATA_DIR` | `/fs/pool-miblab1/users/you/test-datasets` |

The path must be accessible on the cluster compute nodes (pool volumes
must be mounted there).  If the variable is not set or the directory is
not mounted, all `real_data` tests are skipped automatically and the CI
job still passes.

### Artifacts

After each run, JUnit XML reports are uploaded as workflow artifacts:

- **`cluster-test-results-tier1-3`** — `tier1_2_<jobid>.xml` and `tier3_<jobid>.xml`
- **`cluster-test-results-tier4`** — `tier4_<jobid>.xml` (master pushes only)

Download them from the *Actions* tab → select a run → *Artifacts* section.

## Releasing

Versions are derived automatically from git tags by
[setuptools-scm](https://github.com/pypa/setuptools_scm).
There are no version numbers to edit in any file — **the tag IS the
version**.  After `pip install -e .`, the current version is always
accessible at:

```python
import picasso_workflow
print(picasso_workflow.__version__)
```

Between tagged commits the version looks like `1.2.3.dev4+gabcdef`
(commits since tag + short hash).  On an exact tag it is just `1.2.3`.

### Release workflow

```
develop:  A──B──C──D          (feature work, tests pass)
                    \
master:              M──[tag v1.2.3]
                    /
develop (synced):  M
```

**1. Finish and test on `develop`**

Make sure all CI checks pass on `develop` before touching `master`.

**2. Merge `develop` → `master`**

```bash
git checkout master
git merge --no-ff develop      # --no-ff keeps the merge commit
git push origin master
```

Or open a pull request and merge it on GitHub.

**3. Tag the release on `master`**

```bash
git checkout master             # (already there)
git tag v1.2.3                  # annotated tags are fine too: git tag -a v1.2.3 -m "v1.2.3"
git push origin v1.2.3
```

Tag format must be `vMAJOR.MINOR.PATCH` (e.g. `v1.2.3`).

**4. Sync `develop` back to `master`**

```bash
git checkout develop
git merge master                # fast-forwards develop to the merge commit
git push origin develop
```

This is a fast-forward (no new commit), so `develop` and `master` now
point to the same commit and are in sync for the next cycle.

### Choosing a version number

Follow [Semantic Versioning](https://semver.org):

| Change | Example bump |
|---|---|
| Bug fix, small patch | `v1.2.2` → `v1.2.3` |
| New feature, backwards-compatible | `v1.2.3` → `v1.3.0` |
| Breaking change | `v1.3.0` → `v2.0.0` |

### First release (no tags yet)

Until the first tag is pushed, the version reported is `0.0.0.dev0`.
Create the initial tag on `master` after the first merge:

```bash
git checkout master
git tag v0.1.0
git push origin v0.1.0
```

## Contributing

- Install pre commit hooks:
	- `pip install pre-commit` (if not already installed by requirements in pyproject.toml / pip install -e)
	- `cd GitHub/picasso-workflow`
	- `pre-commit install`
	- Now, before commit via git, the hooks will run through and check code and style
	- optionally, the hooks can be run manually: `pre-commit run --all-files`
- For adding new workflow modules, create a new branch (feature/newmodule),
and add new modules to:
	- util/AbstractModuleCollection
	- analyse/AutoPicasso
	- confluence/ConfluenceReporter
	- tests/test_analyse
	- tests/test_confluence
- make sure unit tests run through smoothly (see [Testing](#testing) for the full test workflow):
	- `cd GitHub/picasso-workflow`
	- `pytest -v`                  # unit + template validation
	- `pytest -m integration`      # full integration tests (requires picassosr)
- Please adhere to PEP code style and send pull request when done.

## License

This project is licensed under the [MIT License](LICENSE).
