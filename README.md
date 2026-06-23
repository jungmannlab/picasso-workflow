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

### Prerequisites

Make sure to have (ana)conda installed. On Mac OS, open the terminal (command + space,
type "terminal" hit enter). Then, one after another execute the follwing commands
- `curl -O https://repo.anaconda.com/archive/Anaconda3-2024.09-MacOSX-x86_64.sh`
- `bash Anaconda3-2024.09-MacOSX-x86_64.sh`
- `~/anaconda3/bin/conda init`
- `conda config --remove channels defaults`
- `conda config --add channels conda-forge`
- close the terminal and reopen it, to apply the changes.

### picasso-workflow specific installation

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
- if you have access, see examples in "/Volumes/pool-miblab/users/grabmayr/picasso-workflow_testdata"

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

### Site-wide default configuration (all-users installs)

When picasso-workflow is installed for all users, individual users may not
have their own `config.yaml` yet.  An administrator can place a shared
default at:

| Platform | Site config path |
|---|---|
| Windows | `C:\ProgramData\picasso_workflow\config.yaml` |
| macOS / Linux | `/etc/picasso_workflow/config.yaml` |

Config files are **deep-merged** in this priority order (highest wins):

1. Per-user — `~/.config/picasso_workflow/config.yaml`
2. Site-wide — path above
3. Bundled package default

Each file only needs to contain the keys it wants to override.  For
example, a site config that sets shared cluster and Confluence defaults
while leaving everything else to the package default:

```yaml
Confluence:
  URL: "https://confluence.example.com"
  Space: "PAINT"
SlurmLoginNodes:
  hpccluster: hpcl8001
ClusterEnvironment:
  anaconda_module: "anaconda/3/2023.03"
  conda_env: "picasso-workflow"
```

Users then only need their own config if they want to override something
specific (e.g. their personal Confluence page or a different template
path).  Keys they do not specify are inherited from the site config.

To create the directory and drop in the config on Windows (elevated prompt):

```powershell
New-Item -ItemType Directory -Force "C:\ProgramData\picasso_workflow"
# then copy or create config.yaml there
```

### Confluence credentials

Confluence credentials are split into **non-secret connection settings** and
the **secret token**, and the same scheme covers all three use cases (the
pytest suite, CI, and GUI-launched workflows):

- **Non-secret settings** (`URL`, `Space`, `DefaultPage`, `Username`) live in
  `config.yaml`, in two sections:
  - `Confluence` — operational target (real workflow runs / the GUI).
  - `ConfluenceTest` — target for the pytest suite (a dedicated test space, so
    tests never write to your operational space).

  Override them per machine in your per-user `config.yaml`, or per field with
  the matching environment variables: operational
  `CONFLUENCE_URL` / `CONFLUENCE_SPACE` / `CONFLUENCE_BASE_PAGE` /
  `CONFLUENCE_USERNAME`; tests `TEST_CONFLUENCE_URL` / `TEST_CONFLUENCE_SPACE` /
  `TEST_CONFLUENCE_PAGE` / `TEST_CONFLUENCE_USERNAME`.

- **The token is *only* ever an environment variable** — never stored in
  `config.yaml`, a generated `start_workflow.py`, results files, or logs:
  - operational: `CONFLUENCE_TOKEN` (legacy alias `CONFLUENCE_BEARER`)
  - tests: `TEST_CONFLUENCE_TOKEN`

  All of them resolve through one helper,
  `picasso_workflow.confluence.resolve_confluence_credentials(profile)`.

**Where to set the token per use case**

- **Local (laptop)** — export in your shell (or `picasso_workflow/.env`, which
  is loaded automatically at import by python-dotenv).
- **Cluster** — set it so both the login node *and* SLURM jobs see it.

  > **Pitfall:** plain `export`s in `~/.bashrc` usually do **not** reach a
  > batch job. A distro-default `~/.bashrc` starts with a non-interactive
  > guard (`case $- in *i*) ;; *) return;; esac`) that `return`s before any
  > later lines run. SLURM job scripts run non-interactively, so a
  > `source ~/.bashrc` inside them bails out before reaching your exports —
  > and `--export=ALL` only carries variables that were actually *exported*
  > into the submitting environment.

  **Recommended:** keep the credentials in a dedicated secrets file,
  `~/.picasso_secrets`, that contains only `export` lines and has no
  interactivity guard, so it runs the same way in interactive and batch
  shells:

  ```bash
  cat > ~/.picasso_secrets <<'EOF'
  export CONFLUENCE_TOKEN='…'        # operational (real workflow runs)
  export TEST_CONFLUENCE_TOKEN='…'   # pytest suite
  EOF
  chmod 600 ~/.picasso_secrets
  ```

  Both job paths source this file unconditionally
  (`[ -f ~/.picasso_secrets ] && source ~/.picasso_secrets`): the test
  tiers under `tools/cluster_tests/` and the `run_workflow_slurm.sh` scripts
  the GUI generates for `start_workflow.py`. `submit_all.sh` additionally
  submits with `--export=ALL`, so anything exported on the login node is
  carried in too.

  Alternative: a `.env` next to the installed package, loaded at
  `import picasso_workflow` by python-dotenv, which also works identically
  in interactive shells and batch jobs.
- **CI** — the runner provides only the token env var; the non-secret test
  settings come from the bundled `ConfluenceTest` section.

The GUI's Documentation Config tab shows the non-secret fields (prefilled from
`config.yaml`) but **has no token field** — it always reads `CONFLUENCE_TOKEN`
from the environment at run time.

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
to the macOS `.icns` format automatically using Pillow (installed with
the package) and `iconutil` (built into macOS).  No extra tools needed.

Re-run the script after upgrading the package or moving the conda
environment.

## Testing

### Strategy at a glance

Testing is layered so that the **fast feedback you need while coding stays
local and cheap, while the slow, hardware-hungry checks run on the cluster**
and report back in one place. The selection is driven entirely by two pytest
markers — `integration` and `real_data` — so the *same* test files serve every
layer; you just change which marker expression you select.

The suite is organised in four tiers of increasing cost and fidelity:

| Tier | What it checks | Select with | Needs | Speed |
|---|---|---|---|---|
| **1 — Unit** | every module in isolation, picasso fully mocked | `pytest` (default) | nothing | seconds |
| **2 — Template validation** | snapshotted templates only reference modules that still exist | `pytest` (runs with tier 1) | nothing | seconds |
| **3 — Integration** | the real picasso pipeline on tiny bundled / synthetic data | `pytest -m integration` | `picassosr` | minutes |
| **4 — Real data** | the production pipeline on real acquisitions | `pytest -m "integration and real_data"` | `picassosr` + `PW_TEST_DATA_DIR` mounted | up to hours |

The default `addopts` deselect the `integration` mark, so a bare **`pytest` is
unit-only (tiers 1 + 2)** — that is the loop you run constantly. Everything
heavier is opt-in by marker, or runs on the cluster.

There are three entry points, in increasing order of coverage:

- **`pytest`** — local development. Runs whatever your machine can: tiers 1 + 2
  always, tier 3 if `picassosr` is installed, tier 4 if a data directory is
  configured and mounted.
- **`tools/cluster_tests/submit_all.sh`** — the cluster. Submits all four tiers
  **plus one end-to-end run of every detected template workflow**, then a
  dependent summary job condenses the whole run into a single
  `test-results/latest/SUMMARY.txt`. This is the "submit once, read one file"
  path and the only one that exercises the real templates against real data.
- **GitHub Actions** — automation. `run-unittests.yml` runs the unit tests on
  every push/PR; `run-cluster-tests.yml` runs tiers 1–3 on the cluster for
  every push/PR (tier 4 on push to `master`), surfacing the same `SUMMARY.txt`
  on the run page. See [CI / GitHub Actions](#ci--github-actions).

**Typical developer workflow**

1. While coding, run `pytest` (or `pytest -k <name>`) for the fast unit loop.
2. Touching anything in the picasso pipeline? Run `pytest -m integration`
   locally if you have `picassosr`, to catch pipeline breakage early.
3. Renamed/removed a workflow module, or changed a standard workflow? Re-run
   `python tools/snapshot_templates.py` so the committed template snapshots
   (tiers 2 & 3) stay in sync — see
   [Keeping template snapshots up to date](#keeping-template-snapshots-up-to-date).
4. Push — CI runs the unit tests and cluster tiers 1–3 automatically.
5. Before merging to `master` (or whenever you want full coverage including the
   real templates), run `tools/cluster_tests/submit_all.sh` on the cluster and
   read `test-results/latest/SUMMARY.txt`.

The rest of this section documents each tier, the cluster runner, and CI in
detail.

### Tier 1 — Unit tests

The default `addopts` deselect the `integration` mark, so a bare `pytest` is
**unit-only** — the fast loop you want while developing locally. The heavy
integration tests (real picasso pipeline) only run when you ask for them, or
on the cluster.

```bash
pytest                        # unit tests only (integration deselected)
pytest -v                     # verbose, still unit-only
pytest -m integration         # opt in to the heavy integration tests
pytest -m ""                  # run everything (clears the default filter)
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

**Confluence integration** (optional, skipped when the test token is absent):

The live Confluence test needs only the **token** as an environment variable;
the non-secret connection settings come from the `ConfluenceTest` section of
`config.yaml` (see [Confluence credentials](#confluence-credentials)).

```bash
export TEST_CONFLUENCE_TOKEN=your-test-api-token
pytest -m integration
```

If your test instance differs from the bundled `ConfluenceTest` defaults, set
it once in your per-user `config.yaml`, or override individual fields with
`TEST_CONFLUENCE_URL` / `TEST_CONFLUENCE_SPACE` / `TEST_CONFLUENCE_PAGE` /
`TEST_CONFLUENCE_USERNAME`.

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

Each detected template workflow is also submitted as its own end-to-end job,
and a final **summary job** (`afterany` on everything) condenses the whole
run into a single `SUMMARY.txt` — so you submit one command and read one
file.

```
submit_all.sh
    │
    ├─► [job A] tier1_2.sbatch        unit + template validation
    │         afterok:A ↓
    ├─► [job B] tier3.sbatch          integration (synthetic + bundled data)
    │         afterok:B ↓
    ├─► [job C] tier4.sbatch          real acquired data (skips if not mounted)
    │
    ├─► [job T1..Tn] <template>/run_workflow_slurm.sh
    │                                 one end-to-end run per detected template
    │                                 (afterok:B; only if PW_TEST_DATA_DIR set)
    │         afterany: A,B,C,T1..Tn ↓
    └─► [job S] summary.sbatch        writes SUMMARY.txt + summary.json
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

The script prints all job IDs, a ready-made `squeue` command, and where the
final report will appear:

```
Project directory: /home/you/picasso-workflow
Results directory: /home/you/picasso-workflow/test-results/20260608_..._master_b9e6d95

Submitted Tier 1+2 (unit + template):  job 12345
Submitted Tier 3  (integration):        job 12346  (depends on 12345)
Submitted Tier 4  (real data):          job 12347  (depends on 12346)
Submitted template run:                 job 12350  (.../templates/eva-full)
Submitted template run:                 job 12351  (.../templates/basic-loc)
Submitted summary (report):             job 12352  (depends on all above)

Monitor:  squeue -j 12345,12346,12347,12350,12351,12352
Tail log: tail -f test-results/.../tier1_2_12345.log
Report:   test-results/.../SUMMARY.txt  (written when summary job 12352 finishes)
          → also reachable at test-results/latest/SUMMARY.txt
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

**Start with `SUMMARY.txt`.** Each run gets its own timestamped directory
under `test-results/` (gitignored), and `test-results/latest` always points
at the newest one. The summary job writes a single consolidated report there
once everything finishes:

```bash
cat test-results/latest/SUMMARY.txt
```

```
======================================================================
 picasso-workflow cluster test summary
======================================================================
run_id:     20260608_..._master_b9e6d95
OVERALL: FAIL   (2/3 tiers passed, 1/2 template workflows ran through)

 Pytest tiers
 TIER       JOB    STATE      TESTS PASS FAIL SKIP  RESULT
 tier1_2    12345  COMPLETED    210  208    0    2  PASS
 tier3      12346  COMPLETED     18   17    1    0  FAIL
 tier4      12347  COMPLETED      3    0    0    3  PASS (all skipped)
   tier3 failures:
     - test_z_integration.py::Test_A::test_02_minimal_channel_align

 Template workflows (end-to-end runs)
 Ran through: 1 / 2
 NAME                     JOB    STATE      ERR  RESULT
 eva-full                 12350  FAILED       4  FAIL
 basic-loc                12351  COMPLETED    0  PASS
   eva-full: job state FAILED, 4 error line(s) in logs
     log: .../templates/eva-full/logs/picasso-workflow-job12350-rank0.log
```

The headline line answers the two questions at a glance: did each tier pass,
and **how many detected template workflows ran through** (verdict =
SLURM job `COMPLETED` *and* no `ERROR`/`Traceback` lines in that job's logs).
The summary job's own exit code is 0 on overall PASS, 1 otherwise.

Need a snapshot before the run finishes? Run the summarizer by hand against
any run directory (it reads whatever artefacts exist so far):

```bash
python3 tools/cluster_tests/summarize.py test-results/latest
```

Underlying artefacts, if you need to drill in, all live in the same dir:

```
test-results/<run_id>/
    SUMMARY.txt          # consolidated report (read this first)
    summary.json         # same data, machine-readable
    run_info.txt         # run metadata + job IDs
    jobs.tsv             # job manifest the summarizer reads
    tier1_2_12345.log    # full pytest output + SLURM bookkeeping
    tier1_2_12345.xml    # JUnit XML (machine-readable)
    tier3_12346.log / .xml
    tier4_12347.log / .xml
    summary_12352.log    # the report, echoed by the summary job
# Each template workflow's own logs stay next to the template:
#   <PW_TEST_DATA_DIR>/.../<template>/logs/picasso-workflow-job<jid>-rank*.log
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
then checks exit codes via `sacct`.

Once the jobs finish, each CI job runs the same
`tools/cluster_tests/summarize.py` used by `submit_all.sh` to render a
consolidated `SUMMARY.txt`.  The report is written into the **GitHub Actions
run page** (the job summary, so PASS/FAIL and any failing test names are
visible without downloading anything) and uploaded as a workflow artifact
alongside `summary.json` and the JUnit XML.  Because the `tiers-1-3` and
`tier-4` jobs run separately, each publishes its own report slice.

> The CI does not submit the per-template end-to-end workflow runs (those
> need `PW_TEST_DATA_DIR` and add hours of walltime), so the report's
> "Template workflows ran through" section is empty in CI.  Run
> `submit_all.sh` on the cluster with `PW_TEST_DATA_DIR` set to exercise and
> report on those.

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
