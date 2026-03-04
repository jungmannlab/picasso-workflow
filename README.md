# picasso-workflow

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

master:
![master test](https://img.shields.io/github/actions/workflow/status/jungmannlab/picasso-workflow/run-unittests.yml?branch=master)

develop:
![develop test](https://img.shields.io/github/actions/workflow/status/jungmannlab/picasso-workflow/run-unittests.yml?branch=develop)
![Coveralls develop](https://img.shields.io/coverallsCoverage/github/jungmannlab/picasso-workflow?branch=develop)


A package for automated DNA-PAINT analysis workflows

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
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
