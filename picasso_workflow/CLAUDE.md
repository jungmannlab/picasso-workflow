# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `picasso-workflow`, a Python package for automated DNA-PAINT analysis workflows. The project provides two main workflow types:
- **Single-dataset workflow**: Process individual datasets through localization, clustering, and analysis
- **Aggregation workflow**: Process multiple datasets and aggregate results

## Key Architecture

### Core Modules
- **`workflow.py`**: Main orchestration class `ReportingAnalyzer` that coordinates picasso analysis and confluence reporting. Contains `WorkflowRunner` and `AggregationWorkflowRunner` classes.
- **`analyse.py`**: Contains `AutoPicasso` class - the main picasso interface for localization, clustering, and analysis operations
- **`confluence.py`**: Contains `ConfluenceReporter` and `ConfluenceInterface` classes for generating and uploading analysis reports to Confluence
- **`metaworkflow.py`**: Higher-level analysis functionality for running workflows across multiple conditions and cells with aggregation
- **`util.py`**: Contains `AbstractModuleCollection` and other utility classes for parameter management

### Specialized Modules
- **`dbscan_molint/`**: DBSCAN clustering with molecular interactions analysis
- **`outpost_modules/`**: Extended analysis modules including binding event analysis, rendering, and Ripley's K analysis
- **`ripleys_analysis/`**: Dedicated Ripley's K spatial analysis functionality
- **`spinna*.py`**: SPINNA (Spatial Point Pattern Analysis) related modules for MLE fitting

### Standard Workflows
- **`standard_singledataset_workflows.py`**: Pre-built single dataset analysis workflows
- **`standard_aggregation_workflows.py`**: Pre-built aggregation analysis workflows

## Development Commands

### Testing
```bash
# Run all tests
pytest -v

# Run specific test file
pytest -v picasso_workflow/tests/test_workflow.py

# Run tests with coverage
pytest --cov=picasso_workflow
```

### Code Quality
```bash
# Install and run pre-commit hooks
pip install pre-commit
pre-commit install
pre-commit run --all-files

# Manual linting (based on pyproject.toml config)
black picasso_workflow/
isort picasso_workflow/
flake8 picasso_workflow/
```

### Installation
```bash
# Development installation
pip install -e .

# With cluster support
pip install -e .[cluster]
```

## Configuration

### Environment Variables
- `CONFLUENCE_TOKEN`: API token for operational Confluence integration (legacy
  alias `CONFLUENCE_BEARER`). The token is *only* ever an env var — never in
  `config.yaml`, generated scripts, or logs.
- `TEST_CONFLUENCE_TOKEN`: API token for the pytest suite's Confluence tests.
- `DRIVEPATHS`: Drive path mappings for multi-machine compatibility (format: "machine:path;machine:path")

Non-secret Confluence connection settings (URL/Space/DefaultPage/Username) live
in `config.yaml` under `Confluence` (operational) and `ConfluenceTest` (tests);
all credentials resolve through `confluence.resolve_confluence_credentials(profile)`.

### Code Style
- Line length: 79 characters (Black) / 79 characters (Flake8)
- Uses Black, isort, flake8, mypy, and bandit for code quality
- Docstring convention: NumPy style (numpydoc) — one-line imperative
  summary, then `Parameters` / `Returns` / `Raises` / `Notes` sections with
  dashed underlines and `name : type` fields. Pair with PEP 604 type hints in
  signatures (`from __future__ import annotations`); don't restate a
  parameter's type in prose when the annotation already gives it. Matches the
  upstream `picasso` package.
- Test coverage requirement: 80%

## Testing Structure

Tests are located in `picasso_workflow/tests/` with test data in `TestData/`:
- Unit tests mock dependencies to test individual modules
- Integration tests in `test_z_integration.py` test full workflows
- Test data includes sample DNA-PAINT datasets for realistic testing

## Key Dependencies

- **picassosr**: Core analysis engine (>=0.7.3)
- **atlassian-python-api**: Confluence integration
- **mpi4py**: Cluster computing support (optional)
- **scipy**, **numpy**, **pandas**: Scientific computing
- **matplotlib**, **seaborn**: Visualization
- **aicsimageio**: Image I/O for microscopy formats

## Working inside the Claude Code sandbox

The default Claude Code sandbox blocks most of the things this package's import chain wants to do. Don't rediscover these every time — work around them up front.

### `import picasso_workflow` fails with a numba cache error

The full import pulls `picasso.postprocess`, which calls `@numba.njit(cache=True)` at module load. Numba then tries to write a cache file next to `postprocess.py` inside the installed `picassosr` package; the sandbox denies the write and raises:

```
RuntimeError: cannot cache function '_pick_similar': no locator available
for file '/.../site-packages/picasso/postprocess.py'
```

This has nothing to do with the code under test — `picasso_workflow/__init__.py` dies at line ~16 (`from picasso_workflow.workflow import ...`) before any of its own logic runs.

**Workaround for testing `__init__.py` side-effects** (config loading, `.env` discovery, logger setup): stub out the heavy submodules before importing the package, e.g.

```python
import sys, types
sys.path.insert(0, '/Users/hgrabmayr/GitHub/picasso-workflow')
for name in ('picasso_workflow.workflow',
             'picasso_workflow.standard_singledataset_workflows',
             'picasso_workflow.standard_aggregation_workflows'):
    mod = types.ModuleType(name)
    if name.endswith('.workflow'):
        class _W: pass
        mod.WorkflowRunner = _W
        mod.AggregationWorkflowRunner = _W
    sys.modules[name] = mod
mod = types.ModuleType('picasso_workflow._version')
mod.__version__ = 'test'
sys.modules['picasso_workflow._version'] = mod

import picasso_workflow  # now runs __init__.py without touching numba
```

For anything that genuinely needs `picasso.*` (workflow, analyse, …), run the test outside the sandbox or have the user run it.

### `$HOME` is mostly read-only

The sandbox `write` allowlist excludes most of `$HOME`. Writes to `~/.picasso_workflow/`, `~/.config/picasso_workflow/`, `~/.matplotlib/`, etc. raise `PermissionError: [Errno 1] Operation not permitted`. Writable paths include `.` (project dir), `$TMPDIR`, and a few specific dotfiles — see the sandbox config in the Bash tool description.

**To test code that writes under `$HOME`**, redirect `HOME` to a sandbox-writable dir per-invocation:

```bash
FAKE_HOME="$TMPDIR/fakehome"
rm -rf "$FAKE_HOME" && mkdir -p "$FAKE_HOME"
HOME="$FAKE_HOME" python my_script.py
```

`Path.home()` and `~` expansion both honor `$HOME`, so this transparently relocates user-config / log paths.

### Logger configuration eats your stderr sink

`picasso_workflow.config_logger()` calls `logger.remove()` and re-adds sinks pointing at the logfile + an ERROR-level stderr sink. If you do `logger.add(sys.stderr, level='DEBUG', ...)` **before** `import picasso_workflow`, the import wipes it. To inspect import-time log output, read the logfile after import (path is printed in the first INFO line: `~/.picasso_workflow/logs/picasso-workflow-job{SLURM_JOB_ID}-rank{SLURM_PROCID}.log`).

### `python -c` distorts python-dotenv discovery

`load_dotenv()` / `find_dotenv()` check `_is_interactive()` (true when `__main__` has no `__file__`) and silently switch from frame-inspection to `os.getcwd()` as the search root. In `python -c "..."` invocations, `__main__.__file__` is unset, so dotenv uses cwd and won't find the package-bundled `.env` unless cwd happens to be inside the package. **Always test dotenv behavior with a real `.py` script**, not `python -c`.

### `cd` in compound Bash commands triggers a permission prompt

Don't write `cd /some/dir && python ...` — the harness prompts. Use absolute paths, or set cwd-relevant env (`HOME=...`, `PYTHONPATH=...`) inline before the command.
