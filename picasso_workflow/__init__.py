from dotenv import load_dotenv, find_dotenv, dotenv_values

try:
    from picasso_workflow._version import __version__
except ImportError:
    __version__ = "unknown"

# import logging
from loguru import logger
import os
import sys
import yaml
from pathlib import Path
import importlib.resources
from picasso_workflow.workflow import WorkflowRunner, AggregationWorkflowRunner
from picasso_workflow import standard_singledataset_workflows
from picasso_workflow import standard_aggregation_workflows

# Runtime diagnostics captured during import (logging destination, .env and
# config.yaml sources). Populated by config_logger() / _log_dotenv() /
# load_config() below and surfaced read-only by the GUI "Doctor" tab; holds
# no secret values.
DIAGNOSTICS: dict = {}


# configure logger
def config_logger():
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     "%(asctime)s | %(name)s | %(funcName)s | %(levelname)s -> %(message)s"
    # )
    job_id = os.getenv("SLURM_JOB_ID")  # Get the job ID from the environment
    rank_id = os.getenv("SLURM_PROCID")
    # Log destination, in priority order:
    #   1. PW_LOG_DIR, if set — lets the submitting tooling co-locate the
    #      workflow log with the job's other artefacts (e.g. the SLURM
    #      stdout/stderr in a shared per-run directory).
    #   2. On SLURM, cwd/logs — next to the job's stdout/stderr logs.
    #   3. Local/GUI launches (cwd is unpredictable — Windows shortcut,
    #      macOS .app, etc.): anchor under the user's home so the log stays
    #      discoverable.
    pw_log_dir = os.getenv("PW_LOG_DIR")
    if pw_log_dir:
        log_dir = Path(pw_log_dir)
    elif job_id is not None:
        log_dir = Path.cwd() / "logs"
    else:
        log_dir = Path.home() / ".picasso_workflow" / "logs"
    # Tag the filename by job/rank on SLURM so parallel ranks don't clash.
    if job_id is not None:
        logfile = str(
            log_dir / f"picasso-workflow-job{job_id}-rank{rank_id}.log"
        )
    else:
        logfile = str(log_dir / "picasso-workflow.log")
    log_dir.mkdir(parents=True, exist_ok=True)
    # file_handler = handlers.RotatingFileHandler(
    #     logfile,
    #     maxBytes=1e6,
    #     backupCount=5,
    # )
    # file_handler.setFormatter(formatter)
    # file_handler.setLevel(logging.DEBUG)
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)
    # stream_handler.setLevel(logging.WARNING)
    # logger.addHandler(file_handler)
    # # logger.addHandler(stream_handler)
    logger.remove()  # Remove default stderr sink
    logger.add(
        logfile,
        format="{time:YYYY-MM-DD HH:mm:ss:SSS} | PID:{process} | {name} | {function} | {level} -> {message}",
        rotation="1 MB",
        retention=5,
        enqueue=True,
        serialize=False,
        level="DEBUG",
    )
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss:SSS} | PID:{process} | {name} | {function} | {level} -> {message}",
        level="ERROR",
    )
    logger.info(f"picasso-workflow log file: {logfile}")
    DIAGNOSTICS["logging"] = {
        "logfile": logfile,
        "log_dir": str(log_dir),
        "rotation": "1 MB",
        "retention": 5,
        "file_level": "DEBUG",
        "stderr_level": "ERROR",
        "slurm": job_id is not None,
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with override merged on top of base, recursively."""
    result = base.copy()
    for key, val in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _site_config_path() -> Path:
    """Return the platform-appropriate site-wide config path."""
    if sys.platform == "win32":
        base = Path(os.environ.get("ProgramData", r"C:\ProgramData"))
    else:
        base = Path("/etc")
    return base / "picasso_workflow" / "config.yaml"


def _user_config_dir() -> Path:
    """Per-user config directory (holds config.yaml, .env, onboarding marker).

    Single source of truth shared by load_config() and the GUI, so they can
    never disagree about where settings live.
    """
    return Path.home() / ".config" / "picasso_workflow"


def _load_yaml(path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _load_layer(path):
    """Load one YAML config layer, tolerating problems.

    Returns the parsed mapping, or ``None`` when the file is absent, unparseable
    (a typo in a hand-edited config.yaml), or not a mapping -- so a bad file
    degrades to "skip this layer" with a warning instead of crashing
    ``import picasso_workflow``. A missing file is skipped silently (the caller
    decides whether that is expected).
    """
    try:
        data = _load_yaml(path)
    except FileNotFoundError:
        return None
    except (TypeError, OSError, yaml.YAMLError) as exc:
        logger.warning(f"Ignoring unreadable config at {path}: {exc!r}")
        return None
    if not isinstance(data, dict):
        logger.warning(
            f"Ignoring config at {path}: expected a mapping, got "
            f"{type(data).__name__}."
        )
        return None
    return data


def _mask_value(value: str | None) -> str:
    """Mask a .env value so logs reveal only that (and roughly what) it is.

    The result shows whether the value is set and its first/last few
    characters, with the middle replaced by ``*``. Short values are fully
    starred so nothing meaningful leaks. Values may be secrets (e.g. the
    Confluence token), so the raw text is never logged.

    Parameters
    ----------
    value : str or None
        The value to mask (``None`` for an unset key).

    Returns
    -------
    str
        The masked representation.
    """
    if value is None:
        return "<unset>"
    if value == "":
        return "<empty>"
    n = len(value)
    if n <= 6:
        return "*" * n
    show = min(4, n // 4)
    return f"{value[:show]}{'*' * (n - 2 * show)}{value[-show:]}"


def _log_dotenv():
    """Locate the .env file actually used by load_dotenv() and log it.

    ``load_dotenv()`` (no args) walks up from the caller's ``__file__``
    directory looking for a ``.env``; ``find_dotenv()`` with default
    ``usecwd=False`` reproduces that, since this function is called from
    the same module body. Values are masked (see :func:`_mask_value`) so a
    secret such as the Confluence token never lands in the logs.
    """
    dotenv_path = find_dotenv()
    if dotenv_path:
        logger.info(f"Loading .env from: {dotenv_path}")
        try:
            values = dotenv_values(dotenv_path)
            masked = "\n".join(
                f"{key}={_mask_value(val)}" for key, val in values.items()
            )
            logger.info(f".env content (values masked):\n{masked}")
        except OSError as exc:
            logger.warning(f"Could not read .env at {dotenv_path}: {exc}")
    else:
        logger.info(
            "No .env file found by python-dotenv "
            f"(searched upward from {Path(__file__).parent})."
        )
    # record the source and the keys it defines (never the values) for the
    # Doctor tab
    keys = []
    if dotenv_path:
        try:
            keys = sorted(dotenv_values(dotenv_path).keys())
        except OSError:
            pass
    DIAGNOSTICS["dotenv"] = {"path": dotenv_path or None, "keys": keys}
    return dotenv_path


def load_config():
    """Load the picasso-workflow configuration yaml file.

    Configs are deep-merged in increasing priority order so that
    higher-priority files only need to specify the keys they override:

      1. Bundled package default (config.yaml / config_template.yaml)
      2. Site-wide admin config  (C:\\ProgramData\\picasso_workflow\\config.yaml
                                  or /etc/picasso_workflow/config.yaml)
      3. Per-user config         (~/.config/picasso_workflow/config.yaml)
    """
    sources = []  # (layer, path, loaded) for the Doctor tab
    # 1. Package default. config.yaml is gitignored and not shipped in a
    # wheel, so it is normally absent -- fall back to the bundled
    # config_template.yaml so defaults like "Templates" exist. A malformed
    # bundled file falls back too (rather than crashing import).
    pkg_config = importlib.resources.files("picasso_workflow").joinpath(
        "config.yaml"
    )
    config = _load_layer(pkg_config)
    if config is not None:
        logger.info(f"Loaded bundled config.yaml from: {pkg_config}")
        sources.append(("bundled", str(pkg_config), True))
    else:
        template = importlib.resources.files("picasso_workflow").joinpath(
            "config_template.yaml"
        )
        config = _load_layer(template)
        if config is not None:
            logger.info(
                f"Loaded bundled config_template.yaml from: {template}"
            )
            sources.append(("bundled-template", str(template), True))
        else:
            logger.warning("No usable bundled config.yaml / template.")
            config = {}
            sources.append(("bundled", "<package resource>", False))

    # 2. Site-wide admin config (optional). A typo in it is skipped, not fatal.
    site_config = _site_config_path()
    site_data = _load_layer(site_config) if site_config.exists() else None
    if site_data is not None:
        logger.info(f"Loaded site-wide config.yaml from: {site_config}")
        config = _deep_merge(config, site_data)
        sources.append(("site-wide", str(site_config), True))
    else:
        logger.debug(f"No usable site-wide config.yaml at {site_config}.")
        sources.append(("site-wide", str(site_config), False))

    # 3. Per-user config (optional). This is the file the GUI's onboarding
    # dialog creates/edits, so a user typo must degrade gracefully, not crash.
    user_config = _user_config_dir() / "config.yaml"
    user_data = _load_layer(user_config) if user_config.exists() else None
    if user_data is not None:
        logger.info(f"Loaded per-user config.yaml from: {user_config}")
        config = _deep_merge(config, user_data)
        sources.append(("per-user", str(user_config), True))
    else:
        logger.debug(f"No usable per-user config.yaml at {user_config}.")
        sources.append(("per-user", str(user_config), False))

    DIAGNOSTICS["config_sources"] = sources
    return config


# Configure logger first so that .env and config.yaml loading get logged.
config_logger()
# logger = logging.getLogger(__name__)

# Load the environment variables from the .env file (logs which file is used).
_log_dotenv()
load_dotenv()

# Per-user .env (e.g. the Confluence token saved via the GUI's onboarding
# dialog). A real, non-empty value already in the environment (e.g.
# CONFLUENCE_TOKEN exported on the cluster) still wins, but a per-user value
# fills an unset *or empty* variable -- plain load_dotenv(override=False) would
# treat an empty `CONFLUENCE_TOKEN=` as "already set" and never apply the saved
# token. Secrets live here -- never in config.yaml or logs.
_user_env = _user_config_dir() / ".env"
if _user_env.exists():
    for _k, _v in dotenv_values(_user_env).items():
        if _v is not None and not os.environ.get(_k):
            os.environ[_k] = _v
    logger.info(f"Loaded per-user .env from: {_user_env}")

if os.getenv("SLURM_JOB_ID") is None:
    # keep ON_CLUSTER in sync; original assignment above also reflects this,
    # but .env may have just been loaded, so re-evaluate.
    ON_CLUSTER = False
else:
    ON_CLUSTER = True

CONFIG = load_config()


def reload_config() -> dict:
    """Re-read the config.yaml layers and update ``CONFIG`` in place.

    The dict is mutated (cleared + updated) rather than rebound, so modules
    that imported ``CONFIG`` by reference see the new values without a restart
    -- e.g. after the GUI writes a per-user ``config.yaml`` from its Getting
    Started dialog. Returns the updated ``CONFIG``.
    """
    new = load_config()
    CONFIG.clear()
    CONFIG.update(new)
    return CONFIG


if __name__ == "__main__":
    # This is just to use the classes and not get PEP errors.
    # This is not expected to do anything meaningful.
    wr = WorkflowRunner()
    logger.debug(wr)
    awr = AggregationWorkflowRunner()
    logger.debug(awr)
    logger.debug(standard_singledataset_workflows.minimal())
    logger.debug(standard_aggregation_workflows)
