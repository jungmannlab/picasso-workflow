from dotenv import load_dotenv, find_dotenv

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
from logging import handlers
from picasso_workflow.workflow import WorkflowRunner, AggregationWorkflowRunner
from picasso_workflow import standard_singledataset_workflows
from picasso_workflow import standard_aggregation_workflows


# configure logger
def config_logger():
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     "%(asctime)s | %(name)s | %(funcName)s | %(levelname)s -> %(message)s"
    # )
    os.makedirs("logs", exist_ok=True)
    job_id = os.getenv("SLURM_JOB_ID")  # Get the job ID from the environment
    rank_id = os.getenv("SLURM_PROCID")
    logfile = f"logs/picasso-workflow-job{job_id}-rank{rank_id}.log"
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
        level="DEBUG"
    )
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss:SSS} | PID:{process} | {name} | {function} | {level} -> {message}",
        level="ERROR",
    )


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


def _load_yaml(path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _log_dotenv():
    """Locate the .env file actually used by load_dotenv() and log it.

    ``load_dotenv()`` (no args) walks up from the caller's ``__file__``
    directory looking for a ``.env``; ``find_dotenv()`` with default
    ``usecwd=False`` reproduces that, since this function is called from
    the same module body.
    """
    dotenv_path = find_dotenv()
    if dotenv_path:
        logger.info(f"Loading .env from: {dotenv_path}")
        try:
            with open(dotenv_path, "r") as f:
                content = f.read()
            logger.info(f".env content:\n{content}")
        except OSError as exc:
            logger.warning(f"Could not read .env at {dotenv_path}: {exc}")
    else:
        logger.info(
            "No .env file found by python-dotenv "
            f"(searched upward from {Path(__file__).parent})."
        )
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
    # 1. Package default
    try:
        pkg_config = importlib.resources.files("picasso_workflow").joinpath(
            "config.yaml"
        )
        pkg_data = _load_yaml(pkg_config)
        logger.info(f"Loaded bundled config.yaml from: {pkg_config}")
        logger.info(
            "Bundled config.yaml content:\n"
            f"{yaml.safe_dump(pkg_data, sort_keys=False)}"
        )
        config = pkg_data
    except (FileNotFoundError, TypeError) as exc:
        logger.warning(f"No bundled config.yaml available ({exc!r}).")
        # template = importlib.resources.files("picasso_workflow").joinpath(
        #     "config_template.yaml"
        # )
        # config = _load_yaml(template)
        config = {}

    # 2. Site-wide admin config (optional)
    site_config = _site_config_path()
    if site_config.exists():
        site_data = _load_yaml(site_config)
        logger.info(f"Loaded site-wide config.yaml from: {site_config}")
        logger.info(
            "Site-wide config.yaml content:\n"
            f"{yaml.safe_dump(site_data, sort_keys=False)}"
        )
        config = _deep_merge(config, site_data)
    else:
        logger.debug(
            f"No site-wide config.yaml at {site_config} (skipped)."
        )

    # 3. Per-user config (optional)
    user_config = Path.home() / ".config" / "picasso_workflow" / "config.yaml"
    if user_config.exists():
        user_data = _load_yaml(user_config)
        logger.info(f"Loaded per-user config.yaml from: {user_config}")
        logger.info(
            "Per-user config.yaml content:\n"
            f"{yaml.safe_dump(user_data, sort_keys=False)}"
        )
        config = _deep_merge(config, user_data)
    else:
        logger.debug(
            f"No per-user config.yaml at {user_config} (skipped)."
        )

    return config


# Configure logger first so that .env and config.yaml loading get logged.
config_logger()
# logger = logging.getLogger(__name__)

# Load the environment variables from the .env file (logs which file is used).
_log_dotenv()
load_dotenv()

if os.getenv("SLURM_JOB_ID") is None:
    # keep ON_CLUSTER in sync; original assignment above also reflects this,
    # but .env may have just been loaded, so re-evaluate.
    ON_CLUSTER = False
else:
    ON_CLUSTER = True

CONFIG = load_config()


if __name__ == "__main__":
    # This is just to use the classes and not get PEP errors.
    # This is not expected to do anything meaningful.
    wr = WorkflowRunner()
    logger.debug(wr)
    awr = AggregationWorkflowRunner()
    logger.debug(awr)
    logger.debug(standard_singledataset_workflows.minimal())
    logger.debug(standard_aggregation_workflows)
