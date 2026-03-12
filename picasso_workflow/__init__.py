from dotenv import load_dotenv

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

# Load the environment variables from the .env file
load_dotenv()

if os.getenv("SLURM_JOB_ID") is None:
    ON_CLUSTER = False
else:
    ON_CLUSTER = True


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
        config = _load_yaml(pkg_config)
    except (FileNotFoundError, TypeError):
        template = importlib.resources.files("picasso_workflow").joinpath(
            "config_template.yaml"
        )
        config = _load_yaml(template)

    # 2. Site-wide admin config (optional)
    site_config = _site_config_path()
    if site_config.exists():
        config = _deep_merge(config, _load_yaml(site_config))

    # 3. Per-user config (optional)
    user_config = Path.home() / ".config" / "picasso_workflow" / "config.yaml"
    if user_config.exists():
        config = _deep_merge(config, _load_yaml(user_config))

    return config


config_logger()
# logger = logging.getLogger(__name__)

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
