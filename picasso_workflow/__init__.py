from dotenv import load_dotenv

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
    )
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss:SSS} | PID:{process} | {name} | {function} | {level} -> {message}",
        level="ERROR",
    )


def load_config():
    """Load the picasso-workflow configuration yaml file"""
    # 1. User config path
    user_config = Path.home() / ".config" / "picasso_workflow" / "config.yaml"
    if user_config.exists():
        with open(user_config, "r") as f:
            return yaml.safe_load(f)
    # 2. Fallback to package default
    default_config = importlib.resources.files("picasso_workflow").joinpath(
        "config.yaml"
    )
    with open(default_config, "r") as f:
        return yaml.safe_load(f)


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
