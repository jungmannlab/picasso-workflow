from dotenv import load_dotenv
import logging
from logging import handlers


# Load the environment variables from the .env file
load_dotenv()


# configure logger
def config_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(funcName)s | %(levelname)s -> %(message)s"
    )
    file_handler = handlers.RotatingFileHandler(
        "picasso-workflow.log", maxBytes=1e6, backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)


config_logger()
logger = logging.getLogger(__name__)
