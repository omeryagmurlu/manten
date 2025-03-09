def get_logger(*args, **kwargs):
    try:
        from accelerate.logging import get_logger as _get_logger

        return _get_logger(*args, **kwargs)
    except ImportError:
        from logging import getLogger

        return getLogger(*args, **kwargs)


def setup_custom_logging(log_file):
    import logging
    import sys

    import colorlog

    # Setup logging to output to both console and file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"
        )
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
