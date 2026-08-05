import logging


def setup_logger(name=None, level=logging.INFO):
    """
    Sets up and returns a logger instance.
    :param name: Optional name for the logger.
    :param level: Logging level (default: logging.INFO).
    :return: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
