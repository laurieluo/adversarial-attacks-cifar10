import logging
import sys

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to log messages using ANSI escape codes.
    """
    # Define color codes
    GREY = "\x1b[38;5;240m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    # Define log format for each level
    FORMATS = {
        logging.DEBUG: f"{GREY}%(levelname)-8s{RESET} %(message)s",
        logging.INFO: f"{GREEN}%(levelname)-8s{RESET} %(message)s",
        logging.WARNING: f"{YELLOW}%(levelname)-8s{RESET} %(message)s",
        logging.ERROR: f"{RED}%(levelname)-8s{RESET} %(message)s",
        logging.CRITICAL: f"{BOLD_RED}%(levelname)-8s{RESET} %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logging():
    """
    Configures the root logger to use the colored formatter.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Check if handlers are already added to avoid duplication
    if not logger.handlers:
        # Create console handler and set formatter
        # We use sys.stdout to avoid conflicts with tqdm which uses stderr
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)
