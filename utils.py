from logging import getLogger, DEBUG, Formatter, StreamHandler


def setup_logger():
    """Configures logger with preset format."""

    _logger = getLogger()
    _logger.setLevel(DEBUG)

    formatter = Formatter(
        '%(asctime)s %(processName)s %(threadName)s - %(levelname)s - %(message)s'
    )

    console_handler = StreamHandler()
    console_handler.setFormatter(formatter)

    _logger.addHandler(console_handler)

    return _logger
