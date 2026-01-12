import logging
import os


def get_configured_logger(
    name: str, log_file: str | None = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Tworzy i zwraca skonfigurowany logger.

    :param name: Nazwa loggera.
    :param log_file: Opcjonalna ścieżka do pliku logu. Jeśli None, logi będą wyświetlane w konsoli.
    :param level: Poziom logowania (np. logging.DEBUG, logging.INFO).
    :return: Skonfigurowany logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
