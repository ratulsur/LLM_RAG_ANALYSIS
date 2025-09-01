import os
import logging
from datetime import datetime


class CustomLogger:
    log_file_path = None  

    @staticmethod
    def configure_logger():
        logger = logging.getLogger()
        if logger.handlers:
            return CustomLogger.log_file_path  # Already configured

        os.makedirs("logs", exist_ok=True)
        CustomLogger.log_file_path = os.path.join(
            "logs", f"{datetime.now():%m_%d_%Y_%H_%M_%S}.log"
        )

        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        )

        file_handler = logging.FileHandler(CustomLogger.log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        logger.info(f"Logging initialized. File: {CustomLogger.log_file_path}")
        return CustomLogger.log_file_path

    @staticmethod
    def get_logger(name):
        return logging.getLogger(name)


# Test logger
if __name__ == "__main__":
    path = CustomLogger.configure_logger()
    log = CustomLogger.get_logger(__name__)
    log.info("Logger initialized in main.")
    print(f"Logs are being written to: {path}")
