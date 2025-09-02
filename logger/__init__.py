from .custom_logger import CustomLogger

# Keep one global logger instance
_GLOBAL_LOGGER = None

def get_logger():
    global _GLOBAL_LOGGER
    if _GLOBAL_LOGGER is None:
        _GLOBAL_LOGGER = CustomLogger().get_logger()
    return _GLOBAL_LOGGER

# Expose at package level
GLOBAL_LOGGER = get_logger()
