import platform
from loguru import logger

os_name = platform.system()
logger.success(os_name)
def check_os():
    if os_name == "Windows":
        logger.info("Windows System detected")
        return "Windows"
    elif os_name == "Linux":
        logger.info("Linux system detected")
        return "Linux"
    elif os_name == "Darwin":
        logger.info("MacOs system detected")
        return "MacOs"
    