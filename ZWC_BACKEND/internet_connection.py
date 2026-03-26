import socket
from loguru import logger

def check_internet_connection():
    logger.info("Checking Internet Connnection")
    try:
        socket.create_connection(("8.8.8.8",53), 5)
        logger.info("Internet connection")
        return True
    except OSError:
        pass
    logger.warning("No Internet Connection")
    return False