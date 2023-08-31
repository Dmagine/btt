import logging


class A:
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3


def tmp_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("./tmp.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("test info")
    logger.debug("test debug")
    logger.error("test error")
    logger.warning("test warning")
    logger.critical("test critical")
    logger.exception("test exception")