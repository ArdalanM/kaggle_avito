# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for logging
"""

import os
import logging
import logging.handlers


def _get_logger(logdir, logname, loglevel=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(os.path.join(logdir, logname))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


    # def _get_logger(logdir, logname, loglevel=logging.INFO):
    #     fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    #     formatter = logging.Formatter(fmt)
    #
    #     handler = logging.handlers.RotatingFileHandler(
    #                     filename=os.path.join(logdir, logname),
    #                     maxBytes=10*1024*1024,
    #                     backupCount=10)
    #     handler.setFormatter(formatter)
    #
    #     logger = logging.getLogger("")
    #     logger.addHandler(handler)
    #     logger.setLevel(loglevel)
    #     return logger
