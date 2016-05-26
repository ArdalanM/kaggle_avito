# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief: configs
"""
from lib import os_utils

# ------------------------ PATH ------------------------
ROOT_FOLDER = "../"
CODE_FOLDER = "%s/code"%ROOT_FOLDER

DATA_FOLDER = "%s/data"%ROOT_FOLDER
PICKLE_DATA_FOLDER = "%s/0_pickled_data"%DATA_FOLDER

FEATURES_FOLDER = "%s/2_features"%DATA_FOLDER
WORD2VEC_MODEL_DIR = "%s/3_w2v"%DATA_FOLDER

# ---------------------- CREATE PATH --------------------
DIRS = []
DIRS += [CODE_FOLDER]
DIRS += [DATA_FOLDER, PICKLE_DATA_FOLDER]
DIRS += [FEATURES_FOLDER]
DIRS += [WORD2VEC_MODEL_DIR]

os_utils._create_dirs(DIRS)