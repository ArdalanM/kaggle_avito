# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief: configs
"""
from lib import os_utils

# ------------------------ PATH ------------------------
ROOT_FOLDER = ".."
CODE_FOLDER = "%s/code"%ROOT_FOLDER

DATA_FOLDER = "%s/data"%ROOT_FOLDER
PICKLE_DATA_FOLDER = "%s/0_pickled_data"%DATA_FOLDER

FEATURES_FOLDER = "%s/2_features"%DATA_FOLDER
WORD2VEC_MODEL_DIR = "%s/3_w2v"%DATA_FOLDER

TRAIN_RAW = "{}/train_raw.pkl".format(PICKLE_DATA_FOLDER)
TEST_RAW = "{}/test_raw.pkl".format(PICKLE_DATA_FOLDER)
ALL_DATA_RAW = "{}/all_raw.pkl".format(PICKLE_DATA_FOLDER)



# word2vec/doc2vec
W2V_ALPHA = 0.025
W2V_LEARNING_RATE_DECAY = 0.5
W2V_N_EPOCH = 5
W2V_MIN_COUNT = 3
W2V_DIM = 100
W2V_WINDOW = 5
W2V_WORKERS = 6

# ---------------------- CREATE PATH --------------------
DIRS = []
DIRS += [CODE_FOLDER]
DIRS += [DATA_FOLDER, PICKLE_DATA_FOLDER]
DIRS += [FEATURES_FOLDER]
DIRS += [WORD2VEC_MODEL_DIR]

os_utils._create_dirs(DIRS)