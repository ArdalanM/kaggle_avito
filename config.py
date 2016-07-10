# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief: configs
"""
import os
from utils import os_utils

# ------------------------ PATH ------------------------
ROOT_FOLDER = os.path.abspath("..")
CODE_FOLDER = "%s/code" % ROOT_FOLDER

DATA_FOLDER = "%s/data" % ROOT_FOLDER
PICKLE_DATA_FOLDER = "%s/0_pickled_data" % DATA_FOLDER
DICLOG_FOLDER = "%s/1_diclogs" % DATA_FOLDER
FEATURES_FOLDER = "%s/2_features" % DATA_FOLDER
WORD2VEC_MODEL_DIR = "%s/3_w2v" % DATA_FOLDER
LOG_FOLDER = "%s/logs" % DATA_FOLDER


TRAIN_RAW = "{}/train_raw.pkl".format(PICKLE_DATA_FOLDER)
TEST_RAW = "{}/test_raw.pkl".format(PICKLE_DATA_FOLDER)
ALL_DATA_RAW = "{}/all_raw.pkl".format(PICKLE_DATA_FOLDER)
ALL_DATA_CLEANED = "{}/all_cleaned_dhash.pkl".format(PICKLE_DATA_FOLDER)

ITEMINFO_RAW = "{}/ItemInfo_raw.pkl".format(PICKLE_DATA_FOLDER)
ITEMINFO_CLEANED = "{}/ItemInfo_cleaned.pkl".format(PICKLE_DATA_FOLDER)

# ------------------------ W2V D2V ------------------------
W2V_ALPHA = 0.025
W2V_LEARNING_RATE_DECAY = 0.5
W2V_N_EPOCH = 5
W2V_MIN_COUNT = 5
W2V_DIM = 100
W2V_WINDOW = 5
W2V_WORKERS = 4


LABEL_COL = 'isDuplicate'
TRAIN_SIZE = 2991396
FEAT_FILE_SUFFIX = ".pkl"
# missing value
MISSING_VALUE_STRING = "MISSINGVALUE"
MISSING_VALUE_NUMERIC = -1.

CPU = "cpu"
GPU0 = "gpu0"
GPU1 = "gpu1"
NTHREAD = 6
SEED = 2016

# ---------------------- CREATE PATH --------------------
DIRS = []
DIRS += [CODE_FOLDER]
DIRS += [DATA_FOLDER, PICKLE_DATA_FOLDER, LOG_FOLDER]
DIRS += [FEATURES_FOLDER]
DIRS += [WORD2VEC_MODEL_DIR]
DIRS += [DICLOG_FOLDER]

os_utils._create_dirs(DIRS)
