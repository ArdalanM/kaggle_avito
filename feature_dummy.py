# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: distance features

"""

import os
import re
import sys
import string

import numpy as np
import pandas as pd

import config
from lib import dist_utils, ngram_utils, nlp_utils, np_utils
from lib import logging_utils, time_utils, pkl_utils

def do_corr_and_save(x,y, feat_name=""):
    corr = np_utils._corr(x[:config.TRAIN_SIZE], y[:config.TRAIN_SIZE])
    print(corr)
    logger.info("{}: corr = {}".format(feat_name,corr))
    pkl_utils._save(os.path.join(config.FEATURES_FOLDER,
                                 feat_name + config.FEAT_FILE_SUFFIX), x)


df = pkl_utils._load(config.ALL_DATA_CLEANED)
y = df['isDuplicate'].values
logname = "generate_feature_dummy_%s.log" % time_utils._timestamp()
logger = logging_utils._get_logger(config.LOG_FOLDER, logname)


#Feature Nb ad
dic_itemID_1 = df['itemID_1'].value_counts().to_dict()
dic_itemID_2 = df['itemID_2'].value_counts().to_dict()
feat1 = df['itemID_1'].apply(lambda r: dic_itemID_1[r] if r in dic_itemID_1 else 0)
feat2 = df['itemID_2'].apply(lambda r: dic_itemID_2[r] if r in dic_itemID_2 else 0)

do_corr_and_save(feat1.values,y, feat_name="ctn_itemID_1")
do_corr_and_save(feat2.values,y, feat_name="ctn_itemID_2")







['attrsJSON_1', 'attrsJSON_2', 'categoryID_1',
 'categoryID_2', 'description_1', 'description_2',
 'generationMethod', 'id', 'images_array_1', 'images_array_2',
 'isDuplicate', 'itemID_1', 'itemID_2', 'lat_1', 'lat_2',
 'locationID_1', 'locationID_2', 'lon_1', 'lon_2',
 'metroID_1', 'metroID_2', 'parentCategoryID_1',
 'parentCategoryID_2', 'price_1', 'price_2',
 'regionID_1', 'regionID_2', 'title_1', 'title_2']



