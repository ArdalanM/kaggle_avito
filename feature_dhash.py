# -*- coding: utf-8 -*-
"""
@author: Ardalan Mehrani
@brief: dhash features
"""
import os
import config
import numpy as np

from utils import pkl_utils, np_utils

pd_data = pkl_utils._load(config.ALL_DATA_CLEANED)

y = pd_data['isDuplicate'].values
dhash_1 = pd_data['images_dhash_x'].values
dhash_2 = pd_data['images_dhash_y'].values
del pd_data


# Nb hash in common
nb_in_common = np.zeros(len(dhash_1))
nb_in_common_ration_total = np.zeros(len(dhash_1))

for i, (string_1, string_2) in enumerate(zip(dhash_1, dhash_2)):
    array_1 = string_1.split(", ")
    array_2 = string_2.split(", ")

    nb_in_common[i] = len(set(array_1).intersection(set(array_2)))
    nb_in_common_ration_total[i] = nb_in_common[i] / (len(array_1)+len(array_2))

    if i % 10000 == 0:
        print("{} lines processed".format(i))


filename = "Dhash_nb_in_common_{:0.3f}{}".format(
    np_utils._corr(nb_in_common[:config.TRAIN_SIZE], y[:config.TRAIN_SIZE]),
    config.FEAT_FILE_SUFFIX)

pkl_utils._save(os.path.join(config.FEATURES_FOLDER, filename), nb_in_common)

filename = "Dhash_nb_in_common_ratio_total_{:0.3f}{}".format(
    np_utils._corr(nb_in_common_ration_total[:config.TRAIN_SIZE], y[:config.TRAIN_SIZE]),
    config.FEAT_FILE_SUFFIX)

pkl_utils._save(os.path.join(config.FEATURES_FOLDER, filename), nb_in_common_ration_total)





