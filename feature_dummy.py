# -*- coding: utf-8 -*-
"""
@author: Ardalan Mehrani
@brief: dummy features

"""

import os
import config

import numpy as np
from utils import np_utils, logging_utils, time_utils, pkl_utils

df = pkl_utils._load(config.ALL_DATA_CLEANED)
y = df['isDuplicate'].values

logger = logging_utils._get_logger(config.LOG_FOLDER, "generate_feature_dummy_%s.log" % time_utils._timestamp())

# Feature Nb ad
dic_itemID_1 = df['itemID_1'].value_counts().to_dict()
dic_itemID_2 = df['itemID_2'].value_counts().to_dict()
feat1 = df['itemID_1'].apply(lambda r: dic_itemID_1[r] if r in dic_itemID_1 else 0)
feat2 = df['itemID_2'].apply(lambda r: dic_itemID_2[r] if r in dic_itemID_2 else 0)

filename = "Count_itemID_1_{:0.3f}.pkl".format(np_utils._corr(feat1[:config.TRAIN_SIZE].values, y[:config.TRAIN_SIZE]))
print(filename)
pkl_utils._save(os.path.join(config.FEATURES_FOLDER, filename), feat1)

filename = "Count_itemID_2_{:0.3f}.pkl".format(np_utils._corr(feat2[:config.TRAIN_SIZE].values, y[:config.TRAIN_SIZE]))
print(filename)
pkl_utils._save(os.path.join(config.FEATURES_FOLDER, filename), feat2)

feat = 1 * (df['locationID_1'] == df['locationID_2']).values
filename = "locationID_same_{:0.3f}.pkl".format(np_utils._corr(feat[:config.TRAIN_SIZE], y[:config.TRAIN_SIZE]))
print(filename)
pkl_utils._save(os.path.join(config.FEATURES_FOLDER, filename), feat)

feat = 1 * (df['metroID_1'] == df['metroID_2']).values
filename = "metroID_same_{:0.3f}.pkl".format(np_utils._corr(feat[:config.TRAIN_SIZE], y[:config.TRAIN_SIZE]))
print(filename)
pkl_utils._save(os.path.join(config.FEATURES_FOLDER, filename), feat)

feat = 1 * (df['price_1'] == df['price_2']).values
filename = "price_same_{:0.3f}.pkl".format(np_utils._corr(feat[:config.TRAIN_SIZE], y[:config.TRAIN_SIZE]))
print(filename)
pkl_utils._save(os.path.join(config.FEATURES_FOLDER, filename), feat)

feat = 1 * (df['regionID_1'] == df['regionID_2']).values
filename = "regionID_same_{:0.3f}.pkl".format(np_utils._corr(feat[:config.TRAIN_SIZE], y[:config.TRAIN_SIZE]))
print(filename)
pkl_utils._save(os.path.join(config.FEATURES_FOLDER, filename), feat)

feat = (np.sqrt((df['lat_1'] - df['lat_2']) ** 2 + (df['lon_1'] - df['lon_2']) ** 2)).values
filename = "Sqrt_geodist_{:0.3f}.pkl".format(np_utils._corr(feat[:config.TRAIN_SIZE], y[:config.TRAIN_SIZE]))
print(filename)
pkl_utils._save(os.path.join(config.FEATURES_FOLDER, filename), feat)

# convert decimal degrees to radians
lon1 = df["lon_1"].apply(np.deg2rad).values
lat1 = df["lat_1"].apply(np.deg2rad).values
lon2 = df["lon_2"].apply(np.deg2rad).values
lat2 = df["lat_2"].apply(np.deg2rad).values

# haversine formula
dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arcsin(np.sqrt(a))
r = 6371  # Radius of earth in kilometers. Use 3956 for miles
feat = c * r
filename = "Haversine_geodist_{:0.3f}.pkl".format(np_utils._corr(feat[:config.TRAIN_SIZE], y[:config.TRAIN_SIZE]))
print(filename)
pkl_utils._save(os.path.join(config.FEATURES_FOLDER, filename), feat)





from sklearn.cluster.k_means_ import KMeans


pd_ItemInfo = pkl_utils._load(config.ITEMINFO_CLEANED)
X = np.array(pd_ItemInfo[['lat', 'lon']])

clf = KMeans(n_clusters=10, n_jobs=12)
clf.fit(X)
c1 = clf.predict(np.array(df[['lat_1', 'lon_1']]))
c2 = clf.predict(np.array(df[['lat_2', 'lon_2']]))
feat = 1 * (c1 == c2)
filename = "Kmean10_geodist_{:0.3f}.pkl".format(np_utils._corr(feat[:config.TRAIN_SIZE], y[:config.TRAIN_SIZE]))
print(filename)
pkl_utils._save(os.path.join(config.FEATURES_FOLDER, filename), feat)


clf = KMeans(n_clusters=100, n_jobs=12)
clf.fit(X)
c1 = clf.predict(np.array(df[['lat_1', 'lon_1']]))
c2 = clf.predict(np.array(df[['lat_2', 'lon_2']]))
feat = 1 * (c1 == c2)
filename = "Kmean100_geodist_{:0.3f}.pkl".format(np_utils._corr(feat[:config.TRAIN_SIZE], y[:config.TRAIN_SIZE]))
print(filename)
pkl_utils._save(os.path.join(config.FEATURES_FOLDER, filename), feat)





['attrsJSON_1', 'attrsJSON_2', 'categoryID_1',
 'categoryID_2', 'description_1', 'description_2',
 'generationMethod', 'id', 'images_array_1', 'images_array_2',
 'isDuplicate', 'itemID_1', 'itemID_2', 'lat_1', 'lat_2',
 'locationID_1', 'locationID_2', 'lon_1', 'lon_2',
 'metroID_1', 'metroID_2', 'parentCategoryID_1',
 'parentCategoryID_2', 'price_1', 'price_2',
 'regionID_1', 'regionID_2', 'title_1', 'title_2']


import json

df['attrsJSON_1'] = df['attrsJSON_1'].fillna("")
df['attrsJSON_2'] = df['attrsJSON_2'].fillna("")


arrays_1, arrays_2 = df['attrsJSON_1'].values, df['attrsJSON_2'].values




feat_nb_key_common = np.zeros(len(arrays_1))
feat_nb_value_common = np.zeros(len(arrays_1))
feat_nb_keyvalues_common = np.zeros(len(arrays_1))


for i, (arr_1, arr_2) in enumerate(zip(arrays_1, arrays_2)):

    if (len(arr_1) or len(arr_2)) == 0:
        feat_nb_key_common[i] = -1
        feat_nb_value_common[i] = -1
        feat_nb_keyvalues_common[i] = -1
    else:
        j1 = json.loads(arr_1)
        j2 = json.loads(arr_2)

        tot_key = len(j1.keys()) + len(j2.keys())

        feat_nb_key_common[i] = len(set(j1.keys()).intersection(j2.keys())) / tot_key
        feat_nb_value_common[i] = len(set(j1.values()).intersection(j2.values())) / tot_key
        feat_nb_keyvalues_common[i] = len(set(j1.items()).intersection(j2.items())) / tot_key


    if i % 10000 == 0:
        print(i)
        # break


f1 = "Json_nbkey_{:0.3f}.pkl".format(np_utils._corr(feat_nb_key_common[:config.TRAIN_SIZE], y[:config.TRAIN_SIZE]))
f2 = "Json_nbvalue_{:0.3f}.pkl".format(np_utils._corr(feat_nb_value_common[:config.TRAIN_SIZE], y[:config.TRAIN_SIZE]))
f3 = "Json_nbkeyvalue_{:0.3f}.pkl".format(np_utils._corr(feat_nb_keyvalues_common[:config.TRAIN_SIZE], y[:config.TRAIN_SIZE]))

print(f1, f2, f3)
pkl_utils._save(os.path.join(config.FEATURES_FOLDER, f1), feat_nb_key_common)
pkl_utils._save(os.path.join(config.FEATURES_FOLDER, f2), feat_nb_value_common)
pkl_utils._save(os.path.join(config.FEATURES_FOLDER, f3), feat_nb_keyvalues_common)
