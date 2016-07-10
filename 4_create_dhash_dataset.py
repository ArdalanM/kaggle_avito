# -*- coding: utf-8 -*-
"""
@author:
@brief: base class for feature generation

"""

import os, re, sys, glob
import config

import numpy as np
import pandas as pd

from utils import pkl_utils

input_folder = os.path.join(config.DATA_FOLDER, "hash_images")


filenames = glob.glob1(input_folder, "*.csv")

pd_images = pd.DataFrame()
for filename in filenames:
    print("Loading: {}".format(filename))
    filepath = os.path.join(input_folder, filename)
    pd_images = pd_images.append(pd.read_csv(filepath))


# pd_images['image_id_r'] = pd_images['image_id'].apply(lambda r: int("".join([c for c in r if c.isdigit()])))
pd_images['image_id_r'] = pd_images['image_id'].apply(lambda r: int(r.split("/")[-1]))
dic_id_hash = dict(pd_images[['image_id_r', 'image_hash']].values)



pd_data = pkl_utils._load(config.ITEMINFO_CLEANED)
pd_data['images_array'].fillna("", inplace=1)

l_images_array = pd_data['images_array'].values


l_hash_array = []
for i, image_array in enumerate(l_images_array):

    temp = []
    for image_id in image_array.split(", "):
        if len(image_id) > 0:
            image_id = int(image_id)
            if image_id in dic_id_hash:
                temp.append(dic_id_hash[image_id])
            else:
                print(image_id)
    l_hash_array.append(", ".join(temp))


    if i % 10000 == 0:
        print("Processed {} lines".format(i))


pd_data['images_dhash'] = l_hash_array

filepath = os.path.join(config.PICKLE_DATA_FOLDER, "ItemInfo_cleaned_dhash.pkl")

# pkl_utils._save(filepath, pd_data)

#Creating cleaned dataset
ItemPair_train = os.path.join(config.DATA_FOLDER, "ItemPairs_train.csv.zip")
ItemPair_test = os.path.join(config.DATA_FOLDER, "ItemPairs_test.csv.zip")

pdItemPair_train = pd.read_csv(ItemPair_train, compression='zip', encoding='utf-8')
pdItemPair_test = pd.read_csv(ItemPair_test, compression='zip', encoding='utf-8')
pdItemPair = pdItemPair_train.append(pdItemPair_test)


ItemInfo_cleaned_dhash = os.path.join(config.PICKLE_DATA_FOLDER, "ItemInfo_cleaned_dhash.pkl")
df = pkl_utils._load(ItemInfo_cleaned_dhash)

pd_data = pd.merge(pdItemPair, df, how='left', left_on='itemID_1', right_on='itemID')
pd_data.drop(['itemID_1'], 1, inplace=True)
pd_data.rename(columns={
    'itemID': 'itemID_1',
    'categoryID': 'categoryID_1',
    'title': 'title_1',
    'description': 'description_1',
    'images_array': 'images_array_1',
    'attrsJSON': 'attrsJSON_1',
    'price': 'price_1',
    'locationID': 'locationID_1',
    'metroID': 'metroID_1',
    'lat': 'lat_1',
    'lon': 'lon_1',
    'parentCategoryID': 'parentCategoryID_1',
    'regionID': 'regionID_1'
}, inplace=1)
pd_data = pd.merge(pd_data, df, how='left', left_on='itemID_2', right_on='itemID')
pd_data.drop(['itemID_2'], 1, inplace=True)
pd_data.rename(columns={
    'itemID': 'itemID_2',
    'categoryID': 'categoryID_2',
    'title': 'title_2',
    'description': 'description_2',
    'images_array': 'images_array_2',
    'attrsJSON': 'attrsJSON_2',
    'price': 'price_2',
    'locationID': 'locationID_2',
    'metroID': 'metroID_2',
    'lat': 'lat_2',
    'lon': 'lon_2',
    'parentCategoryID': 'parentCategoryID_2',
    'regionID': 'regionID_2'
}, inplace=1)

pkl_utils._save(os.path.join(config.PICKLE_DATA_FOLDER, "all_cleaned_dhash.pkl"), pd_data)

