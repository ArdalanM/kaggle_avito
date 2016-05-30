"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief: This script will:

        1: load:
            - train_merged-part1.h
            - train_merged-part2.h
            - test_merged.h
        2: create train and test similarity features. Each created features is a pandas DataFrame
        3: Dump those features in a binary file
"""

FOLDER = "/home/ardalan/Documents/kaggle/avito/"

import numpy as np
import pandas as pd


pdtrain1 = pd.read_hdf(FOLDER + "data/train_merged-part1.h")
pdtrain2 = pd.read_hdf(FOLDER + "data/train_merged-part2.h")
pdtrain = pdtrain1.append(pdtrain2)
pdtest = pd.read_hdf(FOLDER + "data/test_merged.h")
pd_data = pdtrain.append(pdtest)

del pdtrain1
del pdtrain2
del pdtest


pd_data['title_1'].fillna("", inplace=True)
pd_data['title_2'].fillna("", inplace=True)
pd_data['description_1'].fillna("", inplace=True)
pd_data['description_2'].fillna("", inplace=True)
pd_data['price_1'].fillna(999999999, inplace=True)
pd_data['price_2'].fillna(999999999, inplace=True)
pd_data['attrsJSON_1'].fillna("", inplace=True)
pd_data['attrsJSON_2'].fillna("", inplace=True)
pd_data['images_array_1'].fillna("", inplace=True)
pd_data['images_array_2'].fillna("", inplace=True)

# FI = [('categoryID_1', 797), ('len_attrsJSON_2', 796), ('len_attrsJSON_1', 789), ('price_2', 654), ('price_1', 552),
#       ('parentCategoryID_2', 398), ('lat_1', 382), ('lon_2', 342), ('lat_2', 328), ('lon_1', 323),
#       ('len_description_1', 305), ('len_description_2', 288), ('len_title_1', 234), ('locationID_2', 212),
#       ('len_title_2', 206), ('locationID_1', 188), ('categoryID_2', 159), ('price_same', 148), ('metroID_1', 122),
#       ('regionID_same', 122), ('metroID_2', 113), ('parentCategoryID_1', 103), ('locationID_same', 94),
#       ('regionID_2', 72), ('regionID_1', 50), ('lon_same', 50), ('lat_same', 42), ('metroID_same', 30)]


# ad specific features
pd_data['attrsJSON_1_len'] = pd_data['attrsJSON_1'].apply(len)
pd_data['attrsJSON_2_len'] = pd_data['attrsJSON_2'].apply(len)
pd_data['description_1_len'] = pd_data['description_1'].apply(len)
pd_data['description_2_len'] = pd_data['description_2'].apply(len)
pd_data['title_1_len'] = pd_data['title_1'].apply(len)
pd_data['title_2_len'] = pd_data['title_2'].apply(len)


# title
string_feat = StringFeatures(pd_data)
pd_data['title_levenshtein'] = string_feat.add_levenshtein('title_1', 'title_2')
pd_data['title_Dlevenshtein'] = string_feat.add_damerau_levenshtein('title_1', 'title_2')
pd_data['title_jaro'] = string_feat.add_jaro('title_1', 'title_2')
pd_data['title_jarowinkler'] = string_feat.add_jaro_winkler('title_1', 'title_2')
pd_data['title_hamming'] = string_feat.add_hamming('title_1', 'title_2')
pd_data['title_difflen'] = np.abs(pd_data['title_2'].apply(len) - pd_data['title_1'].apply(len))



# description
string_feat = StringFeatures(pd_data)
pd_data['description_levenshtein'] = string_feat.add_levenshtein('description_1', 'description_2')
pd_data['description_Dlevenshtein'] = string_feat.add_damerau_levenshtein('description_1', 'description_2')
pd_data['description_jaro'] = string_feat.add_jaro('description_1', 'description_2')
pd_data['description_jarowinkler'] = string_feat.add_jaro_winkler('description_1', 'description_2')
pd_data['description_hamming'] = string_feat.add_hamming('description_1', 'description_2')
pd_data['description_difflen'] = np.abs(pd_data['description_2'].apply(len) - pd_data['description_1'].apply(len))


# price
pd_data['price_diff_abs'] = np.abs(pd_data['price_1'] - pd_data['price_2'])
pd_data['price_diff_square'] = np.square(pd_data['price_1'] - pd_data['price_2'])

# attrJson
pd_data['attrsJSON_difflen'] = np.abs(pd_data['attrsJSON_1'].apply(len) - pd_data['attrsJSON_2'].apply(len))


# images_array
pd_data['images_diff_number'] = np.abs(pd_data['images_array_1'].apply(lambda x: len(x.split(','))) -
                                       pd_data['images_array_2'].apply(lambda x: len(x.split(','))))

# geographic features
geo = GeoFeatures(pd_data)
pd_data['is_metroID_same'] = 1 * (pd_data['metroID_1'] == pd_data['metroID_2'])
pd_data['is_locationID_same'] = 1 * (pd_data['locationID_1'] == pd_data['locationID_2'])
pd_data['is_regionID_same'] = 1 * (pd_data['regionID_1'] == pd_data['regionID_2'])
pd_data['haversine'] = geo.add_haversine('lon_1', 'lat_1', 'lon_2', 'lat_2')
pd_data['haversine'] = (pd_data['haversine'] - pd_data['haversine'].mean()) / pd_data['haversine'].std()

# CATEGORY, PARENT CATEGORY
pd_data['is_categoryID_same'] = 1 * (pd_data['categoryID_1'] == pd_data['categoryID_2'])
pd_data['is_parentCategoryID_same'] = 1 * (pd_data['parentCategoryID_1'] == pd_data['parentCategoryID_2'])


pd_data.drop(['attrsJSON_1', 'attrsJSON_1',
              'description_1', 'description_2',
              'images_array_1', 'images_array_2',
              'itemID_1', 'itemID_2',
              'title_1', 'title_2'])

# D1 = create_D1(pd_data)
# D1.to_hdf(FOLDER + "data/D1_with_adspecific_features_22may.p", 'w')



# # NI
# temp = train['titulaires'].apply(lambda st : st.split(','))
# labos = set([l for j in temp for l in j])
# for lab in labos:
#     train[lab] = train['titulaires'].apply(lambda x : 1 if lab in x else 0)
#     test[lab] = test['titulaires'].apply(lambda x : 1 if lab in x else 0)




