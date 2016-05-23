__author__ = 'Ardalan'

FOLDER = "/home/ardalan/Documents/kaggle/avito/"

import numpy as np
import pandas as pd

from math import radians


def haversine(pddata, lon1='', lat1='', lon2='', lat2=''):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1 = pddata[lon1].apply(radians).values
    lat1 = pddata[lat1].apply(radians).values
    lon2 = pddata[lon2].apply(radians).values
    lat2 = pddata[lat2].apply(radians).values

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


pdtrain1 = pd.read_hdf(FOLDER + "data/train_merged-part1.h")
pdtrain2 = pd.read_hdf(FOLDER + "data/train_merged-part2.h")
pdtrain = pdtrain1.append(pdtrain2)

pdtest = pd.read_hdf(FOLDER + "data/test_merged.h")
pd_data = pdtrain.append(pdtest)

del pdtrain1
del pdtrain2
del pdtest



def create_D1(pd_data):

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

    # feature dataframe
    pd_features = pd.DataFrame()


# FI = [('categoryID_1', 797), ('len_attrsJSON_2', 796), ('len_attrsJSON_1', 789), ('price_2', 654), ('price_1', 552),
#       ('parentCategoryID_2', 398), ('lat_1', 382), ('lon_2', 342), ('lat_2', 328), ('lon_1', 323),
#       ('len_description_1', 305), ('len_description_2', 288), ('len_title_1', 234), ('locationID_2', 212),
#       ('len_title_2', 206), ('locationID_1', 188), ('categoryID_2', 159), ('price_same', 148), ('metroID_1', 122),
#       ('regionID_same', 122), ('metroID_2', 113), ('parentCategoryID_1', 103), ('locationID_same', 94),
#       ('regionID_2', 72), ('regionID_1', 50), ('lon_same', 50), ('lat_same', 42), ('metroID_same', 30)]


    # ad specific categories
    pd_features['categoryID_1'] = pd_data['categoryID_1']
    pd_features['categoryID_2'] = pd_data['categoryID_2']
    pd_features['parentCategoryID_1'] = pd_data['parentCategoryID_1']
    pd_features['parentCategoryID_2'] = pd_data['parentCategoryID_2']
    pd_features['price_1'] = pd_data['price_1']
    pd_features['price_2'] = pd_data['price_2']
    pd_features['parentCategoryID_1'] = pd_data['parentCategoryID_1']
    pd_features['parentCategoryID_2'] = pd_data['parentCategoryID_2']
    pd_features['regionID_1'] = pd_data['regionID_1']
    pd_features['regionID_2'] = pd_data['regionID_2']
    pd_features['attrsJSON_1_len'] = pd_data['attrsJSON_1'].apply(len)
    pd_features['attrsJSON_2_len'] = pd_data['attrsJSON_2'].apply(len)



    # title
    pd_features['title_difflen'] = np.abs(pd_data['title_2'].apply(len) - pd_data['title_1'].apply(len))
    pd_features['title_levenshtein'] = pd.read_hdf(FOLDER + "data/Feat_train_title_leven.h")
    pd_features['title_Dlevenshtein'] = pd.read_hdf(FOLDER + "data/Feat_train_title_demarauleven.h")
    pd_features['title_jaro'] = pd.read_hdf(FOLDER + "data/Feat_train_title_jaro.h")
    pd_features['title_jarowinkler'] = pd.read_hdf(FOLDER + "data/Feat_train_title_jarowinkler.h")
    pd_features['title_hamming'] = pd.read_hdf(FOLDER + "data/Feat_train_title_hamming.h")

    # description
    pd_features['description_difflen'] = np.abs(pd_data['description_2'].apply(len) - pd_data['description_1'].apply(len))
    pd_features['description_levenshtein'] = pd.read_hdf(FOLDER + "data/Feat_train_description_leven.h")
    pd_features['description_Dlevenshtein'] = pd.read_hdf(FOLDER + "data/Feat_train_description_demarauleven.h")
    pd_features['description_jaro'] = pd.read_hdf(FOLDER + "data/Feat_train_description_jaro.h")
    pd_features['description_jarowinkler'] = pd.read_hdf(FOLDER + "data/Feat_train_description_jarowinkler.h")
    pd_features['description_hamming'] = pd.read_hdf(FOLDER + "data/Feat_train_description_hamming.h")

    # price
    pd_features['price_diff_abs'] = np.abs(pd_data['price_1'] - pd_data['price_2'])
    pd_features['price_diff_square'] = np.square(pd_data['price_1'] - pd_data['price_2'])

    # attrJson
    pd_features['attrsJSON_difflen'] = np.abs(pd_data['attrsJSON_1'].apply(len) - pd_data['attrsJSON_2'].apply(len))

    # images_array
    pd_features['images_diff_number'] = np.abs(pd_data['images_array_1'].apply(lambda x: len(x.split(','))) -
                                               pd_data['images_array_2'].apply(lambda x: len(x.split(','))))

    # geographic features
    pd_features['is_metroID_same'] = 1 * (pd_data['metroID_1'] == pd_data['metroID_2'])
    pd_features['is_locationID_same'] = 1 * (pd_data['locationID_1'] == pd_data['locationID_2'])
    pd_features['is_regionID_same'] = 1 * (pd_data['regionID_1'] == pd_data['regionID_2'])
    pd_features['haversine'] = haversine(pd_data, lon1='lon_1', lat1='lat_1', lon2='lon_2', lat2='lat_2')
    pd_features['haversine'] = (pd_features['haversine'] - pd_features['haversine'].mean()) / pd_features['haversine'].std()

    # CATEGORY, PARENT CATEGORY
    pd_features['is_categoryID_same'] = 1 * (pd_data['categoryID_1'] == pd_data['categoryID_2'])
    pd_features['is_parentCategoryID_same'] = 1 * (pd_data['parentCategoryID_1'] == pd_data['parentCategoryID_2'])



    pd_features['isDuplicate'] = pd_data['isDuplicate']
    pd_features['id'] = pd_data['id']

    return pd_features

D1 = create_D1(pd_data)
D1.to_hdf(FOLDER + "data/D1_with_adspecific_features_22may.p", 'w')



# # NI
# temp = train['titulaires'].apply(lambda st : st.split(','))
# labos = set([l for j in temp for l in j])
# for lab in labos:
#     train[lab] = train['titulaires'].apply(lambda x : 1 if lab in x else 0)
#     test[lab] = test['titulaires'].apply(lambda x : 1 if lab in x else 0)

