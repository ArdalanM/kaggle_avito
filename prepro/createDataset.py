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
del pdtrain1
del pdtrain2

pdtrain['title_1'].fillna("", inplace=True)
pdtrain['title_2'].fillna("", inplace=True)
pdtrain['description_1'].fillna("", inplace=True)
pdtrain['description_2'].fillna("", inplace=True)
pdtrain['price_1'].fillna(-9999, inplace=True)
pdtrain['price_2'].fillna(-9999, inplace=True)
pdtrain['attrsJSON_1'].fillna("", inplace=True)
pdtrain['attrsJSON_2'].fillna("", inplace=True)
pdtrain['images_array_1'].fillna("", inplace=True)
pdtrain['images_array_2'].fillna("", inplace=True)

# feature dataframe
pd_features = pd.DataFrame()

# title
pd_features['title_difflen'] = np.abs(pdtrain['title_2'].apply(len) - pdtrain['title_1'].apply(len))
pd_features['title_levenshtein'] = pd.read_hdf(FOLDER + "data/feat_train_title_leven.h")
pd_features['title_Dlevenshtein'] = pd.read_hdf(FOLDER + "data/feat_train_title_demarauleven.h")
pd_features['title_jaro'] = pd.read_hdf(FOLDER + "data/feat_train_title_jaro.h")
pd_features['title_jarowinkler'] = pd.read_hdf(FOLDER + "data/feat_train_title_jarowinkler.h")
pd_features['title_hamming'] = pd.read_hdf(FOLDER + "data/feat_train_title_hamming.h")

# description
pd_features['description_difflen'] = np.abs(pdtrain['description_2'].apply(len) - pdtrain['description_1'].apply(len))
pd_features['description_levenshtein'] = pd.read_hdf(FOLDER + "data/feat_train_description_leven.h")
pd_features['description_Dlevenshtein'] = pd.read_hdf(FOLDER + "data/feat_train_description_demarauleven.h")
pd_features['description_jaro'] = pd.read_hdf(FOLDER + "data/feat_train_description_jaro.h")
pd_features['description_jarowinkler'] = pd.read_hdf(FOLDER + "data/feat_train_description_jarowinkler.h")
pd_features['description_hamming'] = pd.read_hdf(FOLDER + "data/feat_train_description_hamming.h")

# price
pd_features['price_diff_abs'] = np.abs(pdtrain['price_1'] - pdtrain['price_2'])
pd_features['price_diff_square'] = np.square(pdtrain['price_1'] - pdtrain['price_2'])

# attrJson
pd_features['attrsJSON_difflen'] = np.abs(pdtrain['attrsJSON_1'].apply(len) - pdtrain['attrsJSON_2'].apply(len))

# images_array
pd_features['images_diff_number'] = np.abs(pdtrain['images_array_1'].apply(lambda x: len(x.split(','))) -
                                       pdtrain['images_array_2'].apply(lambda x: len(x.split(','))))

# geographic features
pd_features['metroID_same'] = 1 * (pdtrain['metroID_1'] == pdtrain['metroID_2'])
pd_features['locationID_same'] = 1 * (pdtrain['locationID_1'] == pdtrain['locationID_2'])
pd_features['regionID_same'] = 1 * (pdtrain['regionID_1'] == pdtrain['regionID_2'])
pd_features['haversine'] = haversine(pdtrain, lon1='lon_1', lat1='lat_1', lon2='lon_2', lat2='lat_2')
pd_features['haversine'] = (pd_features['haversine'] - pd_features['haversine'].mean()) / pd_features['haversine'].std()

# CATEGORY, PARENT CATEGORY
pd_features['categoryID_same'] = 1 * (pdtrain['categoryID_1'] == pdtrain['categoryID_2'])
pd_features['parentCategoryID_same'] = 1 * (pdtrain['parentCategoryID_1'] == pdtrain['parentCategoryID_2'])



Y = pdtrain['isDuplicate'].values
X = np.array(pd_features)


from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest= train_test_split(X, Y, test_size=0.20, random_state=42)
clf = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=1000, objective="binary:logistic",
                    nthread=8)

clf.fit(xtrain, ytrain, eval_set=[(xtest, ytest)], eval_metric="auc", early_stopping_rounds=100)



