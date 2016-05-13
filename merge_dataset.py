__author__ = 'Ardalan'

FOLDER = "/home/ardalan/Documents/kaggle/avito/"

import os, sys, time, re, collections, operator, copy, itertools, zipfile
import pandas as pd
import numpy as np


def loadFileinZipFile(zip_filename, dtypes=None, parsedate=None, password=None, **kvargs):
    """
        Load file to dataframe.
        """
    with zipfile.ZipFile(zip_filename, 'r') as myzip:
        if password:
            myzip.setpassword(password)

        inside_zip_filename = myzip.filelist[0].filename

        if parsedate:
            pd_data = pd.read_csv(myzip.open(inside_zip_filename), sep=',', parse_dates=parsedate, dtype=dtypes,
                                  **kvargs)
        else:
            pd_data = pd.read_csv(myzip.open(inside_zip_filename), sep=',', dtype=dtypes, **kvargs)
        return pd_data, inside_zip_filename



print("Loading train parts")
pdCategory, _ = loadFileinZipFile(FOLDER + "data/Category.csv.zip")
pdLocation, _ = loadFileinZipFile(FOLDER + "data/Location.csv.zip")
pdItemPairs_train, _ = loadFileinZipFile(FOLDER + "data/ItemPairs_train.csv.zip")
pdItemInfo_train, _ = loadFileinZipFile(FOLDER + "data/ItemInfo_train.csv.zip")

print("Merging all for itemID_1")
item1 = pd.merge(pdItemInfo_train, pdItemPairs_train, how='inner', left_on='itemID', right_on='itemID_1')
item1 = pd.merge(item1, pdCategory, how='inner', on='categoryID')
item1 = pd.merge(item1, pdLocation, how='inner', on='locationID')

assert np.mean((item1['itemID'] - item1['itemID_1'])) == 0
item1.drop(['itemID_1'], 1, inplace=True)

print("Renaming...")
for col in item1.columns.tolist():
    if not col.endswith('_2'):
        old_name = col
        new_name = col + '_1'
        print("{} => {}".format(old_name, new_name))
        item1.rename(columns={old_name: new_name}, inplace=True)

print("Merging item2 into item1")
pdtrain = pd.merge(item1, pdItemInfo_train, how='inner', left_on='itemID_2', right_on='itemID')
pdtrain.drop(['itemID_2'], 1, inplace=True)
pdtrain = pd.merge(pdtrain, pdCategory, how='inner', on='categoryID')
pdtrain = pd.merge(pdtrain, pdLocation, how='inner', on='locationID')

print("Renaming...")
for col in pdtrain.columns.tolist():
    if not col.endswith("_1"):
        old_name = col
        new_name = col + '_2'
        print("{} => {}".format(old_name, new_name))
        pdtrain.rename(columns={old_name: new_name}, inplace=True)
pdtrain.rename(columns={'isDuplicate_1': 'isDuplicate'}, inplace=True)
pdtrain.rename(columns={'generationMethod_1': 'generationMethod'}, inplace=True)

pdtrain.to_csv(FOLDER + 'data/train_merged.csv', index=None)



print("SAME FOR TEST SET")
pdCategory, _ = loadFileinZipFile(FOLDER + "data/Category.csv.zip")
pdLocation, _ = loadFileinZipFile(FOLDER + "data/Location.csv.zip")
pdItemPairs_train, _ = loadFileinZipFile(FOLDER + "data/ItemPairs_test.csv.zip")
pdItemInfo_train, _ = loadFileinZipFile(FOLDER + "data/ItemInfo_test.csv.zip")

# Merging all for itemID_1
item1 = pd.merge(pdItemInfo_train, pdItemPairs_train, how='inner', left_on='itemID', right_on='itemID_1')
item1 = pd.merge(item1, pdCategory, how='inner', on='categoryID')
item1 = pd.merge(item1, pdLocation, how='inner', on='locationID')

assert np.mean((item1['itemID'] - item1['itemID_1'])) == 0
item1.drop(['itemID_1'], 1, inplace=True)

for col in item1.columns.tolist():
    if not col.endswith('_2'):
        old_name = col
        new_name = col + '_1'
        print("{} => {}".format(old_name, new_name))
        item1.rename(columns={old_name: new_name}, inplace=True)

# Merging item2 into item1

pdtrain = pd.merge(item1, pdItemInfo_train, how='inner', left_on='itemID_2', right_on='itemID')
pdtrain.drop(['itemID_2'], 1, inplace=True)
pdtrain = pd.merge(pdtrain, pdCategory, how='inner', on='categoryID')
pdtrain = pd.merge(pdtrain, pdLocation, how='inner', on='locationID')

for col in pdtrain.columns.tolist():
    if not col.endswith("_1"):
        old_name = col
        new_name = col + '_2'
        print("{} => {}".format(old_name, new_name))
        pdtrain.rename(columns={old_name: new_name}, inplace=True)
pdtrain.to_csv(FOLDER + 'data/test_merged.csv', index=None)



