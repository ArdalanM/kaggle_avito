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
pdCategory, _ = loadFileinZipFile(FOLDER + "data/Category.csv.zip", encoding='utf-8')
pdLocation, _ = loadFileinZipFile(FOLDER + "data/Location.csv.zip",  encoding='utf-8')
pdItemPairs_train, _ = loadFileinZipFile(FOLDER + "data/ItemPairs_train.csv.zip", encoding='utf-8')
pdItemInfo_train, _ = loadFileinZipFile(FOLDER + "data/ItemInfo_train.csv.zip", encoding='utf-8')

print("MERGING DATA WITH itemID_1")
pdItemInfo_train.rename(columns={'itemID':'itemID_1'}, inplace=True)
item1 = pd.merge(pdItemPairs_train, pdItemInfo_train, how='inner', on='itemID_1')
item1 = pd.merge(item1, pdCategory, how='inner', on='categoryID')
item1 = pd.merge(item1, pdLocation, how='inner', on='locationID')
print("Renaming... ITEM_1")
print("------------------")
for col in item1.columns.tolist():
    if not col.endswith('_1') and not col.endswith('_2') :
        old_name = col
        new_name = col + '_1'
        print("{} => {}".format(old_name, new_name))
        item1.rename(columns={old_name: new_name}, inplace=True)
print("------------------")

print("MERGING DATA WITH itemID_2")
pdItemInfo_train.rename(columns={'itemID_1':'itemID_2'}, inplace=True)
pdtrain = pd.merge(item1, pdItemInfo_train, how='inner', on='itemID_2')
pdtrain = pd.merge(pdtrain, pdCategory, how='inner', on='categoryID')
pdtrain = pd.merge(pdtrain, pdLocation, how='inner', on='locationID')

print("Renaming... ITEM_2")
print("------------------")
for col in pdtrain.columns.tolist():
    if not col.endswith('_1') and not col.endswith('_2') :
        old_name = col
        new_name = col + '_2'
        print("{} => {}".format(old_name, new_name))
        pdtrain.rename(columns={old_name: new_name}, inplace=True)
print("------------------")

pdtrain.rename(columns={'isDuplicate_1': 'isDuplicate'}, inplace=True)
pdtrain.rename(columns={'generationMethod_1': 'generationMethod'}, inplace=True)

len_pdtrain = 2991396

pdtrain_part1 = pdtrain.loc[:1e6]
pdtrain_part2 = pdtrain.loc[1e6+1:]

pdtrain_part1.to_hdf(FOLDER + 'data/train_merged-part1.h', 'w')
pdtrain_part2.to_hdf(FOLDER + 'data/train_merged-part2.h', 'w')



print("SAME FOR TEST SET")
pdCategory, _ = loadFileinZipFile(FOLDER + "data/Category.csv.zip")
pdLocation, _ = loadFileinZipFile(FOLDER + "data/Location.csv.zip")
pdItemPairs_train, _ = loadFileinZipFile(FOLDER + "data/ItemPairs_test.csv.zip")
pdItemInfo_train, _ = loadFileinZipFile(FOLDER + "data/ItemInfo_test.csv.zip")

# Merging all for itemID_1
print("MERGING DATA WITH itemID_1")
pdItemInfo_train.rename(columns={'itemID':'itemID_1'}, inplace=True)
item1 = pd.merge(pdItemPairs_train, pdItemInfo_train, how='inner', on='itemID_1')
item1 = pd.merge(item1, pdCategory, how='inner', on='categoryID')
item1 = pd.merge(item1, pdLocation, how='inner', on='locationID')

print("Renaming... ITEM_1")
print("------------------")
for col in item1.columns.tolist():
    if not col.endswith('_1') and not col.endswith('_2') :
        old_name = col
        new_name = col + '_1'
        print("{} => {}".format(old_name, new_name))
        item1.rename(columns={old_name: new_name}, inplace=True)
print("------------------")

print("MERGING DATA WITH itemID_2")
pdItemInfo_train.rename(columns={'itemID_1':'itemID_2'}, inplace=True)
pdtrain = pd.merge(item1, pdItemInfo_train, how='inner', on='itemID_2')
pdtrain = pd.merge(pdtrain, pdCategory, how='inner', on='categoryID')
pdtrain = pd.merge(pdtrain, pdLocation, how='inner', on='locationID')

print("Renaming... ITEM_2")
print("------------------")
for col in pdtrain.columns.tolist():
    if not col.endswith('_1') and not col.endswith('_2') :
        old_name = col
        new_name = col + '_2'
        print("{} => {}".format(old_name, new_name))
        pdtrain.rename(columns={old_name: new_name}, inplace=True)
print("------------------")

pdtrain.rename(columns={'id_1': 'id'}, inplace=True)
pdtrain.to_hdf(FOLDER + 'data/test_merged.h', 'wb')

print("DONE... TEST stored")

