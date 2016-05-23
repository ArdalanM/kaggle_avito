__author__ = 'Ardalan'
"""
This script do:
    1: Loading kaggle dataset (train and test_ from zipfiles
    2: Merge all the parts together
    3: Dump merged dataset into binary files (train-merged.h, test-merged.h)
"""


FOLDER = "/home/ardalan/Documents/kaggle/avito/"

import zipfile
import pandas as pd


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
pdLocation, _ = loadFileinZipFile(FOLDER + "data/Location.csv.zip", encoding='utf-8')
pdItemPairs_train, _ = loadFileinZipFile(FOLDER + "data/ItemPairs_train.csv.zip", encoding='utf-8')
pdItemInfo_train, _ = loadFileinZipFile(FOLDER + "data/ItemInfo_train.csv.zip", encoding='utf-8')

pdtrain = pd.merge(pdItemPairs_train, pdItemInfo_train, how='left', left_on='itemID_1', right_on='itemID')
pdtrain = pd.merge(pdtrain, pdCategory, how='left', on='categoryID')
pdtrain = pd.merge(pdtrain, pdLocation, how='left', on='locationID')
pdtrain.drop(['itemID_1'], 1, inplace=True)

pdtrain.rename(columns={
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

pdtrain = pd.merge(pdtrain, pdItemInfo_train, how='left', left_on='itemID_2', right_on='itemID')
pdtrain = pd.merge(pdtrain, pdCategory, how='left', on='categoryID')
pdtrain = pd.merge(pdtrain, pdLocation, how='left', on='locationID')
pdtrain.drop(['itemID_2'], 1, inplace=True)


pdtrain.rename(columns={
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


len_pdtrain = 2991396

pdtrain_part1 = pdtrain.loc[:1e6]
pdtrain_part2 = pdtrain.loc[1e6 + 1:]

print("Saving train set...")
pdtrain_part1.to_hdf(FOLDER + 'data/train_merged-part1.h', 'w')
pdtrain_part2.to_hdf(FOLDER + 'data/train_merged-part2.h', 'w')



print("SAME FOR TEST SET")
print("Loading test parts")
pdCategory, _ = loadFileinZipFile(FOLDER + "data/Category.csv.zip", encoding='utf-8')
pdLocation, _ = loadFileinZipFile(FOLDER + "data/Location.csv.zip", encoding='utf-8')
pdItemPairs_test, _ = loadFileinZipFile(FOLDER + "data/ItemPairs_test.csv.zip", encoding='utf-8')
pdItemInfo_test, _ = loadFileinZipFile(FOLDER + "data/ItemInfo_test.csv.zip", encoding='utf-8')

pdtest = pd.merge(pdItemPairs_test, pdItemInfo_test, how='left', left_on='itemID_1', right_on='itemID')
pdtest = pd.merge(pdtest, pdCategory, how='left', on='categoryID')
pdtest = pd.merge(pdtest, pdLocation, how='left', on='locationID')
pdtest.drop(['itemID_1'], 1, inplace=True)

pdtest.rename(columns={
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

pdtest = pd.merge(pdtest, pdItemInfo_test, how='left', left_on='itemID_2', right_on='itemID')
pdtest = pd.merge(pdtest, pdCategory, how='left', on='categoryID')
pdtest = pd.merge(pdtest, pdLocation, how='left', on='locationID')
pdtest.drop(['itemID_2'], 1, inplace=True)


pdtest.rename(columns={
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

print("Saving test set...")
pdtest.to_hdf(FOLDER + 'data/test_merged.h', 'w')