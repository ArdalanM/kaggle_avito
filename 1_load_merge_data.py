"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief: Load zipped files, merges them and dump into binary files
"""

import os
import pandas as pd
import config
from lib import utils, logging_utils, pkl_utils

print(config.DATA_FOLDER)

def load_merge(Category, Location, ItemPair, ItemInfo):
    logger.info("Loading train parts")
    logger.info(Category)
    logger.info(Location)
    logger.info(ItemPair)
    logger.info(ItemInfo)

    logger.info("Loading files")
    pdCategory, _ = utils.loadFileinZipFile(Category, encoding='utf-8')
    pdLocation, _ = utils.loadFileinZipFile(Location, encoding='utf-8')
    pdItemPairs, _ = utils.loadFileinZipFile(ItemPair, encoding='utf-8')
    pdItemInfo, _ = utils.loadFileinZipFile(ItemInfo, encoding='utf-8')


    logger.info("Merging")
    pdtrain = pd.merge(pdItemPairs, pdItemInfo, how='left', left_on='itemID_1', right_on='itemID')
    pdtrain = pd.merge(pdtrain, pdCategory, how='left', on='categoryID')
    pdtrain = pd.merge(pdtrain, pdLocation, how='left', on='locationID')
    pdtrain.drop(['itemID_1'], 1, inplace=True)

    logger.info("Renaming")
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

    logger.info("Merging")
    pdtrain = pd.merge(pdtrain, pdItemInfo, how='left', left_on='itemID_2', right_on='itemID')
    pdtrain = pd.merge(pdtrain, pdCategory, how='left', on='categoryID')
    pdtrain = pd.merge(pdtrain, pdLocation, how='left', on='locationID')
    pdtrain.drop(['itemID_2'], 1, inplace=True)

    logger.info("Renaming")
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

    return pdtrain

if __name__ == "__main__":

    logger = logging_utils._get_logger(config.CODE_FOLDER, "1_load_merge_data.log")

    logger.info("load train set")
    Category = os.path.join(config.DATA_FOLDER, "Category.csv.zip")
    Location = os.path.join(config.DATA_FOLDER, "Location.csv.zip")
    ItemPair = os.path.join(config.DATA_FOLDER, "ItemPairs_train.csv.zip")
    ItemInfo = os.path.join(config.DATA_FOLDER, "ItemInfo_train.csv.zip")
    pdtrain = load_merge(Category, Location, ItemPair, ItemInfo)

    logger.info("load test set")
    Category_test = os.path.join(config.DATA_FOLDER, "Category.csv.zip")
    Location_test = os.path.join(config.DATA_FOLDER, "Location.csv.zip")
    ItemPair_test = os.path.join(config.DATA_FOLDER, "ItemPairs_test.csv.zip")
    ItemInfo_test = os.path.join(config.DATA_FOLDER, "ItemInfo_test.csv.zip")
    pdtest = load_merge(Category_test, Location_test, ItemPair_test, ItemInfo_test)

    logger.info("merge both dataset")
    pdall = pdtrain.append(pdtest)

    logger.info("save to pickle")
    pkl_utils._save(os.path.join(config.PICKLE_DATA_FOLDER, "train_raw.pkl"), pdtrain)
    pkl_utils._save(os.path.join(config.PICKLE_DATA_FOLDER, "test_raw.pkl"), pdtest)
    pkl_utils._save(os.path.join(config.PICKLE_DATA_FOLDER, "all_raw.pkl"), pdall)



