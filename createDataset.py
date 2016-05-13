DATA_FOLDER = "input/" #TODO change

import zipfile
import pandas as pd
import numpy as np

def loadFileinZipFile(zip_filename, dtypes=None, parsedate=None, password=None, **kvargs):
    """
    Load file to dataframe.
    """
    with zipfile.ZipFile(zip_filename, 'r') as z:
        if password:
            z.setpassword(password)

        inside_zip_filename = z.filelist[0].filename

        if parsedate:
            pd_data = pd.read_csv(z.open(inside_zip_filename), sep=',', parse_dates=parsedate, dtype=dtypes,
                                  **kvargs)
        else:
            pd_data = pd.read_csv(z.open(inside_zip_filename), sep=',', dtype=dtypes, **kvargs)
        return pd_data, inside_zip_filename


print("------- Load... TRAIN item pairs")
pairs_train = loadFileinZipFile(DATA_FOLDER + "ItemPairs_train.csv.zip")
print("------- Load... TEST item pairs")
# pairs_test = loadFileinZipFile(DATA_FOLDER + "ItemPairs_test.csv.zip")

print("------- Load... TRAIN item info")
item_info_train = loadFileinZipFile(DATA_FOLDER + "ItemInfo_train.csv.zip")
print("Check missing values in TRAIN")
print(item_info_train.isnull().sum())

print("------- Load... TEST item info")
# item_info_test = loadFileinZipFile(DATA_FOLDER + "ItemInfo_test.csv.zip")


