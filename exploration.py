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


pdtrain, _ = loadFileinZipFile(FOLDER + "data/train_merged.csv.zip")

