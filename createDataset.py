__author__ = 'Ardalan'

# CODE_FOLDER = "/home/arda/Documents/kaggle/bnp/"
CODE_FOLDER = "/home/ardalan/Documents/kaggle/bnp/"

import os, sys, time, re, collections, operator, copy, itertools, zipfile
import pandas as pd
import numpy as np

class DummycolumnsBins():

    def __init__(self, cols=None, prefix='LOL_', nb_bins=10):

        self.prefix=prefix
        self.nb_bins = nb_bins
        self.cols = cols
        self.bins = None

    def fit(self, data):

        self.bins = np.linspace(data[self.cols].min(), data[self.cols].max(), self.nb_bins)

        return self

    def transform(self, data):

        pd_dummy = pd.get_dummies(np.digitize(data[self.cols], self.bins), prefix=self.prefix)
        # pd_dummy.index = data[self.cols].index
        # pd_dummy = pd_dummy.groupby(pd_dummy.index).sum()

        return pd_dummy

def loadFileinZipFile(zip_filename, filename, dtypes=None, parsedate = None, password=None, **kvargs):
    """
    Load file to dataframe.
    """
    with zipfile.ZipFile(zip_filename, 'r') as myzip:
        if password:
            myzip.setpassword(password)

        if parsedate:
            return pd.read_csv(myzip.open(filename), sep=',', parse_dates=parsedate, dtype=dtypes, **kvargs)
        else:
            return pd.read_csv(myzip.open(filename), sep=',', dtype=dtypes, **kvargs)

class Dummycolumns():


    def __init__(self, cols=None, prefix='LOL_', nb_features=10):

        self.selected_features = None
        self.rejected_features = None
        self.prefix=prefix
        self.nb_features = nb_features
        self.cols = cols

    def fit(self, pd_train, pd_test):

        #Frequent item ==> Dummify
        selected_features_train = pd_train[self.cols].value_counts().index[:self.nb_features]
        selected_features_test = pd_test[self.cols].value_counts().index[:self.nb_features]

        self.selected_features = list(set(selected_features_train).intersection(set(selected_features_test)))

        #Rare items ==> gather all into a "garbage" column
        rejected_features_train = pd_train[self.cols].value_counts().index[self.nb_features:]
        rejected_features_test = pd_test[self.cols].value_counts().index[self.nb_features:]

        self.rejected_features = list(set(rejected_features_train).intersection(set(rejected_features_test)))

    def transform(self, data):

        df_dummy = data[self.cols].apply(lambda r: r if r in self.selected_features else 'LowFreqFeat')

        #Dummy all items
        df_dummy = pd.get_dummies(df_dummy).groupby(df_dummy.index).sum()


        df_dummy = df_dummy.rename(columns=lambda x: self.prefix + str(x))

        return df_dummy

def getAllPermutation(pd_data, cols):
    df = pd.DataFrame()
    for col1 in pd_data[cols]:
        print("all permutation with col", col1)
        for col2 in pd_data[cols]:
            df[col1+"_"+col2] = pd_data[col1] * pd_data[col2]
    return df


pdtrain = loadFileinZipFile(CODE_FOLDER + "data/train.csv.zip", "train.csv")
pdtest = loadFileinZipFile(CODE_FOLDER + "data/test.csv.zip", "test.csv")
