# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>

@brief:
"""

import os, sys, copy, string, config, inspect, regex, operator
import numpy as np
import pandas as pd
import theano.sandbox.cuda

from sklearn import cross_validation

from utils import pkl_utils, time_utils, logging_utils, metric_utils, np_utils

from xgboost import XGBClassifier

theano.sandbox.cuda.use(config.CPU)

def compute_metric(col1, col2, metric=None, round=False):
    tot = 0
    for i, (y, ypred) in enumerate(zip(col1, col2)):

        if round:
            ypred = ypred.argmax(-1)
        tot += metric(y, ypred)
    tot /= i + 1
    return tot
def split_df(df, test_size=0.2):
    train_size = 1 - test_size
    index_split = int(train_size * len(df))

    pdtrain = df[:index_split]
    pdtest = df[index_split:]
    return pdtrain, pdtest
def get_skf(len_Y, test_size=0.2):
    vec = -1 * np.ones(len_Y)

    train_size = 1 - test_size
    thresh_index = int(train_size * len_Y)
    vec[thresh_index:] = 1

    ps = cross_validation.PredefinedSplit(test_fold=vec)
    return ps
def vectorize(df, cols=[], mapping_col=None):
    if mapping_col and len(cols) == 1:
        Y = np.array(df[cols[0]].apply(lambda r: mapping_col[r])).astype(np.int8)
    else:
        Y = np.array(df[cols].astype(np.int8))

    return Y

def get_XY():


    pd_data = pkl_utils._load(config.ALL_DATA_CLEANED)

    Y = pd_data[config.LABEL_COL]
    del pd_data

    pd_dataset = pd.DataFrame({config.LABEL_COL: Y})
    features = []

    import glob
    filenames = glob.glob1(config.FEATURES_FOLDER, "*.pkl")
    for filename in filenames:
        filepath = os.path.join(config.FEATURES_FOLDER, filename)
        features.append(filename)
        pd_dataset[filename] = pkl_utils._load(filepath)

    pd_cv = pd_dataset[pd_dataset[config.LABEL_COL].isnull() == False]
    pd_submission = pd_dataset[pd_dataset[config.LABEL_COL].isnull() == True]

    Y_cv = np.array(pd_cv[config.LABEL_COL]).astype(np.int8)
    X_cv = np.array(pd_cv.drop([config.LABEL_COL], 1))
    X_sub = np.array(pd_submission.drop([config.LABEL_COL], 1))

    return X_cv, Y_cv, X_sub, features


logger = logging_utils._get_logger(config.LOG_FOLDER, "run_xgb.log")

X_cv, Y_cv, X_sub, features = get_XY()
print("nb features: {}".format(len(features)))

skf = cross_validation.StratifiedShuffleSplit(Y_cv, n_iter=1, test_size=0.2, random_state=config.SEED)

# Splitting train/validation
train_idx, test_idx = list(skf)[0]
X_train = X_cv[train_idx, :]
Y_train = Y_cv[train_idx]

X_test = X_cv[test_idx, :]
Y_test = Y_cv[test_idx]


clf = XGBClassifier(n_estimators=5000, learning_rate=0.03, max_depth=10, nthread=4)


clf.fit(X_train, Y_train, eval_set=[(X_test, Y_test)],
        early_stopping_rounds=100, eval_metric='auc', verbose=1)


y_pred = clf.predict_proba(X_test, ntree_limit=clf.best_ntree_limit)

auc = metric_utils.auc(Y_test, y_pred)
print("auc: {}".format(auc))

clf.n_estimators = clf.best_iteration

print("Fitting whole training set")
print("setting n_estimators to: {}".format(clf.n_estimators))
clf.fit(X_cv, Y_cv)
y_submission = clf.predict_proba(X_sub, ntree_limit=clf.best_ntree_limit)[:, 1]

pd_submission = pd.DataFrame({'id': range(len(y_submission)), 'probability': y_submission})
filename = "{0}_nbfeat_{1}_{2:0.3f}_{3}.csv".format(clf.__class__.__name__,
                                                    len(features),
                                                    auc,
                                                    time_utils._timestamp_pretty())
filepath = os.path.join(config.DATA_FOLDER, filename)
print(filepath)

pd_submission.to_csv(filepath, index=None)
