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
from keras import layers, optimizers
from keras.models import Model, Sequential

theano.sandbox.cuda.use(config.GPU0)

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


from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler()

X_cv = scl.fit_transform(X_cv)
X_sub = scl.transform(X_sub)

skf = cross_validation.StratifiedShuffleSplit(Y_cv, n_iter=1, test_size=0.2, random_state=config.SEED)

# Splitting train/validation
train_idx, test_idx = list(skf)[0]
X_train = X_cv[train_idx, :]
Y_train = Y_cv[train_idx]

X_test = X_cv[test_idx, :]
Y_test = Y_cv[test_idx]




class genericKerasCLF():
    def __init__(self, batch_size=128, nb_epoch=2, verbose=2, callbacks= [],
                 shuffle=True, metrics=None, class_weight=None, rebuild=True):
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.callbacks = callbacks
        self.shuffle = shuffle
        self.metrics = metrics
        self.rebuild = rebuild

        self.model = None
        self.logs = None
        self.class_weight = class_weight

    def build_model(self):
        pass

    def fit(self, X, y, validation_data=()):

        # X, y, validation_data = self._set_input_dim(X, y, validation_data)

        # print(y.shape)
        # print(validation_data[1].shape)
        # print("input_dim: {}, output_dim: {}, input_length: {}"
        #     .format(self.input_dim,
        #             self.output_dim,
        #             self.input_length))

        nb_class = len(np.unique(y))
        y = np_utils._to_categorical(y, nb_class)

        validation_data = (validation_data[0],
                           np_utils._to_categorical(validation_data[1], nb_class))

        if self.rebuild:
            self.model = self.build_model()

        logs = self.model.fit(X, y, batch_size=self.batch_size, nb_epoch=self.nb_epoch,
                              validation_data=validation_data,
                              verbose=self.verbose, shuffle=self.shuffle,
                              callbacks=copy.deepcopy(self.callbacks),
                              class_weight=self.class_weight)
        self.logs = logs

    def predict_proba(self, X):

        assert hasattr(self.model, 'predict_proba') or hasattr(self.model, 'predict')

        if hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict_proba(X, verbose=False)
        else:
            prediction = self.model.predict(X, verbose=False)

        return prediction

class MLP_merged(genericKerasCLF):
    def build_model(self):

        print('Build model...')
        input_cat = layers.Input(shape=(X_cv_cat.shape[1],))
        input_text = layers.Input(shape=(X_cv_text.shape[1],))

        cat_output = layers.Dense(50, activation="relu")(input_cat)
        cat_output = layers.Dropout(0.2)(cat_output)

        text_output = layers.Dense(512, activation="relu")(input_text)
        text_output = layers.Dropout(0.2)(text_output)
        text_output = layers.Dense(128, activation="relu")(text_output)
        text_output = layers.Dropout(0.2)(text_output)
        text_output = layers.Dense(50, activation="relu")(text_output)
        text_output = layers.Dropout(0.2)(text_output)

        merged = layers.merge([cat_output, text_output], mode='concat')

        output = layers.Dense(50, activation="relu")(merged)
        output = layers.Dropout(0.2)(output)
        output = layers.Dense(2, activation="softmax")(output)
        model = Model(input=[input_cat, input_text], output=output)
        model.compile(optimizer=optimizers.adam(), loss='categorical_crossentropy', metrics=self.metrics)
        print(model.summary())
        return model

class MLP(genericKerasCLF):
    def build_model(self):

        print('Build model...')
        input_text = layers.Input(shape=(X_cv.shape[1],))

        output = layers.Dense(128, activation="relu")(input_text)
        output = layers.Dropout(0.2)(output)
        output = layers.Dense(128, activation="relu")(output)
        output = layers.Dropout(0.2)(output)
        output = layers.Dense(50, activation="relu")(output)
        output = layers.Dropout(0.2)(output)
        output = layers.Dense(2, activation="softmax")(output)
        model = Model(input=input_text, output=output)
        model.compile(optimizer=optimizers.rmsprop(), loss='categorical_crossentropy', metrics=self.metrics)
        print(model.summary())
        return model


# clf = MLP_merged(batch_size=128, nb_epoch=2, metrics=['accuracy'], verbose=2,  class_weight={0: 1, 1: 1})
clf = MLP(batch_size=128, nb_epoch=100, metrics=['accuracy'], verbose=2,  class_weight={0: 1, 1: 1})
# clf = linear_model.LogisticRegression(C=1.1, penalty='l2')
# clf = ensemble.RandomForestClassifier(verbose=4, n_jobs=12)


clf.fit(X_train, Y_train, validation_data=(X_test, Y_test))


y_pred = clf.predict_proba(X_test, ntree_limit=clf.best_ntree_limit)

auc = metric_utils.auc(Y_test, y_pred)


clf.n_estimators = clf.best_iteration
clf.fit(X_cv, Y_cv)
y_submission = clf.predict_proba(X_sub, ntree_limit=clf.best_ntree_limit)[:, 1]

pd_submission = pd.DataFrame({'id': range(len(y_submission)), 'probability': y_submission})
filename = "{0}_nbfeat_{1}_{2:0.3f}_{3}.csv".format(clf.__class__.__name__,
                                                    len(features),
                                                    auc,
                                                    time_utils._timestamp_pretty())

filepath = os.path.join(config.DATA_FOLDER, filename)
pd_submission.to_csv(filepath, index=None)
