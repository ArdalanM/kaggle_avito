# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>

@brief:
"""

import os, sys, copy, string, config, inspect, regex, operator
import numpy as np
import pandas as pd
import theano.sandbox.cuda

from datetime import datetime
from nltk.corpus import stopwords
from datetime import date, timedelta

from sklearn.base import TransformerMixin
from sklearn import metrics, preprocessing, pipeline, cross_validation, feature_extraction, linear_model

from utils import logging_utils, loading_utils, metric_utils, cv_utils, np_utils, confidence_utils
from utils import skl_tranformers, xgb_utils, skl_models, pkl_utils, time_utils, keras_utils

from keras.models import Sequential, Model
from keras import layers, optimizers


# pd.options.display.max_columns = 999
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

def get_XY(category):

    # category = [
    #     "rencontresgaysetlesbiennesgay",
    #     "rencontressanslendemain",
    #     "rencontreserotica",
    #     "rencontresheteroshommesfemmes",
    # ]

    feat = "EditDistance_description_1_x_description_2_1D.pkl"

    pd_data = pkl_utils._load(config.ALL_DATA_CLEANED)

    Y_all = pd_data[config.LABEL_COL]
    del pd_data

    features = []
    pd_dataset = pd.DataFrame({config.LABEL_COL: Y_all})

    import glob
    filenames = glob.glob1(config.FEATURES_FOLDER, "*.pkl")
    for filename in filenames:
        filepath = os.path.join(config.FEATURES_FOLDER, filename)
        features.append(filename)
        pd_dataset[filename] = pkl_utils._load(filepath)


    Y = pd_dataset[pd_dataset[config.LABEL_COL].isnull() == False][config.LABEL_COL].values
    X = np.array(pd_dataset[pd_dataset[config.LABEL_COL].isnull() == False][features])
    X_cv = X

    X_submission = np.array(pd_dataset[pd_dataset[config.LABEL_COL].isnull() == True][features])


    skf = cross_validation.StratifiedShuffleSplit(Y, n_iter=1, test_size=0.2, random_state=config.SEED)

    # Splitting train/validation
    cv_idx, val_idx = list(skf)[0]
    pd_dataset_cv = pd_dataset.iloc[cv_idx, :].copy()
    pd_dataset_val = pd_dataset.iloc[val_idx, :].copy()
    Y_cv = Y[cv_idx]
    Y_val = Y[val_idx]

    dic= {
        'X_cv_cat': X_cv_cat, 'X_val_cat': X_val_cat,
        'X_cv_text': X_cv_text, 'X_val_text': X_val_text,
        'Y_cv': Y_cv, 'Y_val': Y_val,
        'features_cat': features_cat, 'features_word': features_word,
        'cv_idx': cv_idx, 'val_idx': val_idx, 'pd_dataset': pdgay
    }

    return dic



logger = logging_utils._get_logger(config.LOG_FOLDER, "run_cv_cpu.log")


##### Params #####
category = [
    "rencontresgaysetlesbiennesgay",
    "rencontressanslendemain",
    "rencontreserotica",
    "rencontresheteroshommesfemmes",
]

# rencontresgaysetlesbiennesgay-rencontressanslendemain-rencontreserotica-rencontresheteroshommesfemmes

nb_class = len(set(config.MAPPING_LABEL.values()))

# dic = get_XY(category)

category = [
       "Moda/Acessórios",
    ]
dic = get_skina(category)


feature_type = "vs-cat1234-wd-cw11"
feature_type = "skina-biggest-category"

clf_name = "MLPmerged"
clf_name = "LR"


##### Datasets #####
Y_cv = dic['Y_cv']
Y_val = dic['Y_val']
X_cv_text = dic['X_cv_text']
X_cv_cat = dic['X_cv_cat']

X_val_text = dic['X_val_text']
X_val_cat = dic['X_val_cat']
features_word =  dic['features_word']
features_cat = dic['features_cat']
features = features_word + features_cat


class SimpleCV:

    """
    None.
    """

    def __init__(self, logger=None, skf=None, clf=None,
                 nb_class=None, eval_func=None,
                 reshape_func=None):

        self.logger = logger
        self.skf = skf
        self.clf = clf
        self.nb_class = nb_class
        # self.clf_params = clf_params

        self.eval_func = eval_func
        self.reshape_func = reshape_func

    def _train_predict_clf(self, xtrain, xtest, ytrain, ytest, xval, clf):


        if self.input_is_list:
            clf.fit([xtrain[0], xtrain[1]], ytrain, validation_data=([xtest[0], xtest[1]], ytest))
            ytrain_predprob = clf.predict_proba([xtrain[0], xtrain[1]])
            ytest_predprob = clf.predict_proba([xtest[0], xtest[1]])
            yval_predprob = clf.predict_proba([xval[0], xval[1]])

        else:

            if "eval_set" in inspect.getargspec(clf.fit).args:
                # This is an xgboost model
                clf.fit(xtrain, ytrain, eval_set=[(xtest, ytest)])

            elif "validation_data" in inspect.getargspec(clf.fit).args:
                # This is a Keras model
                clf.fit(xtrain, ytrain, validation_data=(xtest, ytest))

            else:
                clf.fit(xtrain, ytrain)

            ytrain_predprob = clf.predict_proba(xtrain)
            ytest_predprob = clf.predict_proba(xtest)
            yval_predprob = clf.predict_proba(xval)

        # Reshaping predictions
        if self.reshape_func:
            ytrain_predprob = self.reshape_func(ytrain_predprob)
            ytest_predprob = self.reshape_func(ytest_predprob)
            yval_predprob = self.reshape_func(yval_predprob)

        # Hard labelling probabilities
        ytrain_pred = ytrain_predprob.argmax(-1)
        ytest_pred = ytest_predprob.argmax(-1)
        yval_pred = yval_predprob.argmax(-1)

        return ytrain_predprob, ytrain_pred, \
               ytest_predprob, ytest_pred, \
               yval_predprob, yval_pred


    def run_cv(self, X, Y, X_val, Y_val):

        self.logger.info("{} Simple Cross Validation: {} folds {}".format(
            "#" * 10, len(self.skf), "#" * 10))

        self.input_is_list = False
        if type(X) == list:
            self.input_is_list = True
            X, X1 = X[0], X[1]
            X_val, X_val1 = X_val[0], X_val[1]


        blend_X_val_fold = np.zeros((X_val.shape[0], self.nb_class))

        self.diclogs = {'name': str(self.clf),

                        'blend_X_train': np.zeros((X.shape[0], self.nb_class), dtype=np.float16),
                        'blend_X_val': np.zeros((X_val.shape[0], self.nb_class), dtype=np.float16),

                        'blend_Y_train': Y.reshape(len(Y), 1).astype(np.int8),
                        'blend_Y_val': Y_val.reshape(len(Y_val), 1).astype(np.int8),

                        'folds_pred_train': [], 'folds_label_train': [],
                        'folds_pred_test': [], 'folds_label_test': [],
                        'folds_pred_val': [], 'folds_label_val': [],

                        'folds_index_train': [], 'folds_index_test': [],

                        'best_epoch': [], 'best_val_metric': [],

                        'train_error': [], 'test_error': [], 'val_error': [], 'val_error_avg': []}

        # self.filename = '{0}_{1}_{2}_cv{3}'.format(tr_name, preprocessing_name, clf_name, skf.n_folds)

        self.logger.info("Classifier: \n{}\n".format(self.clf))
        self.logger.info("Train shape  :{}".format(X.shape))
        self.logger.info("Val shape   :{}".format(X_val.shape))
        self.logger.info("{0} Starting Cross validation {0}".format("#" * 10))

        # metric initialization
        self.diclogs['train_error'] = {metric_name: [] for metric_name, metric_func in self.eval_func}
        self.diclogs['test_error'] = {metric_name: [] for metric_name, metric_func in self.eval_func}
        self.diclogs['val_error'] = {metric_name: [] for metric_name, metric_func in self.eval_func}



        for fold_indice, (train_idx, test_idx) in enumerate(self.skf):

            self.logger.info("")
            self.logger.info("{} Fold [{}/{}] {}"
                             .format("#" * 5, fold_indice+1, len(self.skf), "#" * 5))

            if self.input_is_list:
                xtrain, xtrain1, ytrain = X[train_idx], X1[train_idx], Y[train_idx]
                xtest, xtest1, ytest = X[test_idx], X1[test_idx], Y[test_idx]

                ytrain_predprob, ytrain_pred, \
                ytest_predprob, ytest_pred, \
                yval_predprob, yval_pred = self._train_predict_clf([xtrain, xtrain1],
                                                                   [xtest, xtest1],
                                                                   ytrain, ytest, [X_val, X_val1], self.clf)

            else:
                xtrain, ytrain = X[train_idx], Y[train_idx]
                xtest, ytest = X[test_idx], Y[test_idx]

                ytrain_predprob, ytrain_pred, \
                ytest_predprob, ytest_pred, \
                yval_predprob, yval_pred = self._train_predict_clf(xtrain, xtest,
                                                                   ytrain, ytest,
                                                                   X_val, self.clf)


            blend_X_val_fold += yval_predprob

            # store metrics
            for metric_name, metric_func in self.eval_func:

                train_err = metric_func(ytrain, ytrain_predprob)
                test_err = metric_func(ytest, ytest_predprob)
                val_err = metric_func(Y_val, yval_predprob)

                self.logger.info("Metric {}: Train|Test|Validation: [{:.4f}|{:.4f}|{:.4f}]"
                                 .format(metric_name, train_err, test_err, val_err))

                self.diclogs['train_error'][metric_name].append(train_err)
                self.diclogs['test_error'][metric_name].append(test_err)
                self.diclogs['val_error'][metric_name].append(val_err)


            # Confusion Matrix
            self.logger.info("CM: Train, Test, Validation: \n{}\n\n{}\n\n{}"
                             .format(metrics.confusion_matrix(ytrain, ytrain_pred, range(self.nb_class)),
                                     metrics.confusion_matrix(ytest, ytest_pred, range(self.nb_class)),
                                     metrics.confusion_matrix(Y_val, yval_pred, range(self.nb_class))))

            # Storing fold indexes
            self.diclogs['folds_index_train'].append(train_idx)
            self.diclogs['folds_index_test'].append(test_idx)

            # Storing fold predictions
            self.diclogs['folds_pred_train'].append(ytrain_predprob)
            self.diclogs['folds_label_train'].append(ytrain)
            self.diclogs['folds_pred_test'].append(ytest_predprob)
            self.diclogs['folds_label_test'].append(ytest)

            self.diclogs['blend_X_train'][test_idx] = ytest_predprob


            if hasattr(self.clf, "get_params"):
                self.diclogs['params'] = self.clf.get_params()
            if hasattr(self.clf, "get_best_epoch"):
                self.logger.info("best_epoch: {}".format(self.clf.get_best_epoch()))
                self.diclogs['best_epoch'].append(self.clf.get_best_epoch())
            if hasattr(self.clf, "get_best_test_metric"):
                self.diclogs['best_test_metric'].append(self.clf.get_best_test_metric())

        self.logger.info("{0} Ending Cross validation {0}\n".format("#" * 10))
        # Averaging test set across fold predictions
        yval_predprob = blend_X_val_fold / len(self.skf)

        # Averaged metrics
        for metric_name, metric_func in self.eval_func:
            val_err_avg = metric_func(Y_val, yval_predprob)

            self.logger.info("Metric {} average: Train|Test|Validation: [{:.4f}|{:.4f}|{:.4f}]"
                 .format(metric_name,
                         np.mean(self.diclogs['train_error'][metric_name]),
                         np.mean(self.diclogs['test_error'][metric_name]),
                         val_err_avg))

            self.diclogs['val_error_avg'] = val_err_avg

        self.logger.info("CM: Validation: \n{}"
                         .format(metrics.confusion_matrix(Y_val, yval_pred, range(self.nb_class))))

        self.diclogs['folds_pred_val'].append(yval_predprob)
        self.diclogs['folds_label_val'].append(Y_val)
        self.diclogs['blend_X_val'] = yval_predprob


        return self.diclogs

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
        input_text = layers.Input(shape=(X_cv_text.shape[1],))

        output = layers.Dense(512, activation="relu")(input_text)
        output = layers.Dropout(0.2)(output)
        output = layers.Dense(128, activation="relu")(output)
        output = layers.Dropout(0.2)(output)
        output = layers.Dense(50, activation="relu")(output)
        output = layers.Dropout(0.2)(output)
        output = layers.Dense(2, activation="softmax")(output)
        model = Model(input=input_text, output=output)
        model.compile(optimizer=optimizers.adam(), loss='categorical_crossentropy', metrics=self.metrics)
        print(model.summary())
        return model


# clf = MLP_merged(batch_size=128, nb_epoch=2, metrics=['accuracy'], verbose=2,  class_weight={0: 1, 1: 1})
# clf = MLP(batch_size=128, nb_epoch=3, metrics=['accuracy'], verbose=2,  class_weight={0: 1, 1: 1})
clf = linear_model.LogisticRegression(C=1.1, penalty='l2')
# from sklearn import naive_bayes
# clf = naive_bayes.MultinomialNB(alpha=0.5)



##### Cross Validation ##
list_metrics = [("Accuracy", metric_utils.accuracy), ("AUC", metric_utils.auc)]
skf = cross_validation.StratifiedKFold(Y_cv, n_folds=5, shuffle=True, random_state=config.SEED)
cv = cv_utils.SimpleCV(logger, skf, clf, nb_class, eval_func=list_metrics, reshape_func=None)
diclogs = cv.run_cv(X_cv_text, Y_cv, X_val_text, Y_val)
# diclogs = cv.run_cv([X_cv_cat, X_cv_text], Y_cv, [X_val_cat, X_val_text], Y_val)


# diclogs['features_cat'] = features_cat
diclogs['features_word'] = features_word
diclogs['feature_importance'] = cv_utils.get_FI(cv.clf, features)
# diclogs['mapping_label'] = config.MAPPING_LABEL
diclogs['mapping_label'] = {'refused':0, 'approved':1}
# diclogs['params'] = clf.model.get_config()


preds_train = diclogs['folds_pred_train']
labels_train = diclogs['folds_label_train']

preds_test = diclogs['blend_X_train']
preds_val = diclogs['blend_X_val']

label_test = diclogs['blend_Y_train']
label_val = diclogs['blend_Y_val']

pd_cv = dic['pd_dataset'].iloc[dic['cv_idx']].reset_index(drop=True)
pd_val = dic['pd_dataset'].iloc[dic['val_idx']].reset_index(drop=True)


##### Gather misprediction #####
# def get_mismatch(predprob, labels, nb_class, pd_dataset):
#     assert len(predprob) == len(labels) == len(pd_dataset)
#
#     preds = predprob.argmax(-1).ravel()
#     labels = labels.astype(int).ravel()
#     confidence = confidence_utils.compute_confidence(predprob, nb_class)
#
#     index_mismatch = preds != labels
#
#     pd_mismatch = pd_dataset.iloc[index_mismatch].copy()
#     pd_mismatch['prediction'] = preds[index_mismatch]
#     pd_mismatch['label'] = labels[index_mismatch]
#     pd_mismatch['confidence'] = confidence[index_mismatch]
#
#     return pd_mismatch
# pd_mismatch = get_mismatch(preds_val, label_val, nb_class, pd_val)
# cols2remove = ['pet_vaccine', 'pet_type', 'location_address',
#                'housing_energy', 'house_type', 'house_area', 'pet_kennel', 'number_of_rooms',
#                'housing_efficiency']
# pd_mismatch.drop(cols2remove, 1, inplace=True)


##### Gather scores #####
train_acc = compute_metric(labels_train, preds_train, metrics.accuracy_score, round=True)
test_acc = metrics.accuracy_score(label_test, preds_test.argmax(-1))
val_acc = metrics.accuracy_score(label_val, preds_val.argmax(-1))

train_auc = compute_metric(labels_train, preds_train, metric_utils.multiclass_auc)
test_auc = metric_utils.multiclass_auc(label_test, preds_test)
val_auc = metric_utils.multiclass_auc(label_val, preds_val)

filename = "{}_{}_nbfeat{}_acc({:.3f}-{:.3f}-{:.3f})_auc({:.3f}-{:.3f}-{:.3f})-{}"\
    .format(feature_type, clf_name, len(features),
            train_acc, test_acc, val_acc,
            train_auc, test_auc, val_auc,
            time_utils._timestamp_pretty())
logger.info(filename)

##### Saving #####
pkl_filepath = os.path.join(config.DICLOG_FOLDER, filename + config.MODEL_SUFFIX)
csv_filepath = os.path.join(config.DICLOG_FOLDER, filename + "_wrong.csv")
print(pkl_filepath)


pkl_utils._save(pkl_filepath, diclogs)
# pd_mismatch.to_csv(csv_filepath, index=None)

