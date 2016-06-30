# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>

@copyright: Copyright (c) 2016, ioSquare SAS. All rights reserved.
The information contained in this file is confidential and proprietary.
Any reproduction, use or disclosure, in whole or in part, of this
information without the express, prior written consent of ioSquare SAS
is strictly prohibited.

@brief: 
"""
import inspect
import operator
import numpy as np
from sklearn import metrics
import pandas as pd

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


            # if hasattr(self.clf, "get_params"):
            #     self.diclogs['params'] = self.clf.get_params()
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


def get_FI(clf, list_feats):
    if hasattr(clf, "model"):
        # This is XGB or NN
        if hasattr(clf.model, "get_fscore"):
            # This is xgb
            dic_fi = clf.model.get_fscore()
            # getting somithing like [(f0, score), (f1, score)]
            importance = [(list_feats[int(key[1:])], dic_fi[key]) for key in dic_fi]
            # same but sorted by score
            importance = sorted(importance, key=operator.itemgetter(1), reverse=True)
            sum_importance = np.sum([score for feat, score in importance])
            importance = [(name, score / sum_importance) for name, score in importance]
        else:
            return None

    elif hasattr(clf, "coef_"):
        pd_fi = pd.DataFrame()
        for class_index, coefs in enumerate(clf.coef_):
            dic_feature_score = [(feature, score) for feature, score in zip(list_feats, coefs)]
            importance = sorted(dic_feature_score, key=operator.itemgetter(1), reverse=True)

            pd_temp = pd.DataFrame(importance, columns=['class_{}'.format(class_index),
                                                        'score_{}'.format(class_index)])
            pd_fi = pd.concat([pd_fi, pd_temp.head(1000)], axis=1)
        return pd_fi

    elif hasattr(clf, "feature_importances_"):
        #     thie is skl forest model
        importance = [(name, score) for name, score in zip(list_feats, clf.feature_importances_)]
        importance = sorted(importance, key=operator.itemgetter(1), reverse=True)
    else:
        return None
    return pd.DataFrame(importance, columns=['feature', 'importance'])

