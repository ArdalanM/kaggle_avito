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

import numpy as np
import matplotlib
import xgboost as xgb

from sklearn import metrics

matplotlib.use('Agg')

def xgb_acc(ypred, dtrain):
        ytrue = dtrain.get_label().astype(int)
        ypred = ypred.argmax(-1)
        return 'acc', -metrics.accuracy_score(ytrue, ypred)

def xgb_auc(ypred, dtrain):
    ytrue = dtrain.get_label().astype(int)
    ypred = ypred[:,1]
    return 'auc', -metrics.roc_auc_score(ytrue, ypred)

class XGBClassifier:
    def __init__(self, num_class=2, booster='gbtree', base_score=0., colsample_bylevel=1.,
                colsample_bytree=1., gamma=0., learning_rate=0.1, max_delta_step=0.,
                max_depth=6, min_child_weight=1., missing=None, n_estimators=100,
                nthread=1, objective='multi:softprob', reg_alpha=1., reg_lambda=0.,
                reg_lambda_bias=0., seed=0, silent=True, subsample=1., feval="rmse",
                early_stopping_rounds=10, verbose=1, scale_pos_weight=0):
        self.param = {
            "objective": objective,
            "booster": booster,
            "eta": learning_rate,
            "max_depth": max_depth,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "alpha": reg_alpha,
            "lambda": reg_lambda,
            "lambda_bias": reg_lambda_bias,
            "seed": seed,
            "silent": 1 if silent else 0,
            "nthread": nthread,
            "max_delta_step": max_delta_step,
            "num_class": num_class,
            'scale_pos_weight': scale_pos_weight
        }
        self.missing = missing if missing is not None else np.nan
        self.n_estimators = n_estimators
        self.base_score = base_score
        self.num_class = num_class
        self.feval = feval
        self.verbose_eval = verbose
        self.early_stopping_rounds = early_stopping_rounds

    def __str__(self):
        return "XGBC"

    def __repr__(self):
        return ("%s(num_class=%d, booster=\'%s\', base_score=%f, colsample_bylevel=%f, \n"
                    "colsample_bytree=%f, gamma=%f, learning_rate=%f, max_delta_step=%f, \n"
                    "max_depth=%d, min_child_weight=%f, missing=\'%s\', n_estimators=%d, \n"
                    "nthread=%d, objective=\'%s\', reg_alpha=%f, reg_lambda=%f, \n"
                    "reg_lambda_bias=%f, seed=%d, silent=%d, subsample=%f)" % (
                    self.__class__.__name__,
                    self.num_class,
                    self.param["booster"],
                    self.base_score,
                    self.param["colsample_bylevel"],
                    self.param["colsample_bytree"],
                    self.param["gamma"],
                    self.param["eta"],
                    self.param["max_delta_step"],
                    self.param["max_depth"],
                    self.param["min_child_weight"],
                    str(self.missing),
                    self.n_estimators,
                    self.param["nthread"],
                    self.param["objective"],
                    self.param["alpha"],
                    self.param["lambda"],
                    self.param["lambda_bias"],
                    self.param["seed"],
                    self.param["silent"],
                    self.param["subsample"],
                ))

    def fit(self, X, y, feature_names=None, eval_set=None):

        if eval_set is not None:
            evals = list(xgb.DMatrix(x[0], label=x[1], missing=self.missing, feature_names=feature_names)
                         for x in eval_set)
            evals = list(zip(evals, ["validation_{}".format(i) for i in range(len(evals))]))
        else:
            evals = ()

        data = xgb.DMatrix(X, label=y, missing=self.missing, feature_names=feature_names)
        data.set_base_margin(self.base_score*np.ones(X.shape[0] * self.num_class))
        self.model = xgb.train(self.param, data, self.n_estimators, evals, feval=self.feval,
                               verbose_eval=self.verbose_eval,  early_stopping_rounds=self.early_stopping_rounds)
        return self

    def predict_proba(self, X, feature_names=None, evals=None):
        data = xgb.DMatrix(X, missing=self.missing, feature_names=feature_names)
        data.set_base_margin(self.base_score*np.ones(X.shape[0] * self.num_class))

        if self.param['booster'] == 'gbtree':
            proba = self.model.predict(data,  ntree_limit=self.model.best_ntree_limit)
        else:
            proba = self.model.predict(data)

        proba = proba.reshape(X.shape[0], self.num_class)
        return proba

    def predict(self, X, feature_names=None):
        proba = self.predict_proba(X, feature_names=feature_names)
        y_pred = np.argmax(proba, axis=1)
        return y_pred





class XXGBClassifier:
    def __init__(self, num_class=2, booster='gbtree', base_score=0., colsample_bylevel=1.,
                colsample_bytree=1., gamma=0., learning_rate=0.1, max_delta_step=0.,
                max_depth=6, min_child_weight=1., missing=None, n_estimators=100,
                nthread=1, objective='multi:softprob', reg_alpha=1., reg_lambda=0.,
                reg_lambda_bias=0., seed=0, silent=True, subsample=1.):
        self.param = {
            "objective": objective,
            "booster": booster,
            "eta": learning_rate,
            "max_depth": max_depth,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "alpha": reg_alpha,
            "lambda": reg_lambda,
            "lambda_bias": reg_lambda_bias,
            "seed": seed,
            "silent": 1 if silent else 0,
            "nthread": nthread,
            "max_delta_step": max_delta_step,
            "num_class": num_class,
        }
        self.missing = missing if missing is not None else np.nan
        self.n_estimators = n_estimators
        self.base_score = base_score
        self.num_class = num_class

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return ("%s(num_class=%d, booster=\'%s\', base_score=%f, colsample_bylevel=%f, \n"
                    "colsample_bytree=%f, gamma=%f, learning_rate=%f, max_delta_step=%f, \n"
                    "max_depth=%d, min_child_weight=%f, missing=\'%s\', n_estimators=%d, \n"
                    "nthread=%d, objective=\'%s\', reg_alpha=%f, reg_lambda=%f, \n"
                    "reg_lambda_bias=%f, seed=%d, silent=%d, subsample=%f)" % (
                    self.__class__.__name__,
                    self.num_class,
                    self.param["booster"],
                    self.base_score,
                    self.param["colsample_bylevel"],
                    self.param["colsample_bytree"],
                    self.param["gamma"],
                    self.param["eta"],
                    self.param["max_delta_step"],
                    self.param["max_depth"],
                    self.param["min_child_weight"],
                    str(self.missing),
                    self.n_estimators,
                    self.param["nthread"],
                    self.param["objective"],
                    self.param["alpha"],
                    self.param["lambda"],
                    self.param["lambda_bias"],
                    self.param["seed"],
                    self.param["silent"],
                    self.param["subsample"],
                ))

    def fit(self, X, y, feature_names=None):
        data = xgb.DMatrix(X, label=y, missing=self.missing, feature_names=feature_names)
        data.set_base_margin(self.base_score*np.ones(X.shape[0] * self.num_class))
        self.model = xgb.train(self.param, data, self.n_estimators)
        return self

    def predict_proba(self, X, feature_names=None):
        data = xgb.DMatrix(X, missing=self.missing, feature_names=feature_names)
        data.set_base_margin(self.base_score*np.ones(X.shape[0] * self.num_class))
        proba = self.model.predict(data)
        proba = proba.reshape(X.shape[0], self.num_class)
        return proba

    def predict(self, X, feature_names=None):
        proba = self.predict_proba(X, feature_names=feature_names)
        y_pred = np.argmax(proba, axis=1)
        return y_pred

    def plot_importance(self):
        ax = xgb.plot_importance(self.model)
        self.save_topn_features()
        return ax

    def save_topn_features(self, fname="XGBClassifier_topn_features.txt", topn=10):
        ax = xgb.plot_importance(self.model)
        yticklabels = ax.get_yticklabels()[::-1]
        if topn == -1:
            topn = len(yticklabels)
        else:
            topn = min(topn, len(yticklabels))
        with open(fname, "w") as f:
            for i in range(topn):
                f.write("%s\n"%yticklabels[i].get_text())

