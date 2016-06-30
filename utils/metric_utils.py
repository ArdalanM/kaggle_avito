# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>

@copyright: Copyright (c) 2016, ioSquare SAS. All rights reserved.
The information contained in this file is confidential and proprietary.
Any reproduction, use or disclosure, in whole or in part, of this
information without the express, prior written consent of ioSquare SAS
is strictly prohibited.

@brief: Machine Learning metrics
"""

from sklearn import metrics
import numpy as np


def accuracy(y_pred, y_score):
    """
    Accuracy
    :param y_pred:
    :param y_score:
    :return:
    """

    if len(y_score.shape) > 1:
        score = metrics.accuracy_score(y_pred, y_score.argmax(-1))
    else:
        score = metrics.accuracy_score(y_pred, y_score)

    return score

def auc(ytrue, ypred):
    """
    Area Under Curve (AUC)
    :param ytrue:
    :param ypred:
    :return:
    """
    if len(ypred.shape) > 1 and ypred.shape[1] == 2:
        score = metrics.roc_auc_score(ytrue, ypred[:, 1])
    else:
        score = metrics.accuracy_score(ytrue, ypred)

    return score

def pres0(ytrue, ypred):
    return metrics.precision_score(ytrue, ypred, pos_label=0)

def pres1(ytrue, ypred):
    return metrics.precision_score(ytrue, ypred, pos_label=1)

def f1_class1(ytrue, ypred):
    return metrics.f1_score(ytrue, ypred, pos_label=1)

def multiclass_auc(y_true, y_predprob):
    mean_auc = []
    for i in range(y_predprob.shape[1]):
        y_true_class = 1 * (y_true == i)

        if np.sum(y_true_class) > 0:
            y_score_class = y_predprob[:, i]
            mean_auc.append(metrics.roc_auc_score(y_true_class, y_score_class))
    return np.mean(mean_auc)
