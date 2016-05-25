__author__ = 'Ardalan'

# IT IS OFFICIAL: EVGENY DO NOT KNOW HOW TO USE GITHUB :)

FOLDER = "/home/ardalan/Documents/kaggle/avito/"
SAVE_FOLDER = FOLDER + "/diclogs/"

import theano.sandbox.cuda

theano.sandbox.cuda.use("cpu")

import os, zipfile, pickle, operator, copy
import pandas as pd
import numpy as np
import keras.backend as K

from xgboost import XGBModel
from sklearn import *
from sklearn.cross_validation import KFold, StratifiedKFold
from keras import optimizers
from keras.models import Sequential
from keras.utils import np_utils

from keras.layers import core

# from ml_metrics import auc


class CVutils():
    def reshapePrediction(self, y):

        assert type(y) == list or type(y) == np.ndarray

        if type(y) == list:
            y = np.array(y)
        else:
            if len(y.shape) > 1:
                if y.shape[1] == 1: y = y[:, 0]
                if y.shape[1] == 2: y = y[:, 1]

        y = self._clipProba(y)
        return y

    def printResults(self, dic_logs):
        l_train_logloss = dic_logs['train_error']
        l_val_logloss = dic_logs['val_error']

        string = ("{0:.4f}-{1:.4f}".format(np.mean(l_train_logloss), np.mean(l_val_logloss)))
        return string

    def dumpPickleSecure(self, dic_logs, filename):

        if os.path.exists(filename):
            print('file exist !')
            raise BrokenPipeError
        else:
            pickle.dump(dic_logs, open(filename, 'wb'))
        return

    def confusion_matrix(self, y_true, y_pred):

        return metrics.confusion_matrix(y_true, y_pred)

    def eval_func(self, ytrue, ypredproba):

        # ytrue = np_utils.to_categorical(ytrue, 2)
        return metrics.roc_auc_score(ytrue, ypredproba)

    def xgb_eval_func(self, ypred, dtrain):
        ytrue = dtrain.get_label().astype(int)
        ypred = self._clipProba(ypred)
        return 'auc', -self.eval_func(ytrue, ypred)

    def _clipProba(self, ypredproba):
        """
        Taking list of proba and returning a list of clipped proba
        :param ypredproba:
        :return: ypredproba clipped
        """""

        ypredproba = np.where(ypredproba <= 0., 0 + 1e-5, ypredproba)
        ypredproba = np.where(ypredproba >= 1., 1 - 1e-5, ypredproba)

        return ypredproba

    def saveDicLogs(self, dic_logs, filename):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(dic_logs, f, protocol=pickle.HIGHEST_PROTOCOL)
        except FileNotFoundError:
            pass


class LoadingDatasets():
    def __init__(self):
        pass

    def loadFileinZipFile(self, zip_filename, dtypes=None, parsedate=None, password=None, **kvargs):
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

    def LoadParseData(self, filename):

        data_name = filename.split('.')[0]
        pd_data = pd.read_hdf(FOLDER + "data/" + filename)
        cols_features = pd_data.drop(['isDuplicate', 'id'], 1).columns.tolist()

        pd_train = pd_data[pd_data['isDuplicate'] >= 0]
        pd_test = pd_data[pd_data['isDuplicate'].isnull()]

        Y = pd_train['isDuplicate'].values.astype(int)
        test_idx = pd_test['id'].values.astype(int)

        X = np.array(pd_train.drop(['isDuplicate', 'id'], 1))
        X_test = np.array(pd_test.drop(['isDuplicate', 'id'], 1))

        return X, Y, X_test, test_idx, pd_data, data_name, cols_features


# General params
STORE = True
n_folds = 5
nthread = 12
model_seed = 456
cv_seed = 123


X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadingDatasets().LoadParseData('D1_with_adspecific_features_22may.p')
D0 = (X, Y, X_test, test_idx, data_name, col_feats)

ratio = 0.5
xtrain = X[:ratio*len(X)]
ytrain = Y[:ratio*len(X)]
xtest = X[ratio*len(X)+100000:]
ytest = Y[ratio*len(X)+100000:]

from xgboost import XGBClassifier
clf = XGBClassifier(max_depth=8, learning_rate=0.1, n_estimators=200, objective="binary:logistic",
                    nthread=nthread, seed=model_seed)
clf.fit(xtrain, ytrain, eval_set=[(xtest, ytest)], eval_metric='auc')

y_pred = clf.predict_proba(X_test)
y_submission = y_pred[:, 1]
pd_submission = pd.DataFrame({'id': test_idx, 'probability': y_submission})
pd_submission.to_csv(SAVE_FOLDER + "submission.csv", index=None)




generic_tree_params = {'n_jobs': nthread, 'random_state': model_seed, 'n_estimators': 5}

tree_cla1 = {'max_features': 50, 'criterion': 'entropy', 'max_depth': 5, 'class_weight': 'balanced'}
tree_cla1.update(generic_tree_params)

tree_reg1 = {'max_features': 50, 'criterion': 'mse', 'max_depth': 5}
tree_reg1.update(generic_tree_params)

generic_xgb_params = {'n_estimators': 400, 'nthread': nthread,
                      'seed': model_seed, 'early_stopping_rounds': 100, 'verbose': True}

xgb_reg1 = {'objective': 'reg:linear', 'max_depth': 5,
            'learning_rate': 0.01}
xgb_reg1.update(generic_xgb_params)

xgb_cla1 = {'objective': 'binary:logistic', 'max_depth': 8,
            'learning_rate': 0.2, 'subsample': 0.9}
xgb_cla1.update(generic_xgb_params)

xgb_cla2 = {'objective': 'binary:logistic', 'max_depth': 7,
            'learning_rate': 0.01}
xgb_cla2.update(generic_xgb_params)

xgb_poi1 = {'objective': 'count:poisson', 'max_depth': 5,
            'learning_rate': 0.01}
xgb_poi1.update(generic_xgb_params)

utils = CVutils()


def lr_function(epoch):
    initial_lr = 0.01
    coef = (int(epoch / 10) + 1)
    return initial_lr / coef


params_mlp = {'batch_size': 256, 'nb_epoch': 30, 'verbose': 2, 'metrics': ["accuracy"],
              'callbacks': [
                  # callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min'),
                  # callbacks.LearningRateScheduler(lr_function),
              ],
              'shuffle': True, 'rebuild': True}


clfs = [
    # (D2, MLP(class_weight={0: 1, 1: 30}, **params_mlp)),

    (D0, XGB(eval_metric=utils.xgb_eval_func, **xgb_cla1),
    KFold(len(D0[1]), n_folds=n_folds, random_state=cv_seed)),
]

def train_and_predict(xtrain, ytrain, xval, yval, X_test, clf):
    assert hasattr(clf, 'fit_cv')
    assert hasattr(clf, 'predict_proba') or hasattr(clf, 'predict')

    clf.fit_cv(xtrain, ytrain, eval_set=(xval, yval))

    if hasattr(clf, 'predict_proba'):

        train_pred_prob = clf.predict_proba(xtrain)
        val_pred_prob = clf.predict_proba(xval)
        test_pred_prob = clf.predict_proba(X_test)

    elif hasattr(clf, 'predict'):
        train_pred_prob = clf.predict(xtrain)
        val_pred_prob = clf.predict(xval)
        test_pred_prob = clf.predict(X_test)

    # reshaping and clipping
    train_pred_prob = utils.reshapePrediction(train_pred_prob)
    val_pred_prob = utils.reshapePrediction(val_pred_prob)
    test_pred_prob = utils.reshapePrediction(test_pred_prob)

    # getting intergers
    train_pred = np.round(train_pred_prob)
    val_pred = np.round(val_pred_prob)
    test_pred = np.round(test_pred_prob)

    return train_pred_prob, train_pred, \
           val_pred_prob, val_pred, \
           test_pred_prob, test_pred

# D1 = (X, Y, X_test, test_idx, data_name, col_feats)


for clf_indice, data_clf in enumerate(clfs):
    print('-' * 50)
    print("Classifier [%i]" % clf_indice)

    X = data_clf[0][0]
    Y = data_clf[0][1]
    X_test = data_clf[0][2]
    test_idx = data_clf[0][3]
    data_name = data_clf[0][4]
    col_feats = data_clf[0][5]
    clf = data_clf[1]
    skf = data_clf[2]
    print(clf)

    clf_name = clf.__class__.__name__
    clf_name_short = clf_name[:3]

    blend_X = np.zeros((len(X), 1))
    blend_X_test = np.zeros((len(X_test), 1))
    blend_X_test_fold = np.zeros((len(X_test), len(skf)))

    dic_logs = {'name': clf_name, 'feat_importance': None,
                'blend_X': blend_X, 'blend_Y': Y,
                'blend_X_test': blend_X_test, 'test_idx': test_idx,
                'params': None, 'prepro': None, 'best_epoch': [], 'best_val_metric': [],
                'train_error': [], 'val_error': []}

    filename = '{}_{}_{}f_CV{}'.format(clf_name_short, data_name, X.shape[1], n_folds)

    for fold_indice, (train_indices, val_indices) in enumerate(skf):
        print("Fold [%i]" % fold_indice)
        xtrain = X[train_indices, :]
        ytrain = Y[train_indices]
        xval = X[val_indices, :]
        yval = Y[val_indices]
        train_pred_prob, train_pred, \
        val_pred_prob, val_pred, \
        test_pred_prob, test_pred = train_and_predict(xtrain, ytrain, xval, yval, X_test, clf)

        # metrics
        train_error = utils.eval_func(ytrain, train_pred_prob)
        val_error = utils.eval_func(yval, val_pred_prob)

        # filling blend data sets
        blend_X_test_fold[:, fold_indice] = test_pred_prob

        print("train/val error: [{0:.4f}|{1:.4f}]".format(train_error, val_error))
        print(utils.confusion_matrix(yval, val_pred))

        dic_logs['blend_X'][val_indices, 0] = val_pred_prob
        dic_logs['train_error'].append(train_error)
        dic_logs['val_error'].append(val_error)
        dic_logs['params'] = clf.get_params()
        dic_logs['best_epoch'].append(clf.get_best_epoch())
        dic_logs['best_val_metric'].append(clf.get_best_val_metric())
        dic_logs['feat_importance'] = clf.get_FI(col_feats)

    test_pred_prob = np.mean(blend_X_test_fold, axis=1)
    dic_logs['blend_X_test'][:, 0] = test_pred_prob

    filename += "{}_{}".format(clf.get_string_params(), utils.printResults(dic_logs))
    print(filename)

    if STORE:
        utils.saveDicLogs(dic_logs, SAVE_FOLDER + filename + '.p')

    # submission
    y_pred = clf.predict_proba(X_test)
    pd_submission = pd.DataFrame({'id': test_idx, 'probability': test_pred_prob})
    pd_submission.to_csv(SAVE_FOLDER + filename + ".csv", index=None)





# tc = pd.DataFrame(X_test, columns=col_feats)
#
#
# preds = pd.read_csv(SAVE_FOLDER + "XGB_D1_316f_CV10_bin-log_md5_lr0.02_csb1_esr300_0.8816-0.8415_0.839062.csv")
#
# nv = tc['num_var33']+tc['saldo_medio_var33_ult3']+tc['saldo_medio_var44_hace2']+tc['saldo_medio_var44_hace3']+\
#      tc['saldo_medio_var33_ult1']+tc['saldo_medio_var44_ult1']
#
#
# preds.loc[nv > 0, 'TARGET'] = 0
# preds.loc[tc['var15'] < 23, 'TARGET'] = 0
# preds.loc[tc['saldo_medio_var5_hace2'] > 160000, 'TARGET'] = 0
# preds.loc[tc['saldo_var33'] > 0, 'TARGET'] = 0
# preds.loc[tc['var38'] > 3988596, 'TARGET'] = 0
# preds.loc[tc['var21'] > 7500, 'TARGET'] = 0
# preds.loc[tc['num_var30'] > 9, 'TARGET'] = 0
# preds.loc[tc['num_var13_0'] > 6, 'TARGET'] = 0
# preds.loc[tc['num_var33_0'] > 0, 'TARGET'] = 0
# preds.loc[tc['imp_ent_var16_ult1'] > 51003, 'TARGET'] = 0
# preds.loc[tc['imp_op_var39_comer_ult3'] > 13184, 'TARGET'] = 0
# preds.loc[tc['saldo_medio_var5_ult3'] > 108251, 'TARGET'] = 0

