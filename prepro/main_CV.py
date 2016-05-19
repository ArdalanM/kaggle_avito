__author__ = 'Ardalan'

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
from keras import optimizers
from keras.models import Sequential
from keras.utils import np_utils

from keras.layers import core

from ml_metrics import auc


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
        return auc(ytrue, ypredproba)

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

class genericSKLCLF():
    def fit_cv(self, X, y, eval_set=()):
        return self.fit(X, y)

    def get_best_epoch(self):
        pass

    def get_best_val_metric(self):
        pass

    def get_FI(self, list_feat):
        pass

    def _int_to_string(self, number):

        if number > 1e6:
            output_string = "{}Mp".format(int(np.ceil(number / 1e6)))
        elif number > 1e3:
            output_string = "{}kp".format(int(np.ceil(number / 1e3)))
        else:
            output_string = "{}p".format(number)
        return output_string

    def get_string_params(self):
        pass

class genericSKLTree(genericSKLCLF):
    def fit_cv(self, X, y, eval_set=(), class_weight=None):
        return self.fit(X, y, sample_weight=class_weight)

    def get_string_params(self):
        clf_name = self.base_estimator.__class__.__name__

        added_params = ["_{}".format(clf_name[:3]),
                        "_{}".format(self.criterion[:3]),
                        "_md{}".format(self.max_depth),
                        "_mf{}".format(self.max_features),
                        "_est{}".format(self.n_estimators)]
        return "".join(added_params)

    def get_FI(self, list_feats):
        importance = [(name, score) for name, score in zip(list_feats, self.feature_importances_)]
        importance = sorted(importance, key=operator.itemgetter(1), reverse=True)
        return pd.DataFrame(importance)

    def get_best_epoch(self):
        pass

    def get_best_val_metric(self):
        pass

class genericKerasCLF(genericSKLCLF):
    def __init__(self, batch_size=128, nb_epoch=2, verbose=1, callbacks=None,
                 shuffle=True, metrics=None, class_weight=None, rebuild=True):

        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.callbacks = callbacks
        self.shuffle = shuffle
        self.metrics = metrics
        self.rebuild = rebuild

        self.input_dim = None
        self.input_length = None

        self.model = None
        self.logs = None
        self.class_weight = class_weight

    def _set_input_dim(self, X, y, eval_set=()):

        assert len(X.shape) == 2 or len(X.shape) == 3

        if len(X.shape) == 3:
            # means we have a sequence
            self.input_dim = X.shape[2]
            self.input_length = X.shape[1]
        else:
            # means we have vectors
            self.input_dim = X.shape[1]
            # handling sparse input
            if type(X).__name__ == "csr_matrix":
                eval_set = (eval_set[0].toarray(), eval_set[1])
                X = X.toarray()

        # handling multicolumn label
        y = np_utils.to_categorical(y, len(np.unique(y)))
        eval_set = (eval_set[0], np_utils.to_categorical(eval_set[1], len(np.unique(eval_set[1]))))
        return X, y, eval_set

    def build_model(self):
        pass

    def fit_cv(self, X, y, eval_set=()):

        X, y, eval_set = self._set_input_dim(X, y, eval_set)

        if self.rebuild:
            self.model = self.build_model()
        if len(eval_set) > 0:
            logs = self.model.fit(X, y, batch_size=self.batch_size, nb_epoch=self.nb_epoch,
                                  validation_data=(eval_set[0], eval_set[1]),
                                  verbose=self.verbose, shuffle=self.shuffle,
                                  callbacks=copy.deepcopy(self.callbacks),
                                  class_weight=self.class_weight)
        else:
            logs = self.model.fit(X, y, batch_size=self.batch_size, nb_epoch=self.nb_epoch,
                                  verbose=self.verbose, shuffle=self.shuffle,
                                  callbacks=copy.deepcopy(self.callbacks), class_weight=self.class_weight)

        self.logs = logs

    def predict_proba(self, X):
        if type(X).__name__ == "csr_matrix":
            X = X.toarray()

        prediction = self.model.predict_proba(X, verbose=False)

        return prediction

    def get_params(self):
        return self.model.get_config()

    def get_best_epoch(self):
        return np.argmin(self.logs.history['val_loss'])

    def get_best_val_metric(self):
        return np.min(self.logs.history['val_loss'])

    def keras_class_weight(self, Y_vec):
        # Find the weight of each class as present in y.
        # inversely proportional to the number of samples in the class
        recip_freq = 1. / np.bincount(Y_vec)
        weight = recip_freq / np.mean(recip_freq)
        dic_w = {index: weight_value for index, weight_value in enumerate(weight)}
        return dic_w

    def get_num_params(self):
        # Compute number of params in a model (the actual number of floats)
        return sum([np.prod(K.get_value(w).shape) for w in self.model.trainable_weights])

class genericXGB(genericSKLCLF):
    def __init__(self, eval_metric="logloss", early_stopping_rounds=300, verbose=True,
                 max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="binary:logistic",
                 nthread=-1, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None):

        self.eval_metric = eval_metric
        self.esr = early_stopping_rounds
        self.verbose = verbose

        self.model = XGBModel(max_depth, learning_rate,
                              n_estimators, silent, objective,
                              nthread, gamma, min_child_weight,
                              max_delta_step, subsample,
                              colsample_bytree, colsample_bylevel,
                              reg_alpha, reg_lambda,
                              scale_pos_weight, base_score, seed, missing)

    def fit_cv(self, X, y, eval_set=(), class_weight=None):

        if len(eval_set) > 0:
            self.model.fit(X, y, eval_set=[eval_set], eval_metric=self.eval_metric,
                           early_stopping_rounds=self.esr, verbose=self.verbose)
        else:
            self.model.fit(X, y, verbose=self.verbose)

    def predict_proba(self, X):

        preds = self.model.predict(X, ntree_limit=self.model.best_ntree_limit)
        return preds

    def predict(self, X):

        preds = self.model.predict(X, ntree_limit=self.model.best_ntree_limit)
        return preds

    def get_string_params(self):
        added_params = ["_{}".format('-'.join(list(map(lambda x: x[:3], self.model.objective.split(':'))))),
                        "_md{}".format(self.model.max_depth),
                        "_lr{}".format(self.model.learning_rate),
                        "_csb{}".format(self.model.colsample_bytree),
                        "_esr{}".format(self.esr),
                        "_est{}".format(self.model.n_estimators)]
        return "".join(added_params)

    def get_FI(self, col_features):

        dic_fi = self.model._Booster.get_fscore()

        # getting somithing like [(f0, score), (f1, score)]
        importance = [(col_features[int(key[1:])], dic_fi[key]) for key in dic_fi]

        # same but sorted by score
        importance = sorted(importance, key=operator.itemgetter(1), reverse=True)
        sum_importance = np.sum([score for feat, score in importance])
        importance = [(name, score / sum_importance) for name, score in importance]

        return pd.DataFrame(importance)

    def get_params(self):
        return self.model.get_params()

    def get_best_epoch(self):
        return self.model.best_iteration

    def get_best_val_metric(self):
        return self.model.best_score


class RFcla(genericSKLTree, ensemble.RandomForestClassifier):
    pass
class RFreg(genericSKLTree, ensemble.RandomForestRegressor):
    pass
class ETcla(genericSKLTree, ensemble.ExtraTreesClassifier):
    pass
class ETreg(genericSKLTree, ensemble.ExtraTreesRegressor):
    pass
class CAL(genericSKLTree, calibration.CalibratedClassifierCV):
    def get_string_params(self):
        # sub_clf = self.base_estimator
        sub_clf_name = self.base_estimator.__class__.__name__

        added_params = ["_{}".format(sub_clf_name[:3]),
                        "_{}".format(self.method[:3]),
                        "_{}".format(self.base_estimator.criterion[:3]),
                        "_md{}".format(self.base_estimator.max_depth),
                        "_mf{}".format(self.base_estimator.max_features),
                        "_est{}".format(self.base_estimator.n_estimators)]
        return "".join(added_params)

    def get_FI(self, list_feats):
        pass
class LR(genericSKLCLF, linear_model.LogisticRegression):
    def get_string_params(self):
        clf_name = self.__class__.__name__

        added_params = ["_{}".format(clf_name[:3]),
                        "_{}".format(self.penalty),
                        "_C{}".format(self.C),
                        "_mi{}".format(self.max_iter)]
        return "".join(added_params)
class XGB(genericXGB):
    def __repr__(self):
        return str(self.get_params())
class MLP(genericKerasCLF):
    def build_model(self):
        model = Sequential()
        model.add(core.Dense(128, init='normal', input_shape=(self.input_dim,),
                             W_regularizer=None))
        model.add(core.Dropout(0.1))
        model.add(core.Activation('relu'))
        model.add(core.Dense(128, init='normal', W_regularizer=None))
        model.add(core.Dropout(0.1))
        model.add(core.Activation('relu'))
        model.add(core.Dense(2))
        model.add(core.Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=.001, momentum=0.9, decay=0.0001),
                      metrics=self.metrics)
        return model

    def __repr__(self):
        return "mlp"

    def get_string_params(self):
        l_layers_params = self.model.get_config()

        nb_params = self.get_num_params()
        nb_params_string = self._int_to_string(nb_params)

        added_params = ["{}_".format(nb_params_string)]
        #
        for i, layer in enumerate(l_layers_params[:-2]):

            if "Dense" in layer['class_name']:
                nb_neurons = layer['config']['output_dim']
                added_params.append("D{}".format(nb_neurons))

        added_params.append("_cw{}-{}".format(self.class_weight[0], self.class_weight[1]))

        return "".join(added_params)

# General params
STORE = True
n_folds = 2
nthread = 8
model_seed = 456
cv_seed = 123


X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadingDatasets().LoadParseData('D1_18may.p')
D0 = (X, Y, X_test, test_idx, data_name, col_feats)

indice = 2393116
xtrain = X[:indice]
ytrain = Y[:indice]
xtest = X[indice+1:]
ytest = Y[indice+1:]

from xgboost import XGBClassifier
clf = XGBClassifier(max_depth=8, learning_rate=0.1, n_estimators=1000, objective="binary:logistic",
                    nthread=nthread, seed=model_seed)

clf.fit(xtrain, ytrain, eval_set=[(xtest, ytest)], eval_metric='auc', early_stopping_rounds=100)



























generic_tree_params = {'n_jobs': nthread, 'random_state': model_seed, 'n_estimators': 200}

tree_cla1 = {'max_features': 50, 'criterion': 'entropy', 'max_depth': 5, 'class_weight': 'balanced'}
tree_cla1.update(generic_tree_params)

tree_reg1 = {'max_features': 50, 'criterion': 'mse', 'max_depth': 5}
tree_reg1.update(generic_tree_params)

generic_xgb_params = {'n_estimators': 1000, 'nthread': nthread,
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
    cross_validation.StratifiedShuffleSplit(D0[1], n_iter=2, test_size=0.2, train_size=None, random_state=cv_seed)),
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
    output_filename = SAVE_FOLDER + filename + '.csv'
    np.savetxt(output_filename, np.vstack((test_idx, test_pred_prob)).T,
               delimiter=',', fmt='%i,%.10f', header='id,probability', comments="")





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

