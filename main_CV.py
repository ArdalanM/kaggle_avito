__author__ = 'Ardalan'

CODE_FOLDER = "/home/ardalan/Documents/kaggle/bnp/"
# CODE_FOLDER = "/home/arda/Documents/kaggle/bnp/"
SAVE_FOLDER = CODE_FOLDER + "/diclogs/"

import theano.sandbox.cuda
theano.sandbox.cuda.use("cpu")

import os, sys, time, re, zipfile, pickle, operator, copy
import pandas as pd
import numpy as np

from xgboost import XGBClassifier, XGBRegressor, XGBModel
from sklearn import metrics, cross_validation, linear_model,\
    ensemble, cluster, calibration, preprocessing

from keras import optimizers, callbacks
from keras.models import Sequential
from keras.utils import np_utils

from keras.layers import core, embeddings, recurrent, advanced_activations, normalization

class CVutils():

    def reshapePrediction(self, ypredproba):
        result = None
        if len(ypredproba.shape) > 1:
            if ypredproba.shape[1] == 1: result = ypredproba[:, 0]
            if ypredproba.shape[1] == 2: result = ypredproba[:, 1]
        else:
            result = ypredproba.ravel()

        result = self._clipProba(result)
        return result

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
        return metrics.log_loss(ytrue, ypredproba)

    def xgb_eval_func(self, ypred, dtrain):
        ytrue = dtrain.get_label().astype(int)
        ypred = self._clipProba(ypred)
        return 'logloss', self.eval_func(ytrue, ypred)

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

    def loadFileinZipFile(self, zip_filename, dtypes=None, parsedate = None, password=None, **kvargs):
        """
        Load file to dataframe.
        """
        with zipfile.ZipFile(zip_filename, 'r') as myzip:
            if password:
                myzip.setpassword(password)

            inside_zip_filename = myzip.filelist[0].filename

            if parsedate:
                pd_data = pd.read_csv(myzip.open(inside_zip_filename), sep=',', parse_dates=parsedate, dtype=dtypes, **kvargs)
            else:
                pd_data = pd.read_csv(myzip.open(inside_zip_filename), sep=',', dtype=dtypes, **kvargs)
            return pd_data, inside_zip_filename

    def LoadParseData(self, filename):

        data_name = filename.split('_')[0]
        pd_data = pd.read_hdf(CODE_FOLDER + "data/" + filename)
        cols_features = pd_data.drop(['ID', 'target'], 1).columns.tolist()

        pd_train = pd_data[pd_data.target >= 0]
        pd_test = pd_data[pd_data.target == -1]

        Y = pd_train['target'].values.astype(int)
        # nb_class = len(np.unique(Y))
        # Y = np_utils.to_categorical(Y, nb_class)


        test_idx = pd_test['ID'].values.astype(int)

        X = np.array(pd_train.drop(['ID', 'target'],1))
        X_test = np.array(pd_test.drop(['ID','target'], 1))

        return X, Y, X_test, test_idx, pd_data, data_name, cols_features

class genericSKLCLF():

    def fit_cv(self, X, y, eval_set=(), class_weight=None):
        return self.fit(X, y, sample_weight=class_weight)

    def get_best_epoch(self):
        pass

    def get_best_val_metric(self):
        pass

    def get_FI(self, list_feat):
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
        importance = [(name,score) for name,score in zip(list_feats, self.feature_importances_)]
        importance = sorted(importance, key=operator.itemgetter(1), reverse=True)
        return pd.DataFrame(importance)

    def get_best_epoch(self):
        pass

    def get_best_val_metric(self):
        pass

class genericKerasCLF(genericSKLCLF):

    def __init__(self, batch_size=128, nb_epoch=2, verbose=1, callbacks=[],
                 shuffle=True, metrics=[],
                 rebuild=True, input_dim=123,  input_length=123):

        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.callbacks = callbacks
        self.shuffle = shuffle
        self.metrics = metrics
        self.rebuild = rebuild

        self.input_dim = input_dim
        self.input_length = input_length

        self.model = self.build_model()
        self.logs = None

    def build_model(self):
        pass

    def fit_cv(self, X, y, eval_set=(), class_weight=None):

        #handling multicolumn label
        y = np_utils.to_categorical(y, len(np.unique(Y)))
        eval_set = (eval_set[0], np_utils.to_categorical(eval_set[1], len(np.unique(eval_set[1]))))

        #handling sparse input
        if type(X).__name__ == "csr_matrix":
            eval_set = (eval_set[0].toarray(), eval_set[1])
            X = X.toarray()

        if self.rebuild:
            self.model = self.build_model()
        if len(eval_set) > 0:
            logs = self.model.fit(X, y, batch_size=self.batch_size, nb_epoch=self.nb_epoch,
                                  validation_data=(eval_set[0], eval_set[1]),
                                  verbose=self.verbose, shuffle=self.shuffle, callbacks=copy.deepcopy(self.callbacks),
                                  class_weight=class_weight)
        else:
            logs = self.model.fit(X, y, batch_size=self.batch_size, nb_epoch=self.nb_epoch,
                                  verbose=self.verbose, shuffle=self.shuffle,
                                  callbacks=self.callbacks, class_weight=class_weight)

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

    def get_string_params(self):
        pass

    def keras_class_weight(self, Y_vec):
        # Find the weight of each class as present in y.
        # inversely proportional to the number of samples in the class
        recip_freq = 1. / np.bincount(Y_vec)
        weight = recip_freq / np.mean(recip_freq)
        dic_w = {index:weight_value for index, weight_value in enumerate(weight)}
        return dic_w

class genericXGB(genericSKLCLF):

    def __init__(self,eval_metric="logloss", early_stopping_rounds=300, verbose=True,
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
                        "_esr{}".format(self.esr)]
        return "".join(added_params)

    def get_FI(self, col_features):

        dic_fi = self.model._Booster.get_fscore()

        #getting somithing like [(f0, score), (f1, score)]
        importance = [(col_features[int(key[1:])], dic_fi[key]) for key in dic_fi]

        #same but sorted by score
        importance = sorted(importance, key=operator.itemgetter(1), reverse=True)
        sum_importance = np.sum([score for feat, score in importance])
        importance = [(name, score/sum_importance) for name, score in importance]

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
        model.add(core.Dense(512, init='normal',input_shape=(self.input_dim,)))
        model.add(core.Dropout(0.5))
        model.add(core.Activation('relu'))
        model.add(core.Dense(512, init='normal'))
        model.add(core.Dropout(0.5))
        model.add(core.Activation('relu'))
        # model.add(core.Dense(1024, init='normal'))
        # model.add(core.Dropout(0.5))
        # model.add(core.Activation('relu'))
        model.add(core.Dense(2))
        model.add(core.Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=.01, momentum=0.9, decay=0.0001),
                      metrics=self.metrics)
        return model

    def __repr__(self):
        return "mlp"



# General params
STORE = True
n_folds = 5
nthread = 12
seed = 123


# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadingDatasets().LoadParseData('D1_[LE-cat]_[NAmean]_[v22v56v125].p')
# X = preprocessing.StandardScaler().fit_transform(X) ; X_test = preprocessing.StandardScaler().fit_transform(X_test)
# D1 = (X, Y, X_test, test_idx, data_name, col_feats)
#
# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadingDatasets().LoadParseData('D2_[LE-cat]_[NA-999]_[v22v56v125].p')
# X = preprocessing.StandardScaler().fit_transform(X) ; X_test = preprocessing.StandardScaler().fit_transform(X_test)
# D2 = (X, Y, X_test, test_idx, data_name, col_feats)

# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadingDatasets().LoadParseData('D4_[OH300]_[NA-999]_[v22v56v125].p')
# X = preprocessing.StandardScaler().fit_transform(X) ; X_test = preprocessing.StandardScaler().fit_transform(X_test)
# D3 = (X, Y, X_test, test_idx, data_name, col_feats)
#
# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadingDatasets().LoadParseData('D4_[OH300]_[NA-999]_[v22v56v125].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D4 = (X, Y, X_test, test_idx, data_name, col_feats)
#
X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadingDatasets().LoadParseData('D5_[OnlyCont]_[NAmean].p')
X = preprocessing.StandardScaler().fit_transform(X) ; X_test = preprocessing.StandardScaler().fit_transform(X_test)
D5 = (X, Y, X_test, test_idx, data_name, col_feats)
# #
# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadingDatasets().LoadParseData('D6_[OnlyCatLE].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D6 = (X, Y, X_test, test_idx, data_name, col_feats)
#
# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadingDatasets().LoadParseData('D7_[OnlyCatOH].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D7 = (X, Y, X_test, test_idx, data_name, col_feats)
#
# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadingDatasets().LoadParseData('D8_[ColsRemoved]_[Namean]_[OH].p')
# X = preprocessing.StandardScaler().fit_transform(X) ; X_test = preprocessing.StandardScaler().fit_transform(X_test)
# D8 = (X, Y, X_test, test_idx, data_name, col_feats)
#
# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadingDatasets().LoadParseData('D9_[ColsRemoved]_[NA-999]_[LE-cat].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D9 = (X, Y, X_test, test_idx, data_name, col_feats)
#
# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadingDatasets().LoadParseData('D10_[ColsRemoved]_[NA-999]_[OH].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D10 = (X, Y, X_test, test_idx, data_name, col_feats)


# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadingDatasets().LoadParseData('D11_[OH1000]_[NA-999].p')
# X = StandardScaler().fit_transform(X) ; X_test = StandardScaler().fit_transform(X_test)
# D11 = (X, Y, X_test, test_idx, data_name, col_feats)


# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadingDatasets().LoadParseData('D12_[2ways]_[NA-999].p')
# X = preprocessing.StandardScaler().fit_transform(X) ; X_test = preprocessing.StandardScaler().fit_transform(X_test)
# D12 = (X, Y, X_test, test_idx, data_name, col_feats)



# X, Y, X_test, test_idx, pd_data, data_name, col_feats = LoadingDatasets().LoadParseData('D13_[2ways]_[v22v56v125]_[NA-999].p')
# X = preprocessing.StandardScaler().fit_transform(X) ; X_test = preprocessing.StandardScaler().fit_transform(X_test)
# D13 = (X, Y, X_test, test_idx, data_name, col_feats)









def models():

    et_params_kaggle_cla = {'max_features': 50, 'criterion': 'entropy',
                            'min_samples_split': 4,'n_estimators':1200,
                            'max_depth':35, 'min_samples_leaf':2,
                            'n_jobs':nthread, 'random_state':seed}

    et_params_kaggle_reg = {'max_features': 50, 'criterion': 'mse',
                            'min_samples_split': 4,'n_estimators':1200,
                            'max_depth':35, 'min_samples_leaf':2,
                            'n_jobs':nthread, 'random_state':seed}


    et_params = {'n_estimators':1200,'n_jobs':nthread, 'random_state':seed}

    generic_xgb_params = {'n_estimators':10000, 'nthread':nthread,
                          'seed':seed, 'early_stopping_rounds': 300, 'verbose':False}

    xgb_reg1 = {'objective':'reg:linear', 'max_depth': 5,
               'learning_rate':0.01} ; xgb_reg1.update(generic_xgb_params)

    xgb_reg2 = {'objective':'reg:linear', 'max_depth': 11,
               'learning_rate':0.01} ; xgb_reg2.update(generic_xgb_params)

    xgb_reg3 = {'objective':'reg:linear', 'max_depth': 16,
               'learning_rate':0.01} ; xgb_reg3.update(generic_xgb_params)

    xgb_cla1 = {'objective':'binary:logistic', 'max_depth': 5,
               'learning_rate':0.01} ; xgb_cla1.update(generic_xgb_params)

    xgb_cla2 = {'objective':'binary:logistic', 'max_depth': 11,
               'learning_rate':0.01} ; xgb_cla2.update(generic_xgb_params)

    xgb_cla3 = {'objective':'binary:logistic', 'max_depth': 16,
               'learning_rate':0.01} ; xgb_cla3.update(generic_xgb_params)

    xgb_poi1 = {'objective':'count:poisson', 'max_depth': 5,
               'learning_rate':0.01} ; xgb_poi1.update(generic_xgb_params)

    xgb_poi2 = {'objective':'count:poisson', 'max_depth': 11,
               'learning_rate':0.01} ; xgb_poi2.update(generic_xgb_params)

    xgb_poi3 = {'objective':'count:poisson', 'max_depth': 16,
               'learning_rate':0.01} ; xgb_poi3.update(generic_xgb_params)

    def lr_function(epoch):
        initial_lr = 0.01
        coef = (int(epoch/10) + 1)
        return initial_lr / coef

    params_mlp = {'batch_size': 128, 'nb_epoch': 100, 'verbose': 2, 'metrics': ["accuracy"],
              'callbacks':[
                  # callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min'),
                  callbacks.LearningRateScheduler(lr_function),
              ],
              'shuffle': True, 'rebuild': True}

    clfs = [
        # (D3, MLP(input_dim=D3[0].shape[1], **params_mlp)),
        (D5, LR())

        # (D5, XGB(eval_metric=utils.xgb_eval_func, **xgb_cla1)),
        # (D5, XGB(eval_metric=utils.xgb_eval_func, **xgb_cla2)),
        # (D5, XGB(eval_metric=utils.xgb_eval_func, **xgb_cla3)),

        # (D5, XGB(eval_metric=utils.xgb_eval_func, **xgb_cla1)),
        # (D5, XGB(eval_metric=utils.xgb_eval_func, **xgb_cla2)),
        # (D5, XGB(eval_metric=utils.xgb_eval_func, **xgb_cla3)),

        # (D5, ETcla(**et_params_kaggle_cla)),
        # (D5, ETreg(**et_params_kaggle_reg)),

        # (D5, ETcla(**et_params_kaggle_cla)),
        # (D5, ETreg(**et_params_kaggle_reg)),


        # (D5, RFcla(**et_params_kaggle_cla)),
        # (D5, RFreg(**et_params_kaggle_reg)),

        # (D5, RFcla(**et_params_kaggle_cla)),
        # (D5, RFreg(**et_params_kaggle_reg)),

    ]
    for clf in clfs:
        yield clf
clfs = models()
utils = CVutils()



#CV
skf = cross_validation.StratifiedKFold(Y, n_folds=n_folds, shuffle=True, random_state=seed)

for clf_indice, data_clf in enumerate(clfs):
    print('-' * 50)
    print("Classifier [%i]" % clf_indice)
    def get_data(data_clf):
        return data_clf[0][0], data_clf[0][1], data_clf[0][2],\
               data_clf[0][3], data_clf[0][4], data_clf[0][5], data_clf[1]

    X, Y, X_test, test_idx, data_name, col_feats, clf = get_data(data_clf)
    clf_name = clf.__class__.__name__
    clf_name_short = clf_name[:3]
    print(clf)

    blend_X = np.zeros((len(X), 1))
    blend_X_test = np.zeros((len(X_test), 1))
    blend_X_test_fold = np.zeros((len(X_test), len(skf)))

    dic_logs = {'name': clf_name, 'feat_importance': None,
                'blend_X': blend_X, 'blend_Y': Y, 'blend_X_test': blend_X_test, 'test_idx': test_idx,
                'params': None, 'prepro': None, 'best_epoch': [], 'best_val_metric': [],
                'train_error': [], 'val_error': []}
    filename = '{}_{}_{}f_CV{}'.format(clf_name_short, data_name, X.shape[1], n_folds)

    for fold_indice, (train_indices, val_indices) in enumerate(skf):
        print("Fold [%i]" % fold_indice)
        def get_train_test(x, y):
            return x[train_indices], y[train_indices],\
                   x[val_indices], y[val_indices]
        xtrain, ytrain, xval, yval = get_train_test(X, Y)

        clf.fit_cv(xtrain, ytrain, eval_set=(xval, yval))

        if hasattr(clf, 'predict_proba'):
            train_pred, val_pred, test_pred =\
                list(map(clf.predict_proba, [xtrain, xval, X_test]))

        elif hasattr(clf, 'predict'):
            train_pred, val_pred, test_pred =\
                list(map(clf.predict, [xtrain, xval, X_test]))

        train_pred, val_pred, test_pred =\
            list(map(utils.reshapePrediction, [train_pred, val_pred, test_pred]))

        # metrics
        train_error = utils.eval_func(ytrain, train_pred)
        val_error = utils.eval_func(yval, val_pred)

        # filling blend data sets
        blend_X_test_fold[:, fold_indice] = test_pred

        print("train/val error: [{0:.4f}|{1:.4f}]".format(train_error, val_error))
        print(utils.confusion_matrix(yval, val_pred.round()))

        dic_logs['blend_X'][val_indices, 0] = val_pred
        dic_logs['train_error'].append(train_error)
        dic_logs['val_error'].append(val_error)
        dic_logs['params'] = clf.get_params()
        dic_logs['best_epoch'].append(clf.get_best_epoch())
        dic_logs['best_val_metric'].append(clf.get_best_val_metric())
        dic_logs['feat_importance'] = clf.get_FI(col_feats)

    dic_logs['blend_X_test'][:, 0] = np.mean(blend_X_test_fold, axis=1)

    filename += "{}_{}".format(clf.get_string_params(), utils.printResults(dic_logs))
    print(filename)

    if STORE:
        utils.saveDicLogs(dic_logs, SAVE_FOLDER + filename + '.p')

    #submission
    y_test_pred = dic_logs['blend_X_test'][:, 0]

    output_filename = SAVE_FOLDER + filename + '.csv'
    np.savetxt(output_filename, np.vstack((test_idx, y_test_pred)).T,
               delimiter=',', fmt='%i,%.10f', header='ID,PredictedProb', comments="")






















