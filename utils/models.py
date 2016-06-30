"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief: ml models
"""

import operator
import copy

import numpy as np
import pandas as pd
import keras.backend as K

from keras.utils import np_utils
from xgboost import XGBModel
from sklearn import linear_model, ensemble, naive_bayes, svm, calibration


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
