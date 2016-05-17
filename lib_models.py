import operator
import pandas as pd

from sklearn import ensemble

class genericSKLCLF():

    def fit_cv(self, X, y, eval_set=(),
               eval_metric="logloss", class_weight=None):
        pass

    def get_best_epoch(self):
        return None

    def get_best_val_metric(self):
        return None

class genericXGB():

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

    def fit_cv(self, X, y, eval_set=(),
               eval_metric="logloss", class_weight=None):

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

class genericSKLTree(genericSKLCLF, ensemble.RandomForestClassifier):

    def get_string_params(self):

        clf_name = self.base_estimator.__class__.__name__

        added_params = ["_{}".format(clf_name[:3]),
                        "_{}".format(self.criterion[:3]),
                        "_md{}".format(self.max_depth),
                        "_mf{}".format(self.max_features),
                        "_est{}".format(self.n_estimators)]
        return "".join(added_params)

    def fit_cv(self, X, y, eval_set=(),
               eval_metric="logloss", class_weight=None):
        return self.fit(X, y)

    def get_FI(self, list_feats):
        importance = [(name,score) for name,score in zip(list_feats, self.feature_importances_)]
        importance = sorted(importance, key=operator.itemgetter(1), reverse=True)
        return pd.DataFrame(importance)



class XGBcla(genericXGB):
    pass

class XGBreg(genericXGB):
    pass

class RFcla(genericSKLCLF, ensemble.RandomForestClassifier):

    def get_string_params(self):

        clf_name = self.base_estimator.__class__.__name__

        added_params = ["_{}".format(clf_name[:3]),
                        "_{}".format(self.criterion[:3]),
                        "_md{}".format(self.max_depth),
                        "_mf{}".format(self.max_features),
                        "_est{}".format(self.n_estimators)]
        return "".join(added_params)

    def fit_cv(self, X, y, eval_set=(),
               eval_metric="logloss", class_weight=None):
        return self.fit(X, y)

    # def get_clf_params(self):
    #     return self.get_params()

    def get_FI(self, list_feats):
        importance = [(name,score) for name,score in zip(list_feats, self.feature_importances_)]
        importance = sorted(importance, key=operator.itemgetter(1), reverse=True)
        return pd.DataFrame(importance)

class RFreg(genericSKLCLF, ensemble.RandomForestRegressor):

    def get_string_params(self):

        clf_name = self.base_estimator.__class__.__name__

        added_params = ["_{}".format(clf_name[:3]),
                        "_{}".format(self.criterion[:3]),
                        "_md{}".format(self.max_depth),
                        "_mf{}".format(self.max_features),
                        "_est{}".format(self.n_estimators)]
        return "".join(added_params)

    def fit_cv(self, X, y, eval_set=(),
               eval_metric="logloss", class_weight=None):
        return self.fit(X, y)

    # def get_clf_params(self):
    #     return self.get_params()

    def get_FI(self, list_feats):
        importance = [(name,score) for name,score in zip(list_feats, self.feature_importances_)]
        importance = sorted(importance, key=operator.itemgetter(1), reverse=True)
        return pd.DataFrame(importance)

class ETcla(genericSKLCLF, ensemble.ExtraTreesClassifier):

    def get_string_params(self):

        clf_name = self.base_estimator.__class__.__name__

        added_params = ["_{}".format(clf_name[:3]),
                        "_{}".format(self.criterion[:3]),
                        "_md{}".format(self.max_depth),
                        "_mf{}".format(self.max_features),
                        "_est{}".format(self.n_estimators)]
        return "".join(added_params)

    def fit_cv(self, X, y, eval_set=(),
               eval_metric="logloss", class_weight=None):
        return self.fit(X, y)

    # def get_clf_params(self):
    #     return self.get_params()

    def get_FI(self, list_feats):
        importance = [(name,score) for name,score in zip(list_feats, self.feature_importances_)]
        importance = sorted(importance, key=operator.itemgetter(1), reverse=True)
        return pd.DataFrame(importance)

class ETreg(genericSKLCLF, ensemble.ExtraTreesClassifier):

    def get_string_params(self):

        clf_name = self.base_estimator.__class__.__name__

        added_params = ["_{}".format(clf_name[:3]),
                        "_{}".format(clf.criterion[:3]),
                        "_md{}".format(clf.max_depth),
                        "_mf{}".format(clf.max_features),
                        "_est{}".format(clf.n_estimators)]
        return "".join(added_params)

    def fit_cv(self, X, y, eval_set=(),
               eval_metric="logloss", class_weight=None):
        return self.fit(X, y)

    # def get_clf_params(self):
    #     return self.get_params()

    def get_FI(self, list_feats):
        importance = [(name,score) for name,score in zip(list_feats, clf.feature_importances_)]
        importance = sorted(importance, key=operator.itemgetter(1), reverse=True)
        return pd.DataFrame(importance)

