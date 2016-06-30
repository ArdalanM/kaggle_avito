# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for scikit-learn models

"""

import numpy as np


from sklearn import neighbors, ensemble, linear_model, tree, pipeline, preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

class CLFFingerprint(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, index=14):
        """
        Called when initializing the classifier
        """
        self.index = index

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        self.nb_class = len(set(y))

        return self

    def predict(self, X, y=None):

        output = 0.5 * np.ones((X.shape[0], 2))

        for i, x in enumerate(X[:, self.index]):
            if x > 0:
                output[i, 0] = 0.9999999
                output[i, 1] = 1 - output[i, 0]

        return output

    def predict_proba(self, X, y=None):
        return self.predict(X)


class CLFfromPipeline():
    def __init__(self, clf=None):
        self.clf = clf

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict_proba(self, X):
        return self.clf.transform(X)

