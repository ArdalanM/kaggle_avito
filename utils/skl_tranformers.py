# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>

@copyright: Copyright (c) 2016, ioSquare SAS. All rights reserved.
The information contained in this file is confidential and proprietary.
Any reproduction, use or disclosure, in whole or in part, of this
information without the express, prior written consent of ioSquare SAS
is strictly prohibited.

@brief: Transformers to use with sklearn.Pipeline
"""

import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class ColumnSelector(TransformerMixin):
    """Select columns of a DataFrame

    Parameters
    ----------
    columns: list of columns to keep

    Returns
    -------
    DataFrame : Pandas DataFrame with selected columns
    """
    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, df, **transform_params):
        return df[self.columns]

    def fit(self, df, y=None, **fit_params):
        return self


class DenseTransformer(TransformerMixin):
    """Turn a DataFrame into numpy array

    Parameters
    ----------
    columns: list of columns to keep

    Returns
    -------
    array : Numpy ndarray
    """


    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, df, **transform_params):
        return np.array(df[self.columns])

    def fit(self, df, y=None, **fit_params):
        return self


class LabelEncoding(TransformerMixin):
    def __init__(self, col="", label_mapping={}):
        self.label_mapping = label_mapping

    def transform(self, df, **transform_params):
        return df[self.features]

    def fit(self, df, y=None, **fit_params):
        return self


class SvdTransformer(TransformerMixin):
    def __init__(self, columns=[], svd_n_component=50):
        self.columns = columns
        self.svd_n_component = svd_n_component
        self.pipe = None

    def transform(self, df, **transform_params):

        concatenated = df[self.columns[0]]
        for col in self.columns[1:]:
            concatenated = concatenated.str.cat(df[col], sep=" ")

        x_svd = self.pipe.transform(concatenated)
        df_svd = pd.DataFrame(x_svd, columns=list(map(lambda x: "SVD_" + str(x), range(self.svd_n_component))))
        return pd.concat([df.reset_index(drop=True), df_svd], axis=1)

    def fit(self, df, y=None, **fit_params):
        self.pipe = make_pipeline(TfidfVectorizer(lowercase=True, ngram_range=(1, 2)),
                                  TruncatedSVD(n_components=self.svd_n_component))

        concatenated = df[self.columns[0]]
        for col in self.columns[1:]:
            concatenated = concatenated.str.cat(df[col], sep=" ")

        self.pipe.fit(concatenated)
        return self


class ConcatStringColumn(BaseEstimator, TransformerMixin):
    """
    Concat string columns from a DataFrame
    input: DataFrame
    output: List of concatenated string
    """

    def __init__(self, columns=None, col_sep=" "):
        self.columns = columns
        self.col_sep = col_sep

    def transform(self, df, **transform_params):
        concatenated = df[self.columns[0]]
        for col in self.columns[1:]:
            concatenated = concatenated.str.cat(df[col], sep=self.col_sep)
        return list(concatenated)

    def fit(self, df, y=None, **fit_params):
        return self


class CLF_fit_on_array(TransformerMixin):
    def __init__(self, clf=None, selecting_array=None):
        self.clf = clf
        self.selecting_array = selecting_array

    def transform(self, X, **transform_params):

        if not self.selecting_array:
            return self.clf.predict_proba(X)[:, 1].reshape(X.shape[0], 1)
        else:
            return self.clf.predict_proba(X[:, self.selecting_array])[:, 1].reshape(X.shape[0], 1)

    def fit(self, X, y=None, **fit_params):

        if not self.selecting_array:
            self.clf.fit(X, y)
        else:
            self.clf.fit(X[:, self.selecting_array], y)

        return self


class Average_predictions(TransformerMixin):
    def __init__(self, weights=[1, 1]):
        self.weights = weights

    def transform(self, X, **transform_params):
        return np.average(X, 1, weights=self.weights)

    def fit(self, X, y=None, **fit_params):
        return self


class StringToOneHot(TransformerMixin):

        def __init__(self, alphabet = None, maxlen=None):

            self.alphabet = alphabet
            self.maxlen = maxlen

            self.token_indice, self.indice_token = self._createMappingfromAlphabet(alphabet)

            # Sanity check both dictionary
            assert self._assert_mapping(self.token_indice, self.indice_token)

        def _encode(self, sentence, maxlen, token_indice):

            """
            Turn a string sentence into 2d array
            :param sentence: string sentence
            :param maxlen:
            :param token_indice:
            :return: 2d array (maxlen x len(token_indice))
            """

            x = np.zeros((maxlen, len(token_indice)), dtype=np.int8)
            for t in range(min(maxlen, len(sentence))):
                token = sentence[t]
                if token in token_indice:
                    x[t, token_indice[token]] = 1
            return x

        def _decode(self, x, indice_token):
            """
            Turns a 2d array into a string sentence
            :param x:
            :param indice_token:
            :return:
            """

            output = []

            for array in x:
                if np.sum(array) != 0:
                    index = array.argmax()
                    output.append(indice_token[index])

            return "".join(output)

        def _createMappingfromAlphabet(self, alphabet):
            """"
            Mapping alphabet to dictionary
            """
            # be use tokens in alphabet are unique
            assert len(alphabet) == len(set(alphabet))

            token_indice = {v: k for k, v in enumerate(alphabet)}
            indice_token = {token_indice[k]: k for k in token_indice}

            return token_indice, indice_token

        def _assert_mapping(self, token_indice, indice_token):

            assert len(token_indice) == len(indice_token)

            for k1, k2 in zip(token_indice, indice_token):
                assert indice_token[token_indice[k1]] == k1
                assert token_indice[indice_token[k2]] == k2
            return True

        def _assert_transform(self, sentence, x, token_indice, indice_token):

            """
            Compare a sentence with the decoded version of the sentence
            :param sentence: string sentence
            :param x: 2d array of encoded sentence
            :param token_indice: dictionary {k:v} == {token:integer}
            :param indice_token: dictionary {k:v} == {integer:token}
            :return:  True/False whether both string or the same or not

            """

            sentence_label = "".join([token for token in sentence[:self.maxlen] if token in token_indice])
            sentence_candidate = self._decode(x, indice_token)
            print(sentence_label)
            print(sentence_candidate)
            assert sentence_label == sentence_candidate

            return True

        def transform(self, sentences, **transform_params):

            X = np.zeros((len(sentences), self.maxlen, len(self.alphabet)), dtype=np.int8)
            for i, sentence in enumerate(sentences):
                X[i] = self._encode(sentence, self.maxlen, self.token_indice)

            #sanity check
            for i in range(min(3, len(sentences))):
                assert self._assert_transform(sentences[i],
                                              X[i], self.token_indice,
                                              self.indice_token)
            return X

        def fit(self, sentences, y=None, **fit_params):

            if not self.maxlen:
                # finding longest sentence
                self.maxlen = max(list(map(len, sentences)))

            if not self.alphabet:
                #all characters creates the alphabet

                empty_set = set()
                for sentence in sentences:
                    for char in sentence:
                        empty_set.add(char)
                self.alphabet = "".join(empty_set)

            return self
