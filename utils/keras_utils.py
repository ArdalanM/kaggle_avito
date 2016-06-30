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
import copy
import numpy as np
from utils import np_utils
from sklearn import preprocessing


class genericKerasCLF():
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
        self.output_dim = None
        self.input_length = None

        self.model = None
        self.logs = None
        self.class_weight = class_weight

        self.scaler = None

    def _set_input_dim(self, X, y, validation_data=()):

        assert len(X.shape) == 2 or len(X.shape) == 3
        assert len(validation_data) == 2 or len(validation_data) == 0

        if len(X.shape) == 3:
            # tensor
            self.input_dim = X.shape[2]
            self.input_length = X.shape[1]
        elif len(X.shape) == 2:
            # matrix
            self.input_dim = X.shape[1]
            # self.input_length = X.shape[1]

        nb_class = len(np.unique(y))
        y = np_utils._to_categorical(y, nb_class)

        if len(validation_data) == 2:
            X_val, y_val = validation_data[0], validation_data[1]
            validation_data = (X_val, np_utils._to_categorical(y_val, nb_class))
        else:
            validation_data = None

        return X, y, validation_data

    def build_model(self):
        pass

    def fit(self, X, y, validation_data=()):

        X, y, validation_data = self._set_input_dim(X, y, validation_data)

        # print(y.shape)
        # print(validation_data[1].shape)
        print("input_dim: {}, output_dim: {}, input_length: {}"
            .format(self.input_dim,
                    self.output_dim,
                    self.input_length))

        if self.rebuild:
            self.model = self.build_model()

        logs = self.model.fit(X, y, batch_size=self.batch_size, nb_epoch=self.nb_epoch,
                              validation_data=validation_data,
                              verbose=self.verbose, shuffle=self.shuffle,
                              callbacks=copy.deepcopy(self.callbacks),
                              class_weight=self.class_weight)
        self.logs = logs

    def predict_proba(self, X):

        assert hasattr(self.model, 'predict_proba') or hasattr(self.model, 'predict')

        if hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict_proba(X, verbose=False)
        else:
            prediction = self.model.predict(X, verbose=False)

        return prediction
