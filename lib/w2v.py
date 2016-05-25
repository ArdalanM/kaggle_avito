# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief: w2v, doc2vec
"""



import os
import sys

import pandas as pd
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from lib import nlp_utils, logging_utils


# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " " # just split the text into tokens


class DataFrameSentences(object):
    def __init__(self, df, columns):
        self.df = df
        self.columns = columns

    def __iter__(self):
        for column in self.columns:
            for sentence in self.df[column]:
                tokens = nlp_utils._tokenize(sentence, token_pattern)
                yield tokens



class DataFrameWord2Vec:
    def __init__(self, df, columns, model_param):
        self.df = df
        self.columns = columns
        self.model_param = model_param
        self.model = Word2Vec(sg=self.model_param["sg"],
                                hs=self.model_param["hs"],
                                alpha=self.model_param["alpha"],
                                min_alpha=self.model_param["alpha"],
                                min_count=self.model_param["min_count"],
                                size=self.model_param["size"],
                                sample=self.model_param["sample"],
                                window=self.model_param["window"],
                                workers=self.model_param["workers"])

    def train(self):
        # build vocabulary
        self.sentences = DataFrameSentences(self.df, self.columns)
        self.model.build_vocab(self.sentences)
        # train for n_epoch
        for i in range(self.model_param["n_epoch"]):
            self.sentences = DataFrameSentences(self.df, self.columns)
            self.model.train(self.sentences)
            self.model.alpha *= self.model_param["learning_rate_decay"]
            self.model.min_alpha = self.model.alpha
        return self

    def save(self, model_dir, model_name):
        fname = os.path.join(model_dir, model_name)
        self.model.save(fname)



logger = logging_utils._get_logger()



