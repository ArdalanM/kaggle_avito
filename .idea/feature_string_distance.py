# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief: Clean and Stemming
"""

import re
import time
import nltk
import os
import config


from lib import logging_utils, pkl_utils, ngram_utils


class JaccardCoef_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        # super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "JaccardCoef_%s"%self.ngram_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        ret



logger = logging_utils._get_logger(config.CODE_FOLDER, "3_clean_string_data.log")

logger.info("KAGGLE: Loading: {}".format(config.ALL_DATA_RAW))
df = pkl_utils._load(config.ALL_DATA_RAW)

columns = [
    "title_1",
    "description_1",
    "title_2",
    "description_2"
]
columns = [col for col in columns if col in df.columns]

out = ngram_utils._ngrams(list(df['title_1']), 1)

