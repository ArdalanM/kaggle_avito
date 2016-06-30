# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: distance features

"""

import re
import sys
import string

import numpy as np
import pandas as pd

import config
from lib import dist_utils, ngram_utils, nlp_utils
from lib import logging_utils, time_utils, pkl_utils

from feature_base import BaseEstimator, PairwiseFeatureWrapper

# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " "  # just split the text into tokens


# ------------------- Jaccard & Dice --------------------------------------
class JaccardCoef_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "JaccardCoef_%s" % self.ngram_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        return dist_utils._jaccard_coef(obs_ngrams, target_ngrams)


class DiceDistance_Ngram(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        return "DiceDistance_%s" % self.ngram_str

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        return dist_utils._dice_dist(obs_ngrams, target_ngrams)



# ------------------ Edit Distance --------------------------------
class EditDistance(BaseEstimator):
    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "EditDistance"

    def transform_one(self, obs, target, id):
        return dist_utils._edit_dist(obs, target)


class EditDistance_Ngram(BaseEstimator):
    """Double aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode_prev="", aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode, None, aggregation_mode_prev)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        feat_name = []
        for m1 in self.aggregation_mode_prev:
            for m in self.aggregation_mode:
                n = "EditDistance_%s_%s_%s" % (self.ngram_str, string.capwords(m1), string.capwords(m))
                feat_name.append(n)
        return feat_name

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        val_list = []
        for w1 in obs_ngrams:
            _val_list = []
            for w2 in target_ngrams:
                s = dist_utils._edit_dist(w1, w2)
                _val_list.append(s)
            if len(_val_list) == 0:
                _val_list = [config.MISSING_VALUE_NUMERIC]
            val_list.append(_val_list)
        if len(val_list) == 0:
            val_list = [[config.MISSING_VALUE_NUMERIC]]
        return val_list


# ------------------ Compression Distance --------------------------------
class CompressionDistance(BaseEstimator):
    """Very time consuming"""

    def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode)

    def __name__(self):
        return "CompressionDistance"

    def transform_one(self, obs, target, id):
        return dist_utils._compression_dist(obs, target)


class CompressionDistance_Ngram(BaseEstimator):
    """Double aggregation features"""

    def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode_prev="", aggregation_mode=""):
        super().__init__(obs_corpus, target_corpus, aggregation_mode, None, aggregation_mode_prev)
        self.ngram = ngram
        self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

    def __name__(self):
        feat_name = []
        for m1 in self.aggregation_mode_prev:
            for m in self.aggregation_mode:
                n = "CompressionDistance_%s_%s_%s" % (self.ngram_str, string.capwords(m1), string.capwords(m))
                feat_name.append(n)
        return feat_name

    def transform_one(self, obs, target, id):
        obs_tokens = nlp_utils._tokenize(obs, token_pattern)
        target_tokens = nlp_utils._tokenize(target, token_pattern)
        obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
        target_ngrams = ngram_utils._ngrams(target_tokens, self.ngram)
        val_list = []
        for w1 in obs_ngrams:
            _val_list = []
            for w2 in target_ngrams:
                s = dist_utils._compression_dist(w1, w2)
                _val_list.append(s)
            if len(_val_list) == 0:
                _val_list = [config.MISSING_VALUE_NUMERIC]
            val_list.append(_val_list)
        if len(val_list) == 0:
            val_list = [[config.MISSING_VALUE_NUMERIC]]
        return val_list


# ---------------------------- Main --------------------------------------
def run_ngram_jaccard():
    logname = "generate_feature_ngram_jaccard_%s.log" % time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_FOLDER, logname)
    logger.info("Loading: {}".format(config.ALL_DATA_RAW))
    dfAll = pkl_utils._load(config.ALL_DATA_RAW)


    for col in dfAll:
        dfAll[col].fillna(config.MISSING_VALUE_STRING, inplace=True)

    generators = [JaccardCoef_Ngram, DiceDistance_Ngram]
    obs_fields_list = []
    target_fields_list = []
    obs_fields_list.append(["title_1", "description_1"][:])
    target_fields_list.append(["title_2", "description_2"][:])
    ngrams = [1, 2, 3, 12, 123]
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        for generator in generators:
            for ngram in ngrams:
                param_list = [ngram]
                pf = PairwiseFeatureWrapper(generator, dfAll, obs_fields, target_fields, param_list, config.FEATURES_FOLDER,
                                            logger)
                pf.go()


def run_edit_distance():
    logname = "generate_feature_edit_distance_%s.log" % time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_FOLDER, logname)
    logger.info("Loading: {}".format(config.ALL_DATA_RAW))
    dfAll = pkl_utils._load(config.ALL_DATA_RAW)

    for col in dfAll:
        dfAll[col].fillna(config.MISSING_VALUE_STRING, inplace=True)

    obs_fields_list = []
    target_fields_list = []
    obs_fields_list.append(["description_1"][:])
    target_fields_list.append(["description_2"][:])
    # ngrams = [1, 2, 3, 12, 123]
    ngrams = [1, 2]
    aggregation_mode_prev = ["mean", "max", "min", "median"]
    aggregation_mode = ["mean", "std", "max", "min", "median"]
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        param_list = []
        pf = PairwiseFeatureWrapper(EditDistance, dfAll, obs_fields, target_fields, param_list, config.FEATURES_FOLDER, logger)
        pf.go()
        for ngram in ngrams:
            param_list = [ngram, aggregation_mode_prev, aggregation_mode]
            pf = PairwiseFeatureWrapper(EditDistance_Ngram, dfAll, obs_fields, target_fields, param_list,
                                        config.FEATURES_FOLDER, logger)
            pf.go()


def run_compression_distance():
    logname = "generate_feature_compression_distance_%s.log" % time_utils._timestamp()
    logger = logging_utils._get_logger(config.LOG_FOLDER, logname)
    logger.info("Loading: {}".format(config.ALL_DATA_RAW))
    dfAll = pkl_utils._load(config.ALL_DATA_RAW)


    for col in dfAll:
        dfAll[col].fillna(config.MISSING_VALUE_STRING, inplace=True)

    obs_fields_list = []
    target_fields_list = []
    obs_fields_list.append(["title_1", "description_1"][:])
    target_fields_list.append(["title_2", "description_2"][:])
    ngrams = [1, 2, 3, 12, 123]
    for obs_fields, target_fields in zip(obs_fields_list, target_fields_list):
        param_list = []
        pf = PairwiseFeatureWrapper(CompressionDistance, dfAll, obs_fields, target_fields, param_list, config.FEATURES_FOLDER,
                                    logger)
        pf.go()
        for ngram in ngrams:
            param_list = [ngram, aggregation_mode_prev, aggregation_mode]
            pf = PairwiseFeatureWrapper(CompressionDistance_Ngram, dfAll, obs_fields, target_fields, param_list,
                                        config.FEATURES_FOLDER, logger)
            pf.go()


def main(which):
    if which == "jaccard":
        run_ngram_jaccard()
    elif which == "edit":
        run_edit_distance()
    elif which == "compression":
        run_compression_distance()


if __name__ == "__main__":
    main(sys.argv[1])
