# -*- coding: utf-8 -*-
"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief: word2vec & doc2vec trainer
"""
import os
import config

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from lib import nlp_utils
from lib import logging_utils, pkl_utils

# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " "  # just split the text into tokens


# W2V defaut params
# size = 100
# alpha = 0.025
# window = 5
# min_count = 5
# max_vocab_size = None
# sample = 1e-3
# seed = 1
# workers = 4
# min_alpha = 0.0001
# sg = 0
# hs = 0
# negative = 5
# cbow_mean = 1
# iter = 5
# null_word = 0


# ---------------------- Word2Vec ----------------------
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


class DataFrameLabelSentences(object):
    def __init__(self, df, columns):
        self.df = df
        self.columns = columns
        self.cnt = -1
        self.sent_label = {}

    def __iter__(self):
        for column in self.columns:
            for sentence in self.df[column]:
                if not sentence in self.sent_label:
                    self.cnt += 1
                    self.sent_label[sentence] = "SENT_%d" % self.cnt
                tokens = nlp_utils._tokenize(sentence, token_pattern)
                yield TaggedDocument(words=tokens, tags=[self.sent_label[sentence]])


class DataFrameDoc2Vec(DataFrameWord2Vec):
    def __init__(self, df, columns, model_param):
        super().__init__(df, columns, model_param)
        self.model = Doc2Vec(dm=self.model_param["dm"],
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
        self.sentences = DataFrameLabelSentences(self.df, self.columns)
        self.model.build_vocab(self.sentences)
        # train for n_epoch
        for i in range(self.model_param["n_epoch"]):
            self.sentences = DataFrameLabelSentences(self.df, self.columns)
            self.model.train(self.sentences)
            self.model.alpha *= self.model_param["learning_rate_decay"]
            self.model.min_alpha = self.model.alpha
        return self

    def save(self, model_dir, model_name):
        fname = os.path.join(model_dir, model_name)
        self.model.save(fname)
        pkl_utils._save("%s.sent_label" % fname, self.sentences.sent_label)


# ---------------------- Main ----------------------
logger = logging_utils._get_logger(config.LOG_FOLDER, "2_train_w2v.log")

logger.info("KAGGLE: Loading: {}".format(config.ITEMINFO_RAW))
df = pkl_utils._load(config.ITEMINFO_RAW)


columns = ["title", "description"]
columns = [col for col in columns if col in df.columns]

logger.info("KAGGLE: Fillna, '\\n' ==> "" ")
for col in columns:
    df[col].fillna("", inplace=True)
    df[col] = df[col].apply(lambda r: r.replace("\n", " "))

# TRAINING DOC2VEC
# model_param = {
#     "alpha": config.W2V_ALPHA,
#     "learning_rate_decay": config.W2V_LEARNING_RATE_DECAY,
#     "n_epoch": config.W2V_N_EPOCH,
#     "sg": 1,  # not use
#     "dm": 1,
#     "hs": 1,
#     "min_count": config.W2V_MIN_COUNT,
#     "size": config.W2V_DIM,
#     "sample": 0.001,
#     "window": config.W2V_WINDOW,
#     "workers": config.W2V_WORKERS,
# }
# model_dir = config.WORD2VEC_MODEL_DIR
# model_name = "d2v_{}_split[{}]_s{}_win{}_mc{}_iter{}_decay{}.model".format(
#     config.ALL_DATA_RAW.split("_")[-1],
#     token_pattern,
#     model_param["size"],
#     model_param["window"],
#     model_param["min_count"],
#     model_param["n_epoch"],
#     model_param['learning_rate_decay'])
# doc2vec = DataFrameDoc2Vec(df, columns, model_param)
# doc2vec.train()
# doc2vec.save(model_dir, model_name)

# TRAINING W2V
model_param = {
    "alpha": config.W2V_ALPHA,
    "learning_rate_decay": config.W2V_LEARNING_RATE_DECAY,
    "n_epoch": config.W2V_N_EPOCH,
    "sg": 1,
    "hs": 1,
    "min_count": config.W2V_MIN_COUNT,
    "size": config.W2V_DIM,
    "sample": 0.001,
    "window": config.W2V_WINDOW,
    "workers": config.W2V_WORKERS,
}
model_dir = config.WORD2VEC_MODEL_DIR
model_name = "w2v_{}_split[{}]_s{}_win{}_mc{}_iter{}_decay{}.model".format(
    config.ALL_DATA_RAW.split("_")[-1],
    token_pattern,
    model_param["size"],
    model_param["window"],
    model_param["min_count"],
    model_param["n_epoch"],
    model_param['learning_rate_decay'])
logger.info("KAGGLE: saving model in: {}".format(model_dir))
logger.info("KAGGLE: model name: {}".format(model_name))
logger.info("KAGGLE: training w2v with params: {}".format(model_param))
word2vec = DataFrameWord2Vec(df, columns, model_param)
word2vec.train()
logger.info("KAGGLE: training complete")
word2vec.save(model_dir, model_name)
logger.info("KAGGLE: model saved in : {}".format(model_dir))
