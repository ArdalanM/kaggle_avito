__author__ = 'Ardalan'
"""
This script will:
Load train and test data ans train w2v
"""

FOLDER = "/home/ardalan/Documents/kaggle/avito/"
import gensim
import os
import shutil
import numpy as np
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_data():

    pdtrain1 = pd.read_hdf(FOLDER + "data/train_merged-part1.h")
    pdtrain2 = pd.read_hdf(FOLDER + "data/train_merged-part2.h")
    pdtrain = pdtrain1.append(pdtrain2)
    pdtest = pd.read_hdf(FOLDER + "data/test_merged.h")
    pd_data = pdtrain.append(pdtest)

    del pdtrain1
    del pdtrain2
    del pdtest
    return pd_data


print("Loading data...")
pd_data = load_data()

t1_desc1 = pd_data.title_1 + " " + pd_data.description_1
t2_desc2 = pd_data.title_2 + " " + pd_data.description_2
t_desc = t1_desc1.append(t2_desc2)
t_desc.drop_duplicates(inplace=True)
t_desc.dropna(inplace=True)
t_desc = t_desc.apply(lambda r: r.replace("\n", " "))

del pd_data


dest_folder = FOLDER + "data/w2v_t_desc/"

# Check if folder exist, if no, create the folder to store rnn data
if os.path.exists(dest_folder):
    print("Folder exist already fool!")
else:
    os.mkdir(dest_folder)


# sentences=None
size = 100
alpha = 0.025
window = 5
min_count = 5
max_vocab_size = None
sample = 1e-3
seed = 1
workers = 4
min_alpha = 0.0001
sg = 0
hs = 0
negative = 5
cbow_mean = 1
iter = 5
null_word = 0

params = {'size': size, 'alpha': alpha, 'window': window, 'min_count': min_count,
          'max_vocab_size': max_vocab_size, 'sample': sample, 'seed': seed, 'workers': workers,
          'min_alpha': min_alpha, 'sg': sg, 'hs': hs, 'negative': negative, 'cbow_mean': cbow_mean,
          'iter': iter, 'null_word': null_word}

dest_filename = "w2v_size{}_win{}_mc{}_sample{}_iter{}.model".format(
    size, window, min_count, sample, iter)


dest_path = dest_folder + dest_filename
print("-" * 50)
print("Saving model path: \n{}".format(dest_folder + dest_filename))
print("w2v params: \n{}".format(str(params)))


class MySentences(object):
    def __init__(self, sentences):
        self.sentences = sentences

    def __iter__(self):
        for sentence in self.sentences:
            yield sentence.split()


sentences = MySentences(t_desc)
model = gensim.models.Word2Vec(sentences)
model.save(dest_path)
