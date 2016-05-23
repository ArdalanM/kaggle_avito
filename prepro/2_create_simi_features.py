__author__ = 'Ardalan'
"""
This script will:

1: load:
    - train_merged-part1.h
    - train_merged-part2.h
    - test_merged.h
2: create train and test similarity features. Each created features is a pandas DataFrame
3: Dump those features in a binary file
"""


FOLDER = "/home/ardalan/Documents/kaggle/avito/"

import jellyfish
import pandas as pd

from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance


def levenshtein_distance(s1, s2):

    if s1 == s2:
        return 0
    rows = len(s1)+1
    cols = len(s2)+1

    if not s1:
        return cols-1
    if not s2:
        return rows-1

    prev = None
    cur = range(cols)
    for r in range(1, rows):
        prev, cur = cur, [r] + [0]*(cols-1)
        for c in range(1, cols):
            deletion = prev[c] + 1
            insertion = cur[c-1] + 1
            edit = prev[c-1] + (0 if s1[r-1] == s2[c-1] else 1)
            cur[c] = min(edit, deletion, insertion)

    return cur[-1]


def get_features(col1, col2, function):
    new_feat = []
    ctx = 0
    for x, y in zip(col1, col2):
        ctx += 1
        new_feat.append(function(x, y))
        if ctx % 100000 == 0:
            print(ctx)
    pd_feat = pd.DataFrame(new_feat)
    return pd_feat



    pd_feat.to_hdf(save_name, 'wb')



# TRAIN SET
pd_data_part1 = pd.read_hdf(FOLDER + "data/train_merged-part1.h")
pd_data_part2 = pd.read_hdf(FOLDER + "data/train_merged-part2.h")
pd_data = pd_data_part1.append(pd_data_part2)

pd_data.description_1.fillna("", inplace=True)
pd_data.description_2.fillna("", inplace=True)

pd_data.title_1.fillna("", inplace=True)
pd_data.title_2.fillna("", inplace=True)

del pd_data_part1
del pd_data_part2


cols = [('title_1', 'title_2'), ('description_1', 'description_2')]

for col1, col2 in cols:
    print("Selecting: {} and {}".format(col1, col2))
    colname = col1.split("_")[0]
    col_1 = pd_data['title_1'].values
    col_2 = pd_data['title_2'].values

    print("Levenshtein")
    pd_feat = get_features(col_1, col_2, jellyfish.levenshtein_distance)
    pd_feat.to_hdf(FOLDER + "data/Feat_train_{}_leven.h".format(colname), 'w')

    print("Jaro")
    pd_feat = get_features(col_1, col_2, jellyfish.jaro_distance)
    pd_feat.to_hdf(FOLDER + "data/Feat_train_{}_jaro.h".format(colname), 'w')

    print("jaro winkler")
    pd_feat = get_features(col_1, col_2, jellyfish.jaro_winkler)
    pd_feat.to_hdf(FOLDER + "data/Feat_train_{}_jarowinkler.h".format(colname), 'w')

    print("hamming")
    pd_feat = get_features(col_1, col_2, jellyfish.hamming_distance)
    pd_feat.to_hdf(FOLDER + "data/Feat_train_{}_hamming.h".format(colname), 'w')

    print("normalized demarau levenstein")
    pd_feat = get_features(col_1, col_2, normalized_damerau_levenshtein_distance)
    pd_feat.to_hdf(FOLDER + "data/Feat_train_{}_demarauleven.h".format(colname), 'w')



# TEST SET

pd_data = pd.read_hdf(FOLDER + "data/test_merged.h")

pd_data.description_1.fillna("", inplace=True)
pd_data.description_2.fillna("", inplace=True)

pd_data.title_1.fillna("", inplace=True)
pd_data.title_2.fillna("", inplace=True)

cols = [('title_1', 'title_2'), ('description_1', 'description_2')]

for col1, col2 in cols:
    print("Selecting: {} and {}".format(col1, col2))
    colname = col1.split("_")[0]
    col_1 = pd_data['title_1'].values
    col_2 = pd_data['title_2'].values

    print("Levenshtein")
    pd_feat = get_features(col_1, col_2, jellyfish.levenshtein_distance)
    pd_feat.to_hdf(FOLDER + "data/Feat_test_{}_leven.h".format(colname), 'w')

    print("Jaro")
    pd_feat = get_features(col_1, col_2, jellyfish.jaro_distance)
    pd_feat.to_hdf(FOLDER + "data/Feat_test_{}_jaro.h".format(colname), 'w')

    print("jaro winkler")
    pd_feat = get_features(col_1, col_2, jellyfish.jaro_winkler)
    pd_feat.to_hdf(FOLDER + "data/Feat_test_{}_jarowinkler.h".format(colname), 'w')

    print("hamming")
    pd_feat = get_features(col_1, col_2, jellyfish.hamming_distance)
    pd_feat.to_hdf(FOLDER + "data/Feat_test_{}_hamming.h".format(colname), 'w')

    print("normalized demarau levenstein")
    pd_feat = get_features(col_1, col_2, normalized_damerau_levenshtein_distance)
    pd_feat.to_hdf(FOLDER + "data/Feat_test_{}_demarauleven.h".format(colname), 'w')
