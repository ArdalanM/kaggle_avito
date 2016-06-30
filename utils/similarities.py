"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief: similarities one words and images
"""

import jellyfish
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance


class SimpleStringSimilarity():

    def levenshtein(self, s1, s2):
        return jellyfish.levenshtein_distance(s1, s2)

    def levenshtein_len_normalized(self, s1, s2):
        return self.levenshtein(s1, s2) / (len(s1) + len(s1))

    def damerau_levenshtein(self, s1, s2):
        return normalized_damerau_levenshtein_distance(s1, s2)

    def jaro(self, s1, s2):
        return jellyfish.jaro_distance(s1, s2)

    def jaro_winkler(self, s1, s2):
        return jellyfish.jaro_winkler(s1, s2)

    def hamming(self, s1, s2):
        return jellyfish.hamming_distance(s1, s2)


class ColStringFeature(SimpleStringSimilarity):

    def __init__(self, df):
        self.df = df

    def addCol(self, col1, col2, function):
        new_feat = []
        ctx = 0
        for x, y in zip(col1, col2):
            ctx += 1
            new_feat.append(function(x, y))
            if ctx % 100000 == 0:
                print(ctx)
        return new_feat

    def add_levenshtein(self, col1, col2):
        return self.addCol(self.df[col1], self.df[col2], self.levenshtein)

    def add_levenshtein_len_normalized(self, col1, col2):
        return self.addCol(self.df[col1], self.df[col2], self.levenshtein_len_normalized)

    def add_damerau_levenshtein(self, col1, col2):
        return self.addCol(self.df[col1], self.df[col2], self.damerau_levenshtein)

    def add_jaro(self, col1, col2):
        return self.addCol(self.df[col1], self.df[col2], self.jaro)

    def add_jaro_winkler(self, col1, col2):
        return self.addCol(self.df[col1], self.df[col2], self.jaro_winkler)

    def add_hamming(self, col1, col2):
        return self.addCol(self.df[col1], self.df[col2], self.hamming)

