"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief: vectorizer that operates on text data
"""

import keras
import numpy as np

from keras.preprocessing.text import base_filter

class genericSequenceVectorizer(object):
    def __init__(self, alphabet=None, maxlen=500,
                 filters=base_filter(), lower=False,
                 split=" ", char_level=True):

        self.token_indice = {}
        self.indice_token = {}
        self.nb_tokens = None

        self.alphabet = alphabet
        self.maxlen = maxlen

        self.filters = filters
        self.lower = lower
        self.split = split
        self.char_level = char_level

        self.token_count = {}
        self.document_count = 0

    def _createMappingfromAlphabet(self, alphabet):
        """"
        Mapping alphabet to dictionary with keys > 0
        """

        # be use tokens in alphabet are unique
        assert len(alphabet) == len(set(alphabet))

        token_indice = {v: k + 1 for k, v in enumerate(alphabet)}
        indice_token = {token_indice[k]: k for k in token_indice}

        return token_indice, indice_token

    def _assert_mapping(self, token_indice, indice_token):

        assert len(token_indice) == len(indice_token)

        for k1, k2 in zip(token_indice, indice_token):
            assert indice_token[token_indice[k1]] == k1
            assert token_indice[indice_token[k2]] == k2
        return True

    def fit(self, texts):
        '''
            required before using texts_to_sequences or texts_to_matrix

        # Arguments
            texts: can be a list of strings,
                or a generator of strings (for memory-efficiency)
        '''
        self.document_count = 0
        for text in texts:
            self.document_count += 1
            if self.char_level:
                seq = text
            else:
                seq = keras.preprocessing.text.text_to_word_sequence(text, self.filters, self.lower, self.split)
            for w in seq:
                if w in self.token_count:
                    self.token_count[w] += 1
                else:
                    self.token_count[w] = 1

        wcounts = list(self.token_count.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]

        if self.alphabet:
            self.token_indice, self.indice_token = self._createMappingfromAlphabet(self.alphabet)
        else:
            self.token_indice = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
            self.indice_token = {self.token_indice[k]: k for k in self.token_indice}

        self._assert_mapping(self.token_indice, self.indice_token)
        self.nb_tokens = len(self.token_indice)

    def transform(self, texts):
        pass

    def fit_transform(self, texts):
        self.fit(texts)

        return self.transform(texts)

    def seq_to_string(self, sequence):

        if self.char_level:
            candidate = "".join([self.indice_token[i] if i in self.indice_token else " " for i in sequence])
        else:
            candidate = " ".join([self.indice_token[i] for i in sequence if i in self.indice_token])
        return candidate

    def _matrix_to_string(self, X):

        # safety check
        output = ""
        for i in range(self.maxlen):
            if np.sum(X[i, :]) > 0:
                output += self.indice_token[X[i, :].argmax()]
            else:
                output += " "
        return output

    def get_params(self):
        return {'maxlen': self.maxlen, 'nb_tokens': self.nb_tokens,
                'alphabet': self.alphabet, 'char_level': self.char_level}

    def get_feature_names(self):
        return self.token_indice

    def __repr__(self):
        return self.__class__.__name__


class VectorizeToSeqOH(genericSequenceVectorizer):
    def _assert_transform(self, sentence, X):

        candidate = self._matrix_to_string(X)
        label = "".join([char if char in self.token_indice else " " for char in sentence[:self.maxlen]])

        # removing leading and tailing white spaces
        candidate = candidate.strip()
        label = label.strip()

        print(label, len(label))
        print(candidate, len(candidate))

        assert candidate == label

    def transform(self, sentences):
        print('Vectorization...')
        X = np.zeros((len(sentences), self.maxlen, self.nb_tokens + 1), dtype=np.int16)
        for i, sentence in enumerate(sentences):
            for t in range(min(self.maxlen, len(sentence))):
                char = sentence[t]

                if char in self.token_indice:
                    X[i, t, self.token_indice[char]] = 1
                else:
                    X[i, t, :] = 0

        assert np.max(X) <= self.nb_tokens
        self._assert_transform(sentences[0], X[0])
        return X


class VectorizeToSeqIndices(genericSequenceVectorizer):

    def _assert_transform(self, sentence, sequence):

        candidate = self.seq_to_string(sequence)
        if self.char_level:
            label = "".join([c for c in sentence if c in self.token_indice])[:self.maxlen]
        else:
            label = " ".join([w for w in sentence.strip().split(self.split)[:self.maxlen] if w in self.token_indice])

        # removing leading and tailing white spaces
        candidate = candidate.strip()
        label = label.strip()

        print(label, len(label))
        print(candidate, len(candidate))

        assert candidate == label

    def transform(self, sentences):

        seqs = []
        for sentence in sentences:
            if not self.char_level:
                sentence = keras.preprocessing.text.text_to_word_sequence(sentence, self.filters, self.lower,
                                                                          self.split)
            seq = []
            for token in sentence:
                if token in self.token_indice:
                    seq.append(self.token_indice[token])
                else:
                    seq.append(0)
            seqs.append(seq)

        # assert self.seq_to_string(seqs[0]) == sentences[0]

        seqs = keras.preprocessing.sequence.pad_sequences(seqs, maxlen=self.maxlen, dtype='int32',
                                      padding='pre', truncating='post', value=0)

        assert np.max(seqs) <= self.nb_tokens
        self._assert_transform(sentences[0], seqs[0])

        return seqs

