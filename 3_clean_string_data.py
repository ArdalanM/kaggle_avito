# -*- coding: utf-8 -*-
"""
@author: Evgeny BAZAROV
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief: Clean string data
"""

import gc
import re
import nltk
import regex
import numpy as np
import pandas as pd
import nltk.corpus
from nltk import SnowballStemmer


class BaseReplacer:
    def __init__(self, pattern_replace_pair_list=[]):
        self.pattern_replace_pair_list = pattern_replace_pair_list

    def transform(self, text):
        for pattern, replace in self.pattern_replace_pair_list:
            try:
                text = regex.sub(pattern, replace, text)
            except:
                pass
        return regex.sub(r"\s+", " ", text).strip()


class LowerCaseConverter(BaseReplacer):
    """
    Traditional -> traditional
    """

    def transform(self, text):
        return text.lower()


class LetterLetterSplitter(BaseReplacer):
    """
    For letter and letter
    /:
    Cleaner/Conditioner -> Cleaner Conditioner
    -:
    Vinyl-Leather-Rubber -> Vinyl Leather Rubber
    For digit and digit, we keep it as we will generate some features via math operations,
    such as approximate height/width/area etc.
    /:
    3/4 -> 3/4
    -:
    1-1/4 -> 1-1/4
    """

    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"([a-zа-я]+)[/\-]([a-zа-я]+)", r"\1 \2"),
        ]


class DigitLetterSplitter(BaseReplacer):
    """
    x:
    1x1x1x1x1 -> 1 x 1 x 1 x 1 x 1
    19.875x31.5x1 -> 19.875 x 31.5 x 1
    -:
    1-Gang -> 1 Gang
    48-Light -> 48 Light
    .:
    includes a tile flange to further simplify installation.60 in. L x 36 in. W x 20 in. ->
    includes a tile flange to further simplify installation. 60 in. L x 36 in. W x 20 in.
    """

    def __init__(self):
        self.pattern_replace_pair_list = [
            #             [^a-zа-я0-9]
            (r"(\d+)[\.\-]*([a-zа-я]+)", r"\1 \2"),
            (r"([a-zа-я]+)[\.\-]*(\d+)", r"\1 \2"),
        ]


class DigitCommaDigitMerger(BaseReplacer):
    """
    1,000,000 -> 1000000
    """

    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"(?<=\d+),(?=000)", r""),
            (r"(?<=\d+).(?=000)", r""),
        ]


class NumberDigitMapper(BaseReplacer):
    """
    один -> 1
    два -> 2
    """

    def __init__(self):
        numbers = [
            "ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять",
            "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать",
            "восемнадцать", "девятнадцать", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
            "восемьдесят", "девяносто", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот",
            "восемьсот", "девятьсот"
        ]
        digits = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100,
            200, 300, 400, 500, 600, 700, 800, 900
        ]
        self.pattern_replace_pair_list = [
            (r"(?<=\W|^)%s(?=\W|$)" % n, str(d)) for n, d in zip(numbers, digits)
            ]


stopwords = frozenset(word \
                      for word in nltk.corpus.stopwords.words("russian") \
                      if word != "не")
stemmer = SnowballStemmer('russian')
engChars = [ord(char) for char in u"cCyoOBaAKpPeE"]
rusChars = [ord(char) for char in u"сСуоОВаАКрРеЕ"]
eng_rusTranslateTable = dict(zip(engChars, rusChars))
rus_engTranslateTable = dict(zip(rusChars, engChars))


def correctWord(w):
    """ Corrects word by replacing characters with written similarly depending
    on which language the word.
    Fraudsters use this technique to avoid detection by anti-fraud algorithms.
    """
    if len(re.findall(r"[а-я]", w)) > len(re.findall(r"[a-z]", w)):
        return w.translate(eng_rusTranslateTable)
    else:
        return w.translate(rus_engTranslateTable)


def getWords(text, stemmRequired=True,
             correctWordRequired=True,
             excludeStopwordsRequired=True):
    """ Splits the text into words, discards stop words and applies stemmer.
    Parameters
    ----------
    text : str - initial string
    stemmRequired : bool - flag whether stemming required
    correctWordRequired : bool - flag whether correction of words required
    """
    cleanText = re.sub(r"\p{P}+", "", text)
    cleanText = cleanText.replace("+", " ")
    #     cleanText = text.replace(",", " ").replace(".", " ")
    #     cleanText = re.sub(u'[^a-zа-я0-9]', ' ', text.lower())
    if correctWordRequired:
        if excludeStopwordsRequired:
            words = [correctWord(w) \
                         if not stemmRequired or re.search("[0-9a-z]", w) \
                         else stemmer.stem(correctWord(w)) \
                     for w in cleanText.split() \
                     if w not in stopwords]
        else:
            words = [correctWord(w) \
                         if not stemmRequired or re.search("[0-9a-z]", w) \
                         else stemmer.stem(correctWord(w)) \
                     for w in cleanText.split()
                     ]
    else:
        if excludeStopwordsRequired:
            words = [w \
                         if not stemmRequired or re.search("[0-9a-z]", w) \
                         else stemmer.stem(w) \
                     for w in cleanText.split() \
                     if w not in stopwords]
        else:
            words = [w \
                         if not stemmRequired or re.search("[0-9a-z]", w) \
                         else stemmer.stem(w) \
                     for w in cleanText.split()
                     ]

    return " ".join(words)


pattern_replace_pair_list = [
    # Remove single & double apostrophes
    ("[\"]+", r""),
    ("[\']+", r""),
    # Remove product codes (long words (>5 characters) that are all caps, numbers or mix pf both)
    # don't use raw string format
    #             ("[ ]?\\b[0-9A-Z-]{5,}\\b", ""),
]

text = BaseReplacer(pattern_replace_pair_list).transform(text)
#     print(text)
text = LowerCaseConverter().transform(text)
#     print(text)
text = DigitLetterSplitter().transform(text)
#     print(text)
text = DigitCommaDigitMerger().transform(text)
#     print(text)
text = NumberDigitMapper().transform(text)
#     print(text)
text = getWords(text)
#     print(text)
