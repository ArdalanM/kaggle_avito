# -*- coding: utf-8 -*-
"""
@author: Evgeny BAZAROV
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief: Clean and Stemming
"""

# import re
import time
import regex
import os
import config

import nltk.corpus
from nltk import SnowballStemmer

from lib import logging_utils, pkl_utils


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


class CorrectStemWords(BaseReplacer):

    def transform(self, text):
        return self.getWords(text)

    def correctWord (self, w):
        """ Corrects word by replacing characters with written similarly depending
        on which language the word.
        Fraudsters use this technique to avoid detection by anti-fraud algorithms.
        """
        if len(regex.findall(r"[а-я]",w))>len(regex.findall(r"[a-z]",w)):
            return w.translate(eng_rusTranslateTable)
        else:
            return w.translate(rus_engTranslateTable)

    def getWords(self, text, stemmRequired = True,
                 correctWordRequired = True,
                 excludeStopwordsRequired = True):
        """ Splits the text into words, discards stop words and applies stemmer.
        Parameters
        ----------
        text : str - initial string
        stemmRequired : bool - flag whether stemming required
        correctWordRequired : bool - flag whether correction of words required
        """
        cleanText = regex.sub(r"\p{P}+", "", text)
        cleanText = cleanText.replace("+", " ")
    #     cleanText = text.replace(",", " ").replace(".", " ")
    #     cleanText = regex.sub(u'[^a-zа-я0-9]', ' ', text.lower())
        if correctWordRequired:
            if excludeStopwordsRequired:
                words = [self.correctWord(w) \
                        if not stemmRequired or regex.search("[0-9a-z]", w) \
                        else stemmer.stem(self.correctWord(w)) \
                        for w in cleanText.split() \
                        if w not in stopwords]
            else:
                words = [self.correctWord(w) \
                        if not stemmRequired or regex.search("[0-9a-z]", w) \
                        else stemmer.stem(self.correctWord(w)) \
                        for w in cleanText.split()
                        ]
        else:
            if excludeStopwordsRequired:
                words = [w \
                        if not stemmRequired or regex.search("[0-9a-z]", w) \
                        else stemmer.stem(w) \
                        for w in cleanText.split() \
                        if w not in stopwords]
            else:
                words = [w \
                        if not stemmRequired or regex.search("[0-9a-z]", w) \
                        else stemmer.stem(w) \
                        for w in cleanText.split()
                        ]

        return " ".join(words)




stopwords = frozenset(word for word in nltk.corpus.stopwords.words("russian")
                      if word != "не")
stemmer = SnowballStemmer('russian')
engChars = [ord(char) for char in u"cCyoOBaAKpPeE"]
rusChars = [ord(char) for char in u"сСуоОВаАКрРеЕ"]
eng_rusTranslateTable = dict(zip(engChars, rusChars))
rus_engTranslateTable = dict(zip(rusChars, engChars))



# ---------------------- Main ----------------------
starttime = time.time()

logger = logging_utils._get_logger(config.LOG_FOLDER, "3_clean_string_data.log")
logger.info("KAGGLE: Loading: {}".format(config.ITEMINFO_RAW))
df = pkl_utils._load(config.ITEMINFO_RAW)

columns = ["title", "description"]
columns = [col for col in columns if col in df.columns]

processors = [
    BaseReplacer([
        # Remove single & double apostrophes
        ("[\"]+", r""),
        ("[\']+", r""),
    ]),
    LowerCaseConverter(),
    LetterLetterSplitter(),
    DigitLetterSplitter(),
    DigitCommaDigitMerger(),
    NumberDigitMapper(),
    CorrectStemWords()
]

test_string = "Hello 2x4 my-name вавиловка 45-swag biggest  100,000 I'am 45,4 asdf один years old"
logger.info("KAGGLE: before: {}".format(test_string))
for processor in processors:
    test_string = processor.transform(test_string)
logger.info("KAGGLE: after: {}".format(test_string))


for col in columns:
    StartCoTime = time.time()
    logger.info("KAGGLE: Cleaning col: {}".format(col))
    pdseries = df[col].fillna("")
    for processor in processors:
        logger.info("KAGGLE: applying {}".format(str(processor)))
        pdseries = pdseries.apply(processor.transform)
        logger.info("KAGGLE: {}, {:.0f} secs".format(str(processor), time.time() - StartCoTime))
    df[col] = pdseries
    logger.info("KAGGLE: Cleaned col: {} in {:.0f} sec".format(col, time.time() - StartCoTime))

filename = "ItemInfo_cleaned.pkl"
logger.info("KAGGLE: Saving in {}".format(filename))
pkl_utils._save(os.path.join(config.PICKLE_DATA_FOLDER, filename), df)
logger.info("KAGGLE: Script finished in {:.0f} sec".format(time.time() - starttime))



#Creating cleaned dataset
from lib import utils
import pandas as pd

ItemPair_train = os.path.join(config.DATA_FOLDER, "ItemPairs_train.csv.zip")
ItemPair_test = os.path.join(config.DATA_FOLDER, "ItemPairs_test.csv.zip")

pdItemPair_train, _ = utils.loadFileinZipFile(ItemPair_train, encoding='utf-8')
pdItemPair_test, _ = utils.loadFileinZipFile(ItemPair_test, encoding='utf-8')
pdItemPair = pdItemPair_train.append(pdItemPair_test)


df = pkl_utils._load(config.PICKLE_DATA_FOLDER + "/ItemInfo_cleaned.pkl")

pd_data = pd.merge(pdItemPair, df, how='left', left_on='itemID_1', right_on='itemID')
pd_data.drop(['itemID_1'], 1, inplace=True)
pd_data.rename(columns={
    'itemID': 'itemID_1',
    'categoryID': 'categoryID_1',
    'title': 'title_1',
    'description': 'description_1',
    'images_array': 'images_array_1',
    'attrsJSON': 'attrsJSON_1',
    'price': 'price_1',
    'locationID': 'locationID_1',
    'metroID': 'metroID_1',
    'lat': 'lat_1',
    'lon': 'lon_1',
    'parentCategoryID': 'parentCategoryID_1',
    'regionID': 'regionID_1'
}, inplace=1)
pd_data = pd.merge(pd_data, df, how='left', left_on='itemID_2', right_on='itemID')
pd_data.drop(['itemID_2'], 1, inplace=True)
pd_data.rename(columns={
    'itemID': 'itemID_2',
    'categoryID': 'categoryID_2',
    'title': 'title_2',
    'description': 'description_2',
    'images_array': 'images_array_2',
    'attrsJSON': 'attrsJSON_2',
    'price': 'price_2',
    'locationID': 'locationID_2',
    'metroID': 'metroID_2',
    'lat': 'lat_2',
    'lon': 'lon_2',
    'parentCategoryID': 'parentCategoryID_2',
    'regionID': 'regionID_2'
}, inplace=1)

pkl_utils._save(os.path.join(config.PICKLE_DATA_FOLDER, "all_cleaned.pkl"), pd_data)


















# for col in columns:
#
#     df[col].fillna("", inplace=True)
#
#     new_col = []
#     for i, row in enumerate(df[col].values):
#         if i % 10000 == 0:
#             print(i)
