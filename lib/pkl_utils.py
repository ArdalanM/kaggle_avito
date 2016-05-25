"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief: utils
"""

import pickle


def _save(fname, data, protocol=3):
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol)

def _load(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)