# -*- coding: utf-8 -*-
"""
@author: ardalan MEHRANI
@brief: loading datasets
"""

import zipfile
import pandas as pd

def load_from_zip(zip_filename, dtypes=None, parsedate=None, password=None, **kvargs):
        """
        Load file to dataframe.
        """
        with zipfile.ZipFile(zip_filename, 'r') as myzip:
            if password:
                myzip.setpassword(password)

            inside_zip_filename = myzip.filelist[0].filename

            if parsedate:
                pd_data = pd.read_csv(myzip.open(inside_zip_filename), sep=',', parse_dates=parsedate, dtype=dtypes,
                                      **kvargs)
            else:
                pd_data = pd.read_csv(myzip.open(inside_zip_filename), sep=',', dtype=dtypes, **kvargs)
            return pd_data, inside_zip_filename
