"""
@author: Ardalan MEHRANI <ardalan.mehrani@iosquare.com>
@brief: distances
"""
import numpy as np
from math import radians

class SimpleGeoSimilarities():
    pass

class ColGeoFeatures(SimpleGeoSimilarities):

    def __init__(self, df):
        self.df = df

    def add_haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1 = self.df[lon1].apply(radians).values
        lat1 = self.df[lat1].apply(radians).values
        lon2 = self.df[lon2].apply(radians).values
        lat2 = self.df[lat2].apply(radians).values

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers. Use 3956 for miles
        return c * r

    def add_square_distance(selfself, lon1, lat1, lon2, lat2):
        pass

