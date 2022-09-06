# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 21:28:39 2022

@author: salih.guder
"""

import warnings
from sklearn.cluster import KMeans
import random
import numpy as np
import pandas as pd
import scipy.spatial.distance as metric
import math
import sklearn.datasets as datasets
import time
from sklearn.preprocessing import normalize
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")
from itertools import combinations


iris = datasets.load_iris()
wine = datasets.load_wine()

iris_data = np.array(iris.data)
wine_data = np.array(wine.data)

wine_data_df = pd.DataFrame(wine_data)


'''Robust Scaler: When there are many instances of outliers in your dataset, 
you can normalize the data with the median divided by the 
IQR = the difference between the 75th and 25th percentiles of your data.'''

def robust_norm(ds):

    for feature in range(ds.shape[1]):
        mean = np.mean(ds[:,feature])
        stdev = np.std(ds[:,feature])
        ds[:,feature] = (ds[:,feature]-mean)/stdev
        print(f"Range of feature {feature+1} = {np.ptp(ds[:,feature])}")

    return ds

def robust_iqr_norm(ds:np.array)->np.array:
    
    for feature in range(ds.shape[1]):
        perc_25 = np.percentile(ds[:,feature],25)
        median = np.median(ds[:,feature])
        perc_75 = np.percentile(ds[:,feature],75)
        tresh = median/(perc_75-perc_25)
        ds[:,feature] = ds[:,feature]/tresh
        print(f"25 perc: {perc_25}\nMedian: {median}\n75perc: {perc_75}\nTreshold:{tresh}\n\n")
    return ds


