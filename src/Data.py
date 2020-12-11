"""
Assignment 3, COMP 472
Due December 13th, 2020
Use of Python 3.8
@author: Marc Vicuna, 40079109, El Hassan Ait Ouaziz, 26791573
"""

import pandas as pd
import numpy as np

class Data():
    def read_train_data(train_file):
        data = pd.read_csv(train_file, sep='\t', usecols=[1,2])
        return data['text'].values, data['q1_label'].values
    def read_test_data(test_file):
        data = pd.read_csv(test_file, sep='\t',header = None, usecols=[0,1,2])
        return np.array(data[0].values, dtype = 'i4'), data[1].values, data[2].values