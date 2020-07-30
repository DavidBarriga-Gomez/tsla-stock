# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 20:43:27 2020

@author: ninja
"""

import yfinance as yf
import numpy as np
from datetime import date

def normalise_windows(window_data):
   normalised_data = []
   for column in range(len(window_data)-1):
       for window in window_data[column]:
           normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
           normalised_data.append(normalised_window)
   return normalised_data

def get_yahoo_finance_data(ticker, seq_len, normalise_window):
    start = '2010-01-01'
    end = date.today()
    yfResult = yf.download(ticker, start=start, end=end, progress=False, interval='1d')

    numpyData = yfResult.to_numpy()
    
    sequence_length = seq_len + 1
    
    data = numpyData
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    # if normalise_window:
    #     result = np.sin(result)
        # result = normalise_windows(result)
        
    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    y_traincolumn = train[:,0,3] #Get closing column as its own array
    ytrain = y_traincolumn[:int(row)]
    x_train = train[:, :-1]
    y_train = ytrain[:]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]
