# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:37:12 2020

@author: David Gomez
"""

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time, get_data

X_train, y_train, X_test, y_test = get_data.get_yahoo_finance_data('TSLA', 50, True)

model = Sequential()
model.add(LSTM(input_dim =1,
               output_dim=50,
               return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(100,
               return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print('compilation time:', time.time()-start)

model.fit(X_train, y_train, batch_size=512, nb_epoch=5, validation_split=0.05)

predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
lstm.plot_results_multiple(predictions, y_test, 50)