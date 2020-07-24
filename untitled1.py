# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 18:41:25 2020

@author: David Gomez
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:37:12 2020

@author: David Gomez
"""

# from keras.layers.core import Dense, Activation, Dropout
# from keras.layers.recurrent import LSTM
# from keras.models import Sequential
import lstm, time, get_data
# from keras.utils.vis_utils import plot_model 

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


X_train, y_train, X_test, y_test = get_data.get_yahoo_finance_data('TSLA', 50, True)

inputs = keras.Input(shape=(len(X_train),))

x = layers.Dense(1)(inputs)

layer_1_ouput = layers.LSTM(input_dim =1,
                units=50,
                return_sequences=True)(inputs)

print(inputs)

# layer_2_output = layers.Dropout(0.2)(layer_1_ouput)

# layer_3_output = layers.LSTM(100, return_sequences=False)

# print(layer_3_ouput)

# model.add(LSTM(100,
#                return_sequences=False))
# model.add(Dropout(0.2))

# model.add(Dense(output_dim=1))
# model.add(Activation('linear'))

# outputs = layers.lstm(0.2)(x)

# model.add(Dense(1))
# model.add(Dense(output_dim=1))
# print(inputs)


# model = Sequential()
# model.add(LSTM(input_dim =1,
#                output_dim=50,
#                return_sequences=True))
# model.add(Dropout(0.2))

# model.add(LSTM(100,
#                return_sequences=False))
# model.add(Dropout(0.2))

# model.add(Dense(output_dim=1))
# model.add(Activation('linear'))

# start = time.time()
# model.compile(loss='mse', optimizer='rmsprop')
# print('compilation time:', time.time()-start)

# model.fit(X_train, y_train, batch_size=512, nb_epoch=5, validation_split=0.05)

# predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
# lstm.plot_results_multiple(predictions, y_test, 50)

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)