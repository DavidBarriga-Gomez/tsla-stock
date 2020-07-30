# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:37:12 2020

@author: David Gomez
"""
import lstm, time, get_data2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

x_train, y_train, x_test, y_test = get_data2.get_yahoo_finance_data('TSLA', 50, True)
inputs = keras.Input(shape=(50,6))
layer_1_output = layers.LSTM(6,
                return_sequences=True)(inputs)
print(layer_1_output)
model = keras.Model(inputs=inputs, outputs=layer_1_output, name="mnist_model")
model.summary()

model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())
model.fit(x_train, y_train, batch_size=512, nb_epoch=2, validation_split=0.05)

predictions = lstm.predict_sequences_multiple(model, x_test, 50, 50)
lstm.plot_results_multiple(predictions, y_test, 50)