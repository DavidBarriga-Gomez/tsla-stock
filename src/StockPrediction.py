from YahooFinanceDataClient import YahooFinanceDataClient
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
from LSTMModelGenerator import LSTMModelGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import os

# Window size or the sequence length
N_STEPS = 60
# Lookup step, 1 is the next day
LOOKUP_STEP = 5
# test ratio size, 0.2 is 20%
TRAIN_SIZE = 0.75
# date now
date_now = time.strftime("%Y-%m-%d")
### model parameters
N_LAYERS = 3
# LSTM cell
CELL = layers.LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = True
### training parameters
# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 75
TICKER = "SPY"

np.random.seed(314)
tf.random.set_seed(314) #this seeds the weights so that the model is the same every run

yahooClient = YahooFinanceDataClient()
processedDataModel = yahooClient.get_data_lstm_processed(TICKER, N_STEPS, LOOKUP_STEP, TRAIN_SIZE)
shape = (processedDataModel.x_train.shape[1], N_STEPS)
inputs = keras.Input(shape=shape)
output_layer = LSTMModelGenerator().create_lstm_layers(inputs=inputs, units=256, cell=layers.LSTM, n_layers=2, dropout=.3, bidirectional=BIDIRECTIONAL)

model = keras.Model(inputs=inputs, outputs=output_layer, name="mnist_model")

model.compile(optimizer='adam', metrics=["mean_absolute_error"], loss=LOSS)

ticker_data_filename = os.path.join("data", f"{TICKER}_{date_now}.csv")
# model name to save, making it as unique as possible based on parameters
model_name = f"{date_now}_{TICKER}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"
    
# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")

# some tensorflow callbacks
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

history = model.fit(processedDataModel.x_train, processedDataModel.y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(processedDataModel.x_test, processedDataModel.y_test),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)

model.save(os.path.join("results", model_name) + ".h5") 

model_path = os.path.join("results", model_name) + ".h5" 
model.load_weights(model_path) 

# evaluate the model
mse, mae = model.evaluate(processedDataModel.x_test, processedDataModel.y_test, verbose=0)
# calculate the mean absolute error (inverse scaling)
mean_absolute_error = processedDataModel.column_scalers["Close"].inverse_transform([[mae]])[0][0]
print("Mean Absolute Error:", mean_absolute_error)

def predict(model, data, classification=False):
    # retrieve the last sequence from data
    last_sequence = data.last_sequence[:N_STEPS]
    # retrieve the column scalers
    column_scalers = data.column_scalers
    # reshape the last sequence
    last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    predicted_price = column_scalers["Close"].inverse_transform(prediction)[0][0]
    return predicted_price

future_price = predict(model, processedDataModel)
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")


def plot_graph(model, data):
    y_test = data.y_test
    X_test = data.x_test
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data.column_scalers["Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data.column_scalers["Close"].inverse_transform(y_pred))
    # last 200 days, feel free to edit that
    plt.plot(y_test[-200:], c='b')
    plt.plot(y_pred[-200:], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()
    
plot_graph(model, processedDataModel)

def get_accuracy(model, data):
    y_test = data.y_test
    X_test = data.x_test
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data.column_scalers["Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data.column_scalers["Close"].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
    return accuracy_score(y_test, y_pred)

print(str(LOOKUP_STEP) + ":", "Accuracy Score:", get_accuracy(model, processedDataModel))