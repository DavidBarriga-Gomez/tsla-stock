from YahooFinanceDataClient import YahooFinanceDataClient
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from math import sqrt

seq_history_len = 60
yahooClient = YahooFinanceDataClient()
processedDataModel = yahooClient.get_yahoo_finance_data('SPY',
                                                        seq_history_len,
                                                        5,
                                                        365*10)
inputs = keras.Input(shape=(seq_history_len, processedDataModel.x_train.shape[2]))
layer_1_output = layers.LSTM(60,
                             input_shape=(processedDataModel.x_train.shape[1], processedDataModel.x_train.shape[2]),
                             dropout=.05,
                             return_sequences=True)(inputs)
layer_2_output = layers.LSTM(60,
                             input_shape=(processedDataModel.x_train.shape[1], processedDataModel.x_train.shape[2]),
                             dropout=.05,
                             return_sequences=False)(layer_1_output)
layer_3_output = layers.Dense(1)(layer_2_output)

model = keras.Model(inputs=inputs, outputs=layer_3_output, name="mnist_model")

model.compile(optimizer='adam', loss='mse')
history = model.fit(processedDataModel.x_train, processedDataModel.y_train,
                    epochs=1, batch_size=200,
                    validation_data=(processedDataModel.x_test, processedDataModel.y_test), verbose=1,
                    shuffle=True)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(processedDataModel.x_test)
x_test_reshaped = processedDataModel.x_test.reshape((processedDataModel.x_test.shape[0], seq_history_len * processedDataModel.n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, x_test_reshaped[:, -processedDataModel.n_features:]), axis=1)
inv_yhat = processedDataModel.scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
y_test = processedDataModel.y_test.reshape((len(processedDataModel.y_test), 1))
inv_y = concatenate((y_test, x_test_reshaped[:, -processedDataModel.n_features:]), axis=1)
inv_y = processedDataModel.scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)