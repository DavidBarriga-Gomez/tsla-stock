import get_data2
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from math import sqrt

x_train, y_train, x_test, y_test, scaler_fitted, seq_history_len, n_features = get_data2.get_yahoo_finance_data('TSLA', 1, 1)
inputs = keras.Input(shape=(1,x_train.shape[2]))
layer_1_output = layers.LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]))(inputs)
layer_2_output = layers.Dense(1)(layer_1_output)

model = keras.Model(inputs=inputs, outputs=layer_1_output, name="mnist_model")
model.summary()

model.compile(optimizer='sgd', loss='mae')
# history = model.fit(x_train, y_train, batch_size=512, epochs=2, validation_split=0.05)
history = model.fit(x_train, y_train, epochs=10, batch_size=72, validation_data=(x_test, y_test), verbose=2, shuffle=False)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(x_test)
x_test = x_test.reshape((x_test.shape[0], seq_history_len * n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, x_test[:, -n_features:]), axis=1)
print(inv_yhat.shape)
inv_yhat = scaler_fitted.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = concatenate((y_test, x_test[:, -7:]), axis=1)
inv_y = scaler_fitted.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


