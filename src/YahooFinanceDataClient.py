import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from LSTMProcessedData import LSTMProcessedData
import numpy
from collections import deque
from sklearn.model_selection import train_test_split

class YahooFinanceDataClient:

    def get_data_lstm_processed(self, ticker, seq_history_len, seq_prediction_len, n_train_percentage):
        yfDataFrame = yf.Ticker(ticker).history(period="max", interval="1d")
        dateColumn = yfDataFrame.index
        yfDataFrame.insert(0,"Day", dateColumn.day)
        yfDataFrame.insert(0,"Month", dateColumn.month)
        yfDataFrame.columns = yfDataFrame.columns.str.replace(' ', '')
        yfDataFrame = yfDataFrame.drop(["Dividends", "StockSplits"], 1)
        n_features = len(yfDataFrame.columns)

        column_scaler = {}
        preprocessed_data = yfDataFrame.copy()
        # scale the data (prices) from 0 to 1
        for column in yfDataFrame.columns:
            scaler = MinMaxScaler()
            column_data = yfDataFrame[column]
            # column_data = yfDataFrame[column].to_numpy().reshape(-1, 1)
            # expanded_data = numpy.expand_dims(column_data, axis=1)
            a =scaler.fit_transform(numpy.expand_dims(column_data, axis=1))
            preprocessed_data[column] = a
            column_scaler[column] = scaler
       
        # add the target column (label) by shifting by `lookup_step`
        preprocessed_data['future'] = preprocessed_data['Close'].shift(-seq_prediction_len)
        # last `lookup_step` columns contains NaN in future column
        # get them before droping NaNs
        last_sequence = numpy.array(preprocessed_data[yfDataFrame.columns].tail(seq_prediction_len))
        # drop NaNs
        preprocessed_data.dropna(inplace=True)
        
        sequence_data = []
        sequences = deque(maxlen=seq_history_len)
        for entry, target in zip(preprocessed_data[yfDataFrame.columns].values, preprocessed_data['future'].values):
            sequences.append(entry)
            if len(sequences) == seq_history_len:
                sequence_data.append([numpy.array(sequences), target])
        
        # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
        # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
        # this last_sequence will be used to predict in future dates that are not available in the dataset
        last_sequence = list(sequences) + list(last_sequence)
        
         # shift the last sequence by -1
        last_sequence = numpy.array(DataFrame(last_sequence).shift(-1).dropna())
        # add to result
        # construct the X's and y's
        X, y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)
        # convert to numpy arrays
        X = numpy.array(X)
        y = numpy.array(y)
        # reshape X to fit the neural network
        X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
        # split the dataset
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1-n_train_percentage, shuffle=True)
        dataModel = LSTMProcessedData(yfDataFrame, x_train, y_train, x_test, y_test, n_features, column_scaler, last_sequence)
        return dataModel