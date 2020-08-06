import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat

class YahooFinanceDataClient:
    def __init__(self):    
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def normalise_windows(self, window_data):
        smoothing_window_size = 150
        for di in range(0,10000,smoothing_window_size):
            window_data[di:di+smoothing_window_size,:] = self.scaler.fit_transform(window_data[di:di+smoothing_window_size,:])
            
        return window_data

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
    	n_vars = 1 if type(data) is list else data.shape[1]
    	df = DataFrame(data)
    	cols, names = list(), list()
    	# input sequence (t-n, ... t-1)
    	for i in range(n_in, 0, -1):
    		cols.append(df.shift(i))
    		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    	# forecast sequence (t, t+1, ... t+n)
    	for i in range(0, n_out):
    		cols.append(df.shift(-i))
    		if i == 0:
    			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    		else:
    			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    	# put it all together
    	agg = concat(cols, axis=1)
    	agg.columns = names
    	# drop rows with NaN values
    	if dropnan:
    		agg.dropna(inplace=True)
    	return agg
    
    def get_yahoo_finance_data(self, ticker, seq_history_len, seq_prediction_len, n_train_days):
        yfDataFrame = yf.Ticker(ticker).history(period="max", interval="1d")
        dateColumn = yfDataFrame.index
        yfDataFrame.insert(0,"Day", dateColumn.day)
        yfDataFrame.insert(0,"Month", dateColumn.month)
        yfDataFrame.columns = yfDataFrame.columns.str.replace(' ', '')
        yfDataFrame = yfDataFrame.drop(["Dividends", "StockSplits"], 1)
        n_features = len(yfDataFrame.columns)-1
        
        # transformed = self.scaler.fit_transform(yfDataFrame)
        # reframed_values = self.series_to_supervised(transformed, seq_history_len, seq_prediction_len, True).values
        # train = reframed_values[:n_train_days, :]
        # test = reframed_values[n_train_days:, :]
        
        reframed_values = self.series_to_supervised(yfDataFrame, seq_history_len, seq_prediction_len, True).values        
        train = self.scaler.fit_transform(reframed_values[:n_train_days, :])
        test = self.scaler.transform(reframed_values[n_train_days:, :])
        
        n_obs = seq_history_len * n_features
        x_train = train[:, :n_obs]
        y_train = train[:, -n_features]
        x_test = test[:, :n_obs]
        y_test = test[:, -n_features]
        x_train = x_train.reshape((x_train.shape[0], seq_history_len, n_features))
        x_test = x_test.reshape((x_test.shape[0], seq_history_len, n_features))
       
        return [x_train, y_train, x_test, y_test, n_features]
