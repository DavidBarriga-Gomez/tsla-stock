import yfinance as yf
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat

class YahooFinanceDataClient:
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
    
    def get_yahoo_finance_data(self, ticker, seq_history_len, seq_prediction_len):
        start = '2010-01-01'
        end = date.today()
        yfResult = yf.download(ticker, start=start, end=end, progress=False, interval='1d')
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(yfResult)
        reframed = self.series_to_supervised(scaled, seq_history_len, seq_prediction_len, True)
       
        values = reframed.values    
        
        n_train_years = 5
        n_train_days = 365 * n_train_years    
        train = values[:n_train_days, :]
        test = values[n_train_days:, :]
        
        n_features = 5 #columns in input data set
        n_obs = seq_history_len * n_features
        x_train, y_train = train[:, :n_obs], train[:, -n_features]
        x_test, y_test = test[:, :n_obs], test[:, -n_features]
        x_train = x_train.reshape((x_train.shape[0], seq_history_len, n_features))
        x_test = x_test.reshape((x_test.shape[0], seq_history_len, n_features))
       
        return [x_train, y_train, x_test, y_test, scaler, n_features]
