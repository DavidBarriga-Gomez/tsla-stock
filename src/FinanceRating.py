import requests
import pandas as pd
from pandas import DataFrame
from pandas import concat

r = requests.get('https://financialmodelingprep.com/api/v3/historical-rating/AAPL?apikey=cf8050149d7fc94c4a855a1409640b36')
#convert to panda df (adding colums name)   
dF = pd.DataFrame.from_records(r.json())
dF_dropped= dF.drop(['ratingDetailsDERecommendation','symbol','rating','ratingDetailsPERecommendation','ratingDetailsPBRecommendation','ratingRecommendation','ratingDetailsDCFRecommendation','ratingDetailsROERecommendation','ratingDetailsROARecommendation',],1)
print(dF_dropped)


 
def series_to_supervised(dF, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(dF) is list else dF.shape[1]
	df = DataFrame(dF)
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
#values = [x for x in range(10)]
big_data = series_to_supervised(pd.DataFrame.from_records(dF_dropped), 5)
print(big_data)




