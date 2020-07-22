#for this to work pip install yfinance
import yfinance as yf
import pandas as pd
import lxml
import json
from datetime import date
from pandas_datareader import data as pdr

# ticker_list = ["TSLA"]
t = ['TSLA']
a = []
start = '2020-01-01'
end = '2020-01-31'
for str in t:
    stock = yf.Ticker(str)
    a.append(yf.download('TSLA', start=start, end=end, progress=False, interval='60m'))
print (a)
