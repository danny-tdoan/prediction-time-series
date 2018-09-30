import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (15.0, 7.5)

def load_all_stocks(file):
	"""load the pre-processed stocks from the csv"""
	data=pd.read_csv(file,parse_dates=True,index_col=['t','s'])
	
	#replicate the time and stock column for easier access
	data['time']=data.index.get_level_values('t')
	data['stock']=data.index.get_level_values('s')
	
	return data
	
def load_one_stock(stock_data,stock_name,dropna=True):
	stock=stock_data[stock_data['stock']==stock_name]
	
	if dropna:
		stock=stock.dropna()

	return stock

