import scipy.stats as scs
import pandas as pd
import numpy as np
import re
import json
import requests
import nltk
from datetime import date, datetime, timedelta
from collections import Counter
import functions as func

## Load data into Python
data_2016 = pd.read_csv('data/ks-projects-201612.csv')
data_2018 = pd.read_csv('data/ks-projects-201801.csv')

## Eliminiate excess Columns
data_2016 = data_2016.iloc[:,:14]
data_2018 = data_2018.iloc[:,:14]

## Add Identifier row for which dataset it was pullef from
data_2016['dataset'] = 2016
data_2018['dataset'] = 2018

## Clean up column names
data_2016.columns = [x.strip() for x in data_2016.columns]
data_2018.columns = [x.strip() for x in data_2018.columns]

## Combine datasets
data = pd.concat([data_2016, data_2018])

## Identify bad imports (uses function from functions.py)
bad_imports = data[data['goal'].apply(func.is_bad_import)]
good_imports = data[~data['goal'].apply(~func.is_bad_import)]
print(bad_import.shape)

## Save combined DF as new .csv
good_imports.to_csv('data/data_nobadrows.csv', index=False)

###================================================================================================
'''The Proceeding code was intended to convert null usd values to relevant values based on the exchange rate, it utilizes the function contained within src/exchange.py.  Since this function utilized an API, it cannot be repeated (we used all avaliable free API calls for the month) Those values were cached within data/exchange.csv'''

## Read consolidated Dataframe from above
df = good_imports.copy()

## Eliminate duplicate ID columns (if present)
keep_cols = [x for x in df.columns if 'Unnamed' not in x]
df = df[keep_cols]

## Ensure Columns are of corret dtype
df.loc[:,'goal'] = df.loc[:,'goal'].astype(int)
df.loc[:,'pledged'] = df.loc[:,'pledged'].astype(int)
df.loc[:,'backers'] = df.loc[:,'backers'].astype(int)
df['launched'] = df['launched'].apply(func.get_date)
df['deadline'] = df['deadline'].apply(func.get_date)

## Find columns which have nulls in place of 'usd pledged'
get_null_usd = df[np.isnan(df['usd pledged'])]
dates = get_null_usd['launched'].unique()

## Read exchange rates generated from the 724 launch days idenfied in the dataset
exchange_rates = pd.read_csv('data/exchange.csv').set_index('date')

## Our API was not able to Gather data from the Danish currency, so we will be dropping these null values from our analysis
get_null_usd_ = get_null_usd[get_null_usd['currency'] != 'DKK']


## Exchange pledged value to USD based on exchange rate on that date 
get_null_usd_['usd pledged'] = get_null_usd_.T.apply(lambda x: x['pledged'] * \
                 exchange_rates.loc[str(x['launched']),'USD'] / \
                 exchange_rates.loc[str(x['launched']),x['currency']]) 

## Replace values in original dataframe with fixed currencies
df.loc[get_null_usd_.index, 'usd pledged'] = get_null_usd['usd pledged']

## Drop null DKK exchange values
df_ = df[~np.isnan(df['usd pledged'])]

## Save cleaned Dataframe as new file (not avaliable on github due to file size restictions)
df_.to_csv('data/data_with_usd.csv')