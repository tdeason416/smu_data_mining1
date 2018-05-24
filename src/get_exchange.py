import pandas as pd
import time
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
from datetime import date, datetime, timedelta

def get_date(dt):
    '''
    convert datetime string to python datetime.date output
    --------
    PARAMETERS
    dt: str - Date or datetime input (can also be in datetime.datetime format)
    --------
    RETURNS
    dateout: datetime.date - truncated datetime as date
    '''
    dlst = str(dt).strip().split()[0].split('-')


df = pd.read_csv('data/data_nobadrows.csv')
df.columns  = [col.strip for col in df.columns]
currency_key = os.environ['curr_key']

df.loc[:,'goal'] = df.loc[:,'goal'].astype(int)
df.loc[:,'pledged'] = df.loc[:,'pledged'].astype(int)
df.loc[:,'backers'] = df.loc[:,'backers'].astype(int)
df.iloc[:,'launched'] = df.iloc[:,'launched'].apply(get_date)
df.iloc[:,'launched'] = df.iloc[:,'launched'].apply(get_date)

no_usd = df[np.isnan(df['usd pledged'])]

dates = no_usd.unique()
all_currencys = ','.join(no_usd['currency'])

url_dict = {'date' : 'XXXX-XX-XX',
           'key' : curr_key,
           'curs': currencys}
url_dict['date'] = dates[0]

url= "http://data.fixer.io/api/{date}?access_key={key}&symbols={curs}&format=1"
url_frmt = url.format(**url_dict): 

api_return = requests.get(url_frmt)
cols = ['date']
for val in api_return.json()['rates'].keys():
    cols.append(val)

with open('data/exchange.csv', 'w') as exchange:
    exchange.write(','.join(cols))

for date in dates:
    url_dict['date'] = date
    api_req = requests.get(url.format(**url_dict))
    api_json = api_req.json()
    out =  [api_json[cols[0]]]
    for col in cols[1:]:
        out.append(str(api_json['rates'][col]))
    with open('data/exchange.csv', 'ab') as exchange:
        exchange.write('\n'+','.join(out))
    #prevent api from DDOS limiting
    time.sleep(2)

exchange_rates = pd.read_csv('data/exchange.csv')

def times_exchange_rate(x):
    return x['pledged'] * \
        exchange_rates.loc[str(x['launched']),'USD'] / \
        exchange_rates.loc[str(x['launched']),x['currency']]

no_usd_nodkk = no_usd[no_usd['currency'] != 'DKK']

no_usd_nodkk.iloc[:,'usd_pledged'] = no_usd_nodkk.T.apply(times_exchange_rate)
df.loc[no_usd.index, 'usd pledged'] = no_usd['usd pledged']
df_ = df[~np.isnan(df['usd pledged'])]
df_.to_csv('data/data_with_usd.csv")
