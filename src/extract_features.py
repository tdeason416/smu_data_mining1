import scipy.stats as scs
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import re
import json
import src.functions as func
import nltk
import functions as func
from datetime import date, datetime, timedelta
from collections import Counter

'''
This script takes the cleaned up dataset and extracts additional features from the Time, name and catagorical based datasets.
this script takes a very long time to run (~40 min), as the dataset is large, and parts of speech tagging is resource intensive.
'''

###==================================================
### Add parts of speech tagging for Kickstarter Names
###==================================================
df = pd.read_csv('data/data_with_usd.csv')
keep_cols = [x for x in df.columns if 'Unnamed' not in x]
df = df[keep_cols]
## Extract name column from data and fill any Empty name columns with the string 'None'
names = df['name']
df.loc[:,'name'] = df.loc[:,'name'].fillna('None')
## Using regex, count 
#   * the number of letters, 
#   * the number of words, 
#   * the punctuation normalized vs letter count,
#   * the capital letter count normalized vs the letter count
lett_count = names.apply(lambda x: len(re.sub('[^a-z]', '', x.lower())))
word_count = names.apply(lambda x: len(re.findall("[a-z']+", x.lower())))
punc_count = names.apply(lambda x: len(re.findall("[\p{P}\d]", x))) / lett_count
caps_count = names.apply(lambda x: len(re.findall("[A-Z]", x))) / lett_count
## split Kickstarter names into lists of words
words = names.apply(lambda x: re.findall("[a-z']+", x.lower()))

## Use nltk to apply parts of speech tagging to word vectors extracted from kickstarter names
## then count how many of each parts of speech there is in each name
parts_of_speech = words.apply(lambda x: nltk.pos_tag(x))
just_pos = parts_of_speech.apply(lambda x: [i[1] for i in x])
pos_counts = just_pos.apply(get_count)

## Count the ratio of
#   * Possesives
#   * Nouns
#   * Adjectives
#   * Verbs
#   * Prepositions
#   * Determinates
## in each Kickstarter name
possesive_count = pos_counts.apply(lambda x: x.get('WP', 0)) / word_count
noun_count = pos_counts.apply(lambda x: x.get('NN', 0)) / word_count
adj_count = pos_counts.apply(lambda x: x.get('JJ', 0)) / word_count
verb_count = pos_counts.apply(lambda x: x.get('VB', 0)) / word_count
preposition_count = pos_counts.apply(lambda x: x.get('IN', 0)) / word_count
determinator_count = pos_counts.apply(lambda x: x.get('DT', 0)) / word_count

## Apply the count ratios to the original dataframe
df['name$word_count'] = word_count
df['name$punc_count'] = punc_count
df['name$caps_count'] = caps_count
df['name$possesive_count'] = possesive_count
df['name$noun_count'] = noun_count
df['name$adj_count'] = adj_count
df['name$verb_count'] = verb_count
df['name$preposition_count'] = preposition_count
df['name$determinator_count'] = determinator_count

## Save updated dataframe as new file (not included in github due to file size constraints)
df.to_csv('data/data_with_pos.csv')

###============================================
### Add dummy Columns for Catagorical Data
###============================================
ndf = pd.read_csv('data/data_with_pos.csv')
keep_cols = [x for x in ndf.columns if 'Unnamed' not in x]
ndf = ndf[keep_cols]

## For Each catagorical dataset, Create an additional column for each unique value within that dataset.
## Seperate the name of the original column from the dataset using a '&' character
for cat in df['category'].unique():
    ndf['category&{}'.format(cat)] = df['category'] == cat
for cat in df['main_category'].unique():
    ndf['main_category&{}'.format(cat)] = df['main_category'] == cat
for cat in df['country'].unique():
    ndf['country&{}'.format(cat)] = df['country'] == cat
for cat in df['currency'].unique():
    ndf['currency&{}'.format(cat)] = df['currency'] == cat
for cat in df['country'].unique():
    ndf['country&{}'.format(cat)] = df['country'] == cat    
ndf.drop(['category', 'main_category', 'country', 'currency', 'name', 'country'], axis=1, inplace=True)

## Endure time based columns are of datetime format
ndf['deadline'] = df.loc[:,'deadline'].apply(pd.Timestamp)
ndf['launched'] = df.loc[:,'launched'].apply(pd.Timestamp)

## Extract the deadline and launch date month from thier respective columns
ndf['deadline_month'] = ndf.loc[:,'deadline'].apply(lambda x: x.strftime('%b'))
ndf['launched_month'] = ndf.loc[:,'launched'].apply(lambda x: x.strftime('%b'))

## Extract the year which the kickstarter was launched
ndf['launched_year'] = ndf.loc[:,'launched'].apply(lambda x: x.strftime('%Y'))

## Extract the number of days the kickstarter was active
ndf['length'] = (ndf['deadline'] - ndf['launched']).apply(lambda x: x.days)

## Convert Launch and Deadline month to catagorical data by genrating dummy variables.
for cat in ndf['deadline_month'].unique():
    ndf['deadline_month&{}'.format(cat)] = ndf['deadline_month'] == cat 
for cat in ndf['launched_month'].unique():
    ndf['launched_month&{}'.format(cat)] = ndf['deadline_month'] == cat

## Drop non-numerical date based data from dataframe
ndf.drop(['launched_month',
          'launched_year',
          'deadline_month',
          'launched',
          'deadline'], axis=1, inplace=True)

## Save new dataframe (with all numerical and catagorical dummy variables) as new .csv -- not avaliable on github due to file size
ndf.to_csv('data/data_with_dummies.csv')


###============================================
### Create binary classification dataset
###============================================
ndf = pd.read_csv('data/data_with_dummies.csv')
## Create new row 'success' where wither the amount pledged has exceeded the goal, or the 'state' is successful
ndf['success'] = (ndf['state'] == 'successful') | (ndf['pledged'] > ndf['goal'])
## Drop the catagorical 'state' variable'
ndf.drop('state', axis=1, inplace=True)
keep_cols = [x for x in ndf.columns if 'Unnamed' not in x]
ndf = ndf[keep_cols]
## Save as new .csv file (not avalible on github because file size is too large)
ndf.to_csv('data/data_with_dummies_and_class.csv')