import scipy.stats as scs
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import re
import json
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from collections import Counter

from sklearn import linear_model as lm
from sklearn import ensemble as ens
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

import functions as func

## load dataset
df = pd.read_csv('data/data_with_dummies_and_class.csv')
keep_cols = [x for x in df.columns if 'Unnamed' not in x]
df = df[keep_cols].set_index('ID')

## Drop all 'catagory' variables, as they were determined to dissproportionatly skew 
## the random forest feature selection
df_logis = dfl.drop([col for col in dfl.columns if col.split('&')[0] == 'category'], axis=1)

## remove columns which may contain direct reference to y-values, or could not be considered at the onset of a Kickstarter campaign
dfl = df.drop(['usd pledged', 'pledged', 'usd_pledged_real', 'backers'], axis=1)

## Train model
prob_folds, atcual = func.cv_build_model(df_logis, 'success', n_estimators=30)

## Identify Area under the ROC curve
tup = find_auc(prob_folds[0][:,1], atcual[0])

## plot ROC curve
plot_auc(tup)
