import scipy.stats as scs
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import nltk
from datetime import date, datetime, timedelta
from collections import Counter
from sklearn import ensemble as ens
from sklearn import linear_model as lm
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


def is_bad_import(x):
    '''
    Checks if row is of numeric type.  If row is numeric, returns True, else returns false
    --------
    PARAMETERS
    x: str, bool, int, list, set, etc...
        -   Input to check
    --------
    RETURNS
    __: bool
        -   True if x is not numeric, False if x is numeric
    '''
    try:
        float(x)
        return True
    except:
        return False


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
    return date(int(dlst[0]), int(dlst[1]), int(dlst[2]))

def get_count(cell):
    '''
    converts nltk parts of speech tuples into counts of parts of speech
    '''
    cnt_dict = Counter()
    for i in cell:
        cnt_dict[i[:2]] += 1
    return cnt_dict

def get_feature_counts(df, sep='&'):
    '''
    extract number of catagories in each catagorical dataset
    features which are numeric will return a value of 1
    --------
    PARAMETERS
    df: pd.DataFrame
        -   All catagorical datasets are converted to numeric
    sep: str 
        -   string which seperates parent catagorical type from value
    --------
    RETURNS
    __ : pd.Series
        -   Contains column names as index, and count of catagorical data as values
    '''
    features = Counter()
    for col in df.columns:
        features[col.split(sep)[0].strip()] += 1
    return pd.Series(features)

def cv_build_model(df, y_col, nfolds=5, n_estimators=100, max_features='sqrt', 
                    criterion='gini', min_samples_leaf=1, random_state=42):
    '''
    Input Dataframe ready for analysis ouputs probabilities of y_col (y col must be binary or bool)
    --------
    PARAMETERS
    df: pd.DataFrame
        -   Must contain all numerical or bool values
    y_col: str
        -   must be a column within df and contain only either binary or bool values
    nfolds: int (default 5):
        -   number of times to split dataframe between train and test sets (min value should be 3)
    n_estimators: int:
        -   Number of decision trees to generate out of the dataset
    max_features: str ('sqrt' | 'log2' | 'None' ) {default= 'sqrt'}
        -   Max number of features allowed in one individual estimator
    criterion: str ('gini' | 'entropy' ) {default=  'gini'}
        -   Function used to determine information gain from a split
    min_samples_leaf: int default(1)
        -   the minimum number of samples in a final terminus for the decision tree
    random_state: int {default= 42}
        -   random seed for decision tree feature selection and bootstrap 
    --------
    RETURNS
    prob_folds, atcual: tuple (list of np.array dtype=float, list of np.array dtype=bool)
        -   prob_folds contains the probability that a given value in the test_set is equal to 1
        -   atcual contains the atcual value for that datapoint
    '''
    X = df.copy()
    y = X.pop(y_col)
    n_samples, n_features = X.shape
    ## Define random seed
    random_state = np.random.RandomState(random_state)
    # Run classifier with cross-validation 
    cv = StratifiedKFold(nfolds)
    classifier = ens.RandomForestClassifier(n_estimators=n_estimators,
                                            max_features=max_features, 
                                            bootstrap=True,
                                            n_jobs=-1,
                                            criterion=criterion,
                                            min_samples_leaf= min_samples_leaf,
                                            random_state=random_state) 
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    prob_folds = []
    atcual = []
    for train, test in cv.split(X, y):
        prob_folds.append(classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test]))
        atcual.append(y.iloc[test])
    return prob_folds, atcual


def cv_build_model_logres(df, y_col, nfolds=5, random_state=42):
    '''
    Input Dataframe ready for analysis ouputs probabilities of y_col (y col must be binary or bool)
    --------
    PARAMETERS
    df: pd.DataFrame
        -   Must contain all numerical or bool values
    y_col: str
        -   must be a column within df and contain only either binary or bool values
    nfolds: int (default 5):
        -   number of times to split dataframe between train and test sets (min value should be 3)
    random_state: int {default= 42}
        -   random seed for decision tree feature selection and bootstrap 
    --------
    RETURNS
    prob_folds, atcual: tuple (list of np.array dtype=float, list of np.array dtype=bool)
        -   prob_folds contains the probability that a given value in the test_set is equal to 1
        -   atcual contains the atcual value for that datapoint
    '''
    X = df.copy()
    y = X.pop(y_col)
    n_samples, n_features = X.shape
    ## Define random seed
    random_state = np.random.RandomState(random_state)
    # Run classifier with cross-validation 
    cv = StratifiedKFold(nfolds)
    classifier = lm.LogisticRegression(C=.6, max_iter=10000, n_jobs=-1, random_state=random_state) 
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    prob_folds = []
    atcual = []
    for train, test in cv.split(X, y):
        prob_folds.append(classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test]))
        atcual.append(y.iloc[test])
    return prob_folds, atcual

def find_auc(probs, atcual):
    '''
    Identify Area under ROC curve for a specific model
    --------
    PARAMETERS
    probs: np.array dtype= float
        -   probability of the value being True
    atcual np.array dtype= bool or int
        -   Atcual value of predictor variable as a bool or binary integer
    --------
    RETURNS (tuple with 3 values)
    roc_auc: float
        -   area under the curve for the given model
    fpr:
        -   ratio of atcual false values which were predicted true
    tpr:
        -   ratio of atcual true values predicted correctly
    '''
    fpr, tpr, thresholds = roc_curve(atcual, probs)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_auc(folds):
    '''
    plots ROC curve with area under curve for a given model
    --------
    PARAMETERS list of folds - (fpr, tpr, roc_auc)
    folds: list of indeterminite length tuples containing (fpr: array, tpr: array, roc_auc: float)
        * tpr: array dtype= float
            -   ratio of positive values predicted positive by the model
        * fpr: array dtype= float
            -   ratio of positive values predicted negative by the model
        * roc_auc: float
            -   area under the ROC curve
    --------
    RETURNS
    None: 
        - graph is returned from this funciton
    '''
    i=0
    for fpr, tpr, roc_auc in folds:
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Luck', alpha=.8)
    mean_tpr = np.array([np.mean(i) for i in zip(*[t[1] for t in folds])])
    mean_fpr = np.array([np.mean(i) for i in zip(*[t[0] for t in folds])])
    std_tpr = np.array([np.std(i) for i in zip(*[t[1] for t in folds])])
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std([i[2] for i in folds])
    print mean_auc 
    print std_auc
    plt.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()