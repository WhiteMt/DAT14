# data set directions for writing to csv file from .arff
# import for fast easy and expressive data structure
import pandas as pd
import numpy as np
import arff
import urllib2
import matplotlib.pyplot as plt
import re
# importing library ^^ for plot production and interactive 2D data viz

from sklearn.ensemble import RandomForestClassifier
%matplotlib inline
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             accuracy_score, roc_auc_score)

#loading data with path to file directions and naming dataset

data_arff = arff.load(open ('Dataset.arff', 'rb'))
# direction ^^ to load data, now stored in dict

column_names = [x[0] for x in data_arff['attributes']]
# direction ^^ to get column names by calling key 'attributes' getting first valuein each tuple

df = pd.DataFrame(data_arff['data'], columns = column_names)
# direction ^^ to load data into pandas data frame && set column names

df = df.astype(int)
# direction ^^ to change column types from 'object' to 'int'

df.Result = df.Result.map(lambda x: 0 if x <= -1 else 1)
# dict 1=phishing -1=nonphishing

#csv data ready
df.to_csv('phishingdata.csv')
df = pd.read_csv('phishingdata.csv')
df.head()

