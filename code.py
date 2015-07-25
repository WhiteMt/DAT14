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

