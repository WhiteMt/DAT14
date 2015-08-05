//# data set directions for writing to csv file from weka .arff
//# import for fast easy and expressive data structure
import pandas as pd
import numpy as np
import arff
import urllib2
import matplotlib.pyplot as plt
import re
//# importing library ^^ for plot production and interactive 2D data viz

from sklearn.ensemble import RandomForestClassifier
%matplotlib inline
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             accuracy_score, roc_auc_score)

//#loading data with path to file directions and naming dataset

data_arff = arff.load(open ('Dataset.arff', 'rb'))
//# direction ^^ to load data, now stored in dict

column_names = [x[0] for x in data_arff['attributes']]
//# direction ^^ to get column names by calling key 'attributes' getting first valuein each tuple

df = pd.DataFrame(data_arff['data'], columns = column_names)
//# direction ^^ to load data into pandas data frame && set column names

df = df.astype(int)
//# direction ^^ to change column types from 'object' to 'int'

df.Result = df.Result.map(lambda x: 0 if x <= -1 else 1)
//# dict 1=phishing -1=nonphishing

//#load data into csv format

{
  "cells":  [
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
//#Go to https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff\n
//#right click on the page and click 'Save As' and name something ending with .arff
      ]
    },
    {
     "cell_type": "code",
     "execution_count": 34,
     "metadata": {
      "collapsed": false
    },
    "outputs": [
      {
        "name": "stdout",
        "output_type": "stream",
        "text": [
        "\u001b[33mYou are using pip version 7.0.1, however version 7.0.3 is available.\n",
        "You should consider upgrading via the 'pip install --upgrade pip' command.\u001b[0m\n",
        "Requirement already satisfied (use --upgrade to upgrade): liac-arff in ./anaconda/lib/python2.7/site-packages\n"
       ]
      }
     ],
    "source": [
//# install 'liac-arff', a python module that can load arff files\n
      "!pip install liac-arff"
     ]
    },
    {
     "cell_type": "code",
      "execution_count": 36,
      "metadata": {
       "collapsed": false
    },
    "outputs": [],
    "source": [
     "# import pandas and arff\n",
     "import pandas as pd\n",
     "import arff\n",
     "import urllib2"
    ]
   },
   {
    "cell_type": "markdown",
     "metadata": {},
     "source": [
//# load the data with the path to the file and the name you gave it
     ]
    },
    {
     "cell_type": "code",
     "execution_count": 3,
     "metadata": {
     "collapsed": false
    },
     "outputs": [],
     "source": [
//# load the data, which is now stored in a dictionary\n
      "\n",
      "data_arff = arff.load(open('TheDirectoryItIsIn/TheNameYouGaveIt.arff', 'rb'))"
     ]
    },
    {
     "cell_type": "code",
     "execution_count": 16,
     "metadata": {
      "collapsed": false
    },
      "outputs": [],
      "source": [
//# get the column names by calling the key 'attributes' and getting the first value in each tuple\n
       "column_names = [x[0] for x in data_arff['attributes']]"
     ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
       "collapsed": false
    },
      "outputs": [],
       "source": [
//# load the data into a pandas data frame and set the column names\n
        "df = pd.DataFrame(data_arff['data'], columns = column_names)"
    ]
   },
   {
    "cell_type": "code",
    "execution_count": 30,
    "metadata": {
      "collapsed": false
    },
    "outputs": [],
    "source": [
//# change the column types from 'object' to 'int'\n
     "df = df.astype(int)"
    ]
   },
   {
     "cell_type": "code",
     "execution_count": null,
     "metadata": {
     "collapsed": true
    },
    "outputs": [],
    "source": [
   ]
  }
 ],
  "metadata": {
    "kernelspec": {
    "display_name": "Python 2",
    "language": "python",
    "name": "python2"
   },
   "language_info": {
   "codemirror_mode": {
   "name": "ipython",
   "version": 2
   },
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython2",
    "version": "2.7.10"
   }
  },
   "nbformat": 4,
   "nbformat_minor": 0
}

//#csv data ready
df.to_csv('phishingdata.csv')
df = pd.read_csv('phishingdata.csv')
df.head()

//#exploring with a quick look at Age of Domain feature with a histogram visualization
df.age_of_domain.value_counts()
df.age_of_domain.hist();

//# exploring a quick look at Web Traffic feature with histogram visualization
df.web_traffic.hist();

//#correlation matrix variable e= df.corr
df_corr = df.corr()

//#PLOT exploration
plt.rcParams['figure.figsize'] = (10.0, 50.0)

def plot_notnull(df, col, idx):
    return df.groupby(col).Result.mean().plot(kind = 'bar', title = col, ax=axs[idx]);

fig, axs = plt.subplots(10,1);
for ix, col in enumerate(corr_idx):
    print plot_notnull(df, col, ix);

//# plotting each feature for visual understanding
f, ax = plt.subplots(figsize=9, 9))

//#color palate
cmap = sns.diverging_palette(220, 10, as_cmpa=True)

//#plotting actual data
sns.corrplot(df_corr, annot=False, sig_stars=False,
             diag_names=False, cmap=cmap, ax=ax)

//#plotting large correlational matrix
//#<matplotlib.axes._subplots.AxesSubplot at 0x10fde06d0>

//# ordering correlation + or - first, from highest to lowest correlation,
//# getting all in series starting at first integer position
corr_series = df_corr.Result.abs().order(ascending = False)[1:]

//#plotting up to the 10th
corr_series[:10].plot(kind = 'bar');
corr_idx = corr_series [:10].index
corr_idx

//#plt.rcParams['figure.figsize'] = (10.0, 50.0)
//#def plot_notnull(df, col, idx):
//#    return df.groupby(col).Result.mean().plot(kind = 'bar', title = col, ax=axs[idx]);

//# fig, axs = plt.subplots(10,1)
#for ix, col in enumerate(corr_idx):
//#  print plot_notnull(df, col, ix);

//# for col in corr_idx:
        #print df.groupby(col).Result.mean()

//#Predictive Model


