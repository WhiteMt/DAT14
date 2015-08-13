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


df.to_csv('phishingdata.csv')
df = pd.read_csv('phishingdata.csv')
df.head()
//#csv data ready

df.age_of_domain.value_counts()
df.age_of_domain.hist();
//#exploring with a quick look at Age of Domain feature with a histogram visualization

df.web_traffic.hist();
//# exploring a quick look at Web Traffic feature with histogram visualization

df_corr = df.corr()
//#correlation matrix variable e= df.corr

plt.rcParams['figure.figsize'] = (10.0, 50.0)
//#PLOT exploration

def plot_notnull(df, col, idx):
    return df.groupby(col).Result.mean().plot(kind = 'bar', title = col, ax=axs[idx]);

fig, axs = plt.subplots(10,1);
for ix, col in enumerate(corr_idx):
    print plot_notnull(df, col, ix);

f, ax = plt.subplots(figsize=9, 9))
//# plotting each feature for visual understanding

cmap = sns.diverging_palette(220, 10, as_cmpa=True)
//#color palate

sns.corrplot(df_corr, annot=False, sig_stars=False,
             diag_names=False, cmap=cmap, ax=ax)
//#plotting actual data

//#plotting large correlational matrix
//#<matplotlib.axes._subplots.AxesSubplot at 0x10fde06d0>

corr_series = df_corr.Result.abs().order(ascending = False)[1:]
//# ordering correlation + or - first, from highest to lowest correlation,
//# getting all in series starting at first integer position

corr_series[:10].plot(kind = 'bar');
corr_idx = corr_series [:10].index
corr_idx
//#plotting up to the 10th

//#plt.rcParams['figure.figsize'] = (10.0, 50.0)
//#def plot_notnull(df, col, idx):
//#    return df.groupby(col).Result.mean().plot(kind = 'bar', title = col, ax=axs[idx]);

//# fig, axs = plt.subplots(10,1)
#for ix, col in enumerate(corr_idx):
//#  print plot_notnull(df, col, ix);

//# for col in corr_idx:
        #print df.groupby(col).Result.mean()


df.groupby('URL_Anchor').Result.mean()
//#taking attribute of -1 and adding up all results
//#98% if it is -1 it will be either a nonfishing and 1 a fishing email

URL_df = df[['URL_of_Anchor', 'Result']]
URL_df.head(10)
//#group by is taking attribute

URL_df[URL_df.URL_of_Anchor == -1]
//#Results is 1 when email with attribute of -1

URL_df[URL_df.URL_of_Anchor == -1].Result.mean()
//#interesting stronger predictor, though 1s and -1s so we will change to a percentage

pd.scatter_matrix(df[[u'Domain_registeration_length',
                      u'Request_URL',
                      u'Links_in_tags']],
                  figsize=(8,8))
plt.suptitle('Scatterplot URL_of_Anchor',size=25)
//# more plotting


//#Scatter and Line Plot

grouped1 = df.groupby('having_IP_Address').Result.agg{{'mean', 'count'))
//#grouping feature to view mean and count

grouped1.index = ['having_IP_Address' + str(string) for string in grouped1.index]
grouped1
//#list comprehension appending collumn name to indeces

varsGrouped = pd.DataFrame()
for col in corr_series.index:
//#creating for loop for data frame

    grouped = df.groupby(col).Result.agg(['mean', 'count'])
//#grouping by a single column
    grouped.index = [col + str(string) for string in grouped.index]
    varsGrouped = pd.concat([varsGrouped, grouped])

varsGrouped

varsGrouped['count'].order(ascending = False).plot(kind = 'bar', figzise = (12,6));

//#importing scatter plot
from matplotlib.artist import setp
fig = plt.gcf()
//# setting axes and size for scatter plot
fig.set_size_inches(18.5, 10.5)
x = range(len(varsGrouped.index))
y = varsGrouped['mean']

//#function for setting vertical labels to the x axis
my_xticks = varsGrouped.index

plt.xticks(x, my_xticks)
plt.scatter(x,
            y,
            s = varsGrouped['count'],
            alpha = .5)
plt.xticks(rotation=90)
plt.ylim(-0.05,1.05)
plt.show()

varsGrouped
# dictionary
# 1 = phishing
# 0 = non phishing
#Prefix_Suffix0	0.316865	1174
# 31% of

corr_series.plot(kind = 'bar')
//#<matplotlib.axes._subplots.AxesSubplot at 0x11199fe90>

varsGrouped.head()
//#grouping all features to view mean and count, correlation importance

varsGrouped['mean'].plot(kind = 'kde')
//#should be weighted
//#<matplotlib.axes._subplots.AxesSubplot at 0x10e359850>
//#X
//#from sklearn.preprocessing import OneHotEncoder
//#enc = OneHotEncoder(sparse=False)

X = df.ix[:,:-1]
//# -1 is last collumn, all but Results
//# setting = to var

//#onehotlabelencoder can only use positive integers. addin 1 to entire df, therefore 2=1, 1=0, 0=-1
//#X = X + 1
//#X = enc.fit_transform(X)
//#onehotelabel encoder mostly decreased accuracy in models below, therefore leaving it out

y = df.Result
//#Result column



//#Predictive Model

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn import neighbors
n_neighbors=range(1, 101, 2)
scores = []
for n in n_neighbors:
    clf = neighbors.KNeighborsClassifier(n)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))
//#connect data to classification model and predictive ml alogrithm using Knearest neighbors
//# http://www.galvanize.com/blog/2015/05/28/classifying-and-visualizing-musical-pitch-with-k-means-clustering/

plt.plot(n_neighbors, scores, linewidth=3.0)
//#fiting model x-train, y-train
//#accuracy score plotted over 100 values of K

//#Knearest neighbors above
//#Accuracy score is good, though not very promising for use of prediction


//#Grouped Features

//#Using the Random Set (Section III-B), we tokenize each phishing URL by splitting it using
//#non-alphanumeric characters

//#Modularizing
//#function created for different models to iterate at same time, making code reusable
knn = neighbors.KNeighborsClassifier(1)
svc = svm.SVC(kernel='linear', probability=True)
nb = GaussianNB()
lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=100)

knn

k = str(rf).split('(')
//# for parameter names changing to a string splitting on open parenthesis for purpose of ending as a list
k[0]
//#getting first item in the list

score_dict = {}
//#to use in plt_Model

def plot_confusion_matrix(cm, cmap, title='Confusion matrix'):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
//#True label = what it actually is
    plt.xlabel('Predicted label')

def plot_roc_curve(y_test, p_proba):
//#calculates: false positive rate, true positive rate,
    fpr, tpr, thresholds = roc_curve(y_test, p_proba[:, 1])
    
    roc_auc = auc(fpr, tpr)
//#Plot ROC curve
    plt.plot(fpr, tpr, label= 'AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
//#random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('ROC')
    plt.legend(loc="lower right")

def plt_Model(X_train, y_train, X_test, y_test, clf, score_dict, cmap = plt.cm.Blues):
    fig = plt.figure()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print "Accuracy Score: ", accuracy_score(y_test, y_pred)
    
//#making score dict
    clf_list = str(clf).split('(')
    score_dict[clf_list[0]] = accuracy_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    print "\nConfusion Matrix:\n", cm
    plot_confusion_matrix(cm, cmap, title='Confusion matrix')
    p_proba = clf.predict_proba(X_test)
    fig = plt.figure()
    plot_roc_curve(y_test, p_proba)

plt_Model(X_train, y_train, X_test, y_test, knn, score_dict, cmap=plt.cm.Blues)
//#Nearest Neighbors

//#AUC = 0.955
//#Confusion Matrix:
//#[[328  18]
//#[ 10 258]]


plt_Model(X_train, y_train, X_test, y_test, svc, score_dict, cmap=plt.cm.BrBG)
//#Support Vector Machines (SVMs with rbf 'linear' kernel)

//#AUC = o.981
//#Confusion Matrix:
//#[[326  20]
//#[ 23 245]]


plt_Model(X_train, y_train, X_test, y_test, nb, score_dict, cmap=plt.cm.RdPu)
//#Naïve Bayes (NB)

//#AUC = 0.972
//#Confusion Matrix:
//#[[319  27]
//#[ 22 246]]


plt_Model(X_train, y_train, X_test, y_test, lr, score_dict, cmap=plt.cm.Pastel1_r)
//#Logistic Regression (LR)

//#AUC = 0.982
//#Confusion Matrix:
//#[[326  20]
//#[ 25 243]]


plt_Model(X_train, y_train, X_test, y_test, rf, score_dict, cmap=plt.cm.ocean)
//#Random Forest (RF)


//#AUC = 0.993
//#Confusion Matrix:
//#[[337   9]
//#[  9 259]]

rf.fit(X_train, y_train)
score_dict.keys()
score_df = pd.DataFrame(score_dict.values(), index = score_dict.keys(), columns = ['accuracy_score'])

score_df.plot(kind = 'bar', ylim= (.9,1));
//#plotting accuracy rate of models along same axis


fi = sorted(zip(rf.feature_importances_, df.columns), reverse=True)
fi_df = pd.DataFrame(fi).rename(columns = {0: 'feature_importance', 1 : 'column_name'}).set_index(['column_name'])

fi_df.plot(kind = 'bar', ylim= (0,.3));
//#plotting of five most important features indicative of most probably to host phishing emails, according to data set.

fi_df.ix[:5].plot(kind = 'bar', ylim= (0,.3));

fi_df.head()
//#most predictive importance in accuracy for model

//#Nearest Neighbors
//#plt_Model(X_train, y_train, X_test, y_test, knn,  cmap=plt.cm.Blues)

df.columns
df.describe()

for x in ['knn 0.954397394137', 'svm 0.92996742671', 'nb 0.920195439739', 'lr 0.92671009772', 'rf 0.962540716612']:
    print x

//#access models, index, plot graph, specify tuple
//#plt.plot(,scores, linewidth=3.0)

ModelScore = (0.954397394137, 0.92996742671, 0.920195439739, 0.92671009772, 0.962540716612)
print rf
print ModelScore
print x


//#index data archive http://archive.ics.uci.edu/ml/machine-learning-databases/00327/

//#weka http://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff

//#repo https://archive.ics.uci.edu/ml/datasets/Phishing+Websites