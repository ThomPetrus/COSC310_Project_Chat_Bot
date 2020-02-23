# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:28:34 2020

Scikit-Learn - Notes

INSTAL :
    anaconda prompt w/ admin :
    conda install scikit-learn
    
basic process:
    import model - built in
    split data - built in
    fit data - literally model.fit()
    once trained ready to test - prediction_Y = model.predict(X_test)
    Evaluation method depends on what ML algo youre using
        regression, classification, clustering etc.
        
@author: tpvan
"""

import numpy as np
#reads csv / tsv files
import pandas as pd

# read in provided tab seperated file 'dataframe'
df = pd.read_csv('smsspamcollection.tsv', sep='\t')
print(df.head())

# some info display methods -- there's a ton on visualization as well skipped for now.
print('\n')
# ensure dataframe is complete: if all false == 0
print(df.isnull().sum())
print('\n')
print(df['label'].unique())
print('\n')
print(df['label'].value_counts())

# simple model using punctuation and length of messages to predict:
# first import model
from sklearn.model_selection import train_test_split

#X is feature data - 
X = df[['length', 'punct']]

#y is label data
y = df['label']

# then split - both X and y have test and train data. - set test size and random state integer
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# check out shape: shows (rows, columns)
# not in order because random!
print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_test)

# import the actual ML model - i.e. logisticRegression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
# All parameters can be changed and adjusted in constructor call
print(lr_model.fit(X_train, y_train))

# then check accuracy of model
from sklearn import metrics
predictions = lr_model.predict(X_test)

# Print info
print(metrics.confusion_matrix(y_test, predictions))
print('Only ~5 right spams. ok.')

print(metrics.classification_report(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))

# different model experiment ? 
from sklearn.naive_bayes import MultinomialNB
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
predictions = nb_model.predict(X_test)
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.classification_report(y_test, predictions))

print('Even worse.. lol -- TIME FOR TEXT FEATURE EXTRACTION! WAHOOO')


# Support vector classifier ... see next set of notes.
from sklearn.svm import SVC
svc_model = SVC(gamma='auto')
svc_model.fit(X_train, y_train)

predictions = svc_model.predict(X_test)
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.classification_report(y_test, predictions))
# slightly better.

"""
So obviously punctuation and length are not good parameters to train on.
We have to learn about text feature extraction 
"""

