# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:37:09 2020

Sample Text Classification Program - Movies

Trying out the text classification process using a provided data set
of movie review - similar to the nltk one.

@author: tpvan
"""

"""
--------------------------------------------------------------------
imports
--------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
"""
--------------------------------------------------------------------
Read data
--------------------------------------------------------------------
"""
# read in data file
df = pd.read_csv('moviereviews.tsv', sep='\t')

# here's the head, just a bunch of reviews
print(df.head())

"""
--------------------------------------------------------------------
'Clean' data
--------------------------------------------------------------------
"""
# check for missing data - there's 35 in original file.
print(df.isnull().sum())
# remove missing data
df.dropna(inplace=True)
print(df.isnull().sum())

# remove blank lines -- isspace method
blanks=[]

for index, label, review_text in df.itertuples():
    if review_text.isspace():
        blanks.append(index)

len(df)
# drop the blank lines
df.drop(blanks, inplace=True)
len(df)

"""
--------------------------------------------------------------------
Split data
--------------------------------------------------------------------
"""
from sklearn.model_selection import train_test_split

X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""
--------------------------------------------------------------------
Set up Pipeline
--------------------------------------------------------------------
"""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Pipeline
text_classifier = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
# fit data
text_classifier.fit(X_train, y_train)
# get predictions - i.e. test
predictions = text_classifier.predict(X_test)

"""
--------------------------------------------------------------------
Summary of Test
--------------------------------------------------------------------
"""
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print('\n\n\nMovie Review Data')
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

"""
--------------------------------------------------------------------
User input ? Needs a substantial amount of text to classifiy as positive. 
Seems to classifiy most short messages as neg
Assumption made is that the short reviews tend to be negative.
If you look at the data set most of the reviews are fairly long.
--------------------------------------------------------------------
"""
i =0;
while(i<3):
    prediction = text_classifier.predict([input("Enter a \'Review\':\n")])
    print(prediction)
    i+=1