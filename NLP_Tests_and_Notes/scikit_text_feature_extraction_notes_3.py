# -*- coding: utf-8 -*-
"""
----------------------------------------------------------------------------------        
Created on Wed Feb 19 16:34:16 2020

scikit - text feature extraction

@author: tpvan
----------------------------------------------------------------------------------        
"""

"""
------------------------------------------------------------------------------------
CountVectorizer - fyi - not really worth using here.
returns a document term matrix DTM - representation of al the unique 
words and whether they occur in a message
------------------------------------------------------------------------------------
"""

"""
# imports for regular countvectorizer :
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()

#-counts occurences of unique words
vect.get_feature_names()
"""


"""
----------------------------------------------------------------------------------        
Alternative to CountVectorizer is TfidfVectorizer
-Term frequency inverse document frequency:
    Term Frequency - raw count of a term in a document
        Tf alone will place too much importance on common words therefore:
    Inverse Document Frequency Factor - diminishes the weigh of terms that occur very frequently in the doc
        and increases the weight of terms that occur rarely (tho I'd image filtering stop words would still be a benefit?)
        - It is the log scaled inverse fraction of: docs that contain the term:
            i.e. divide the total nr of docs (N) by the number of docs that contain the term (doc(t)), then take the log of the quotient.
         
        - TFIDF = termfrequency * (log(N/|doc(t)==true|)) -- pseudo .. google for formula.
----------------------------------------------------------------------------------        
 """
messages = ["Hey, lets go to the game today", "Call you sister.", "Want to go walk your dogs?"]
 
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
dtm = vect.fit_transform(messages)
print(dtm)
 
"""
----------------------------------------------------------------------------------        
Building a vocabulary - example 2 texts (1.txt, 2.txt)
----------------------------------------------------------------------------------        
"""
vocab = {}
i = 1

with open('1.txt') as f:
    x = f.read().lower().split()
    
    for word in x:
        if word in vocab:
            continue
        else:
            vocab[word]=i
            i+=1

with open('2.txt') as f:
    x = f.read().lower().split()
    
    for word in x:
        if word in vocab:
            continue
        else:
            vocab[word]=i
            i+=1

print(vocab)

# create an empty vector with space for each word in the vocab
one = ['1.txt']+[0]*len(vocab)
with open('1.txt') as f:
    x = f.read().lower().split()

    for word in x:
        one[vocab[word]]+=1

print(one)

# Same same for vector numero 2
two = ['2.txt']+[0]*len(vocab)
with open('2.txt') as f:
    x = f.read().lower().split()

    for word in x:
        two[vocab[word]]+=1

print(two)
"""
----------------------------------------------------------------------
Basic Vectorization
----------------------------------------------------------------------
"""
import numpy as np
import pandas as pd

df = pd.read_csv('smsspamcollection.tsv', sep='\t')
# ensure complete == 0 ?
print(df.isnull().sum())
print(df['label'].value_counts())

# split the data : same as before
from sklearn.model_selection import train_test_split
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# vectorize - first count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

#-- fit vectorizer to data (build vocabulary, counts nr of words etc)
#count_vect.fit(X_train)

#-- Transform the original text message to vector
#X_train_counts = count_vect(X_train)

# you can do both simultaneously: 
X_train_counts = count_vect.fit_transform(X_train)
print(X_train_counts)

# second number represents the number of unique words.
print(X_train.shape)
print(X_train_counts.shape)

"""
----------------------------------------------------------------------
Vectorization - Using tfidf
----------------------------------------------------------------------
"""
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

X_train_tfidf= tfidf_transformer.fit_transform(X_train_counts)

# It is super common to perform a count vectorization followed by TFIDF - once again can be combined in one step:
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

# Next train the Support vector classifier using tfidf data
from sklearn.svm import LinearSVC

clf = LinearSVC()
print(clf.fit(X_train_tfidf, y_train))


print('\n')
"""
----------------------------------------------------------------------
######################################################################
The Meat and potatoes comin up ~ combining it into a pipeline 
######################################################################

Only our training data has currently been vectorized into a full vocabulary. In order
to perform an analysis we would have to do the same for the test set. We can use a pipeline class to
perfrom all of this in one go! Woah ikr.

Takes in a list of tuples.
behave the same as everything we've seen previously in a single call.
So vectorizes and running a classifier on data in one step.

Even more can be included, like text feature extraction, lemmatization, stop word removal etc.
----------------------------------------------------------------------
"""
from sklearn.pipeline import Pipeline
text_classifier = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])

# note that it takes in the original raw data and performs the desire actions :O woot
print(text_classifier.fit(X_train, y_train))
# sew much easier :D

# Test on test data
predictions = text_classifier.predict(X_test)

# show results
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))
print("Hiiiiiits it out of the park bb ~ now to adapt it to a chat bot. err.")

# NOW to use it on user input:
i=0
while(i<3):
    prediction = text_classifier.predict([input("Enter a message:")])
    print(prediction)
    i+=1

"""
Works fairly well
Obviously spammy messages get recognized. Too short and it wont work.
Congratulations! You've been selected as a winner! Text won to 44444 for free entries.

Mind you fraudelent messages are not detected i.e.
I am an arabian prince, I want to give you all my money etc.

If you look at the training set the messages are very explicitly 'spam'

"""