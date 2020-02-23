# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:02:33 2020

NLTK Notes Stemming
Porter Stemmer / Snowball Stemmer


 Meh



@author: tpvan
"""

import nltk
from nltk.stem.porter import PorterStemmer

"""
-------------------------------------------------------------------------------
NLTK - Porter Stemmer
-------------------------------------------------------------------------------
"""

print("~~~~~Porter Stemmer~~~~")
p_stemmer = PorterStemmer()

words = ['run', 'runner', 'ran', 'running', 'runs', 'easily', 'fairly']
for word in words:
    print(word+'--->'+p_stemmer.stem(word))

print("Not awesome")    
print("\n\n")

"""
-------------------------------------------------------------------------------
NLTK - Snowball Stemmer
-------------------------------------------------------------------------------
"""

print("~~~~~Snowball Stemmer~~~~~~")

from nltk.stem.snowball import SnowballStemmer
s_stemmer = SnowballStemmer(language='english')

for word in words:
    print(word + "-->" + s_stemmer.stem(word))

print("Still not awesome.")

words = ['Generously','Generation','Generous','Generate']
for word in words:
    print(word + "-->" + s_stemmer.stem(word))
    
"""
Suggestion is to use lemmatization instead.
"""
