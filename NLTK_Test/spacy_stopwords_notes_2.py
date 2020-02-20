# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:44:35 2020

Spacy - Stop words Notes

@author: tpvan
"""

import spacy
nlp = spacy.load('en_core_web_sm')

# All the stop words - 305 of em:
# print(nlp.Defaults.stop_words)

# Checking if stop word
print(nlp.vocab['is'].is_stop)

# adding stop words - pretty self-explanatory
nlp.Defaults.stop_words.add("btw")
nlp.vocab['btw'].is_stop = True

print(nlp.vocab['btw'].is_stop)

# Removing certain stopwords
# nlp.Defaults.stop_words.remove('beyond')
# nlp.vocab['beyond'].is_stop = False

# print(nlp.vocab['beyond'].is_stop)

# Example use case including punctuation filter
import re

filtered_words = []
text = "This is a sentence with a bunch of words, some of which are stop words. Please remove the stop words almighty python."
pattern = r"[^!.?, ]+"

# remove punctuation
no_punctuation_words = re.findall(pattern, text)

# remove stop words
for word in no_punctuation_words:
    if not (nlp.vocab[word].is_stop):
        filtered_words.append(word)
    
print(filtered_words)


