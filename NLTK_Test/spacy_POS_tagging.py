# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:57:47 2020

Spacy - Part of Speech Tagging

@author: tpvan
"""

import spacy
nlp = spacy.load('en_core_web_sm')

# The following converts each word to a token - on which we can retrieve information
doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")

# pos_ returns string representation of pos tag.
# google spacy pos tags for definitions. tag_ returns fine grained pos tags
for token in doc:
    print(f"{token.text:{10}}  {token.pos_:{10}}  {token.tag_:{10}}  {spacy.explain(token.tag_):{50}}")

# Count the number of a pos occurences 
POS_counts = doc.count_by(spacy.attrs.POS)

for k, v in sorted(POS_counts.items()):
    print(f"{k}. {doc.vocab[k].text:{5}} {v}")

DEP_count = doc.count_by(spacy.attrs.DEP)
TAG_count = doc.count_by(spacy.attrs.TAG)

for k, v in sorted(TAG_count.items()):
    print(f"{k}. {doc.vocab[k].text:{5}} {v}")
    
