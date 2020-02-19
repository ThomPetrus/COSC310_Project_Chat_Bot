# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:12:37 2020

Spacey - Lemmatization

Lemmatization looks beyond simple word reduction.
Takes context into account.

@author: tpvan
"""

import spacy
nlp=spacy.load('en_core_web_sm')

# Lemmatization --------------------------------------------------------
print("~~~~~~~~~~~ Lemmatization ~~~~~~~~~~~~~~")
doc = nlp("Last week I ran in a race because I love to run. I have been running since I was wee runner.")


for token in doc:
    print(token.text, '\t', token.pos_, token.lemma, '\t', token.lemma_)

print("That's better vs Tokenization")
print('\n\n\ns')

# cheeky function for format -------------------------------------------
def show_lemmas(text):
    for token in text:
        print(f'{token.text:{12}}{token.pos_:{6}}{token.lemma:<{22}}{token.lemma_}')

show_lemmas(doc)

"""
Everythin can be accessed using regular list syntax
"""

print("\n\n\n")
print(doc)

for token in doc:
    if("VERB" in token.pos_):
        print(token.text)


