# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 22:33:00 2020

Spacy - Named Entity Recognition - Notes 2

@author: tpvan
"""
import spacy
nlp = spacy.load('en_core_web_sm')


# Function to print the entities.
def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text + ' - ' + ent.label_ + ' - ' + str(spacy.explain(ent.label_)))
    else:
        print('No entities found.')

print("\n\n\n")
doc = nlp(u"May I got to Washington, DC next May to see the Washington Monument?")
show_ents(doc)

print("\n\n\n")
doc = nlp(u'Tesla to build U.K. factory for $6 million')
show_ents(doc)

print("\n\n\n")
# Adding specific NER items to a specific doc ------------------------------------
# adding tesla :
from spacy.tokens import Span
ORG = doc.vocab.strings[u"ORG"]
new_entity = Span(doc, 0, 1, label=ORG)
doc.ents = list(doc.ents) + [new_entity]

show_ents(doc)


print("\n\n\n")
# Adding phrases or multiple things as NER - example is vacuums as a PRODUCT
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)

# create list and then doc objects from list
phrase_list = ['vacuum cleaner', 'vacuum-cleaner']
phrase_patterns = [nlp(text) for text in phrase_list]

# add those to matcher
matcher.add('new products', None, *phrase_patterns)

#sample text to doc
doc2 = nlp(u'Our company created a brand new vacuum cleaner', u'This new vacuum-cleaner is the best around')

#find matches 
found_matches = matcher(doc2)

from spacy.tokens import Span
PROD = doc2.vocab.strings[u"PRODUCT"]

new_ents = [Span(doc2, match[1],match[2], label=PROD) for match in found_matches]

doc2.ents = list(doc2.ents) + new_ents

show_ents(doc2)