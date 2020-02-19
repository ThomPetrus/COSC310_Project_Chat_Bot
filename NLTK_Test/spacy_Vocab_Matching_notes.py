# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:10:45 2020

Spacy - Vocabulary and Matching
User defined pattern matching of vocab.

@author: tpvan
"""

# import
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import Matcher

#create matcher
matcher = Matcher(nlp.vocab)


# list of dictionaries - user defined patterns
# trying to detect any variation on the value
# Solar Power
pattern1 = [{'LOWER':'solarpower'}]
# Solar-power - op - makes punct optional 0 or more times *, also can be any punct!
pattern2 = [{'LOWER':'solar'}, {'IS_PUNCT':True, 'OP':'*'}, {'LOWER':'power'}]


# None for callbacks -- adding the patterns
matcher.add('SolarPowerMatch', None, pattern1, pattern2)

# Sentence to process + matcher call.
doc = nlp(u"The Solar Power Industry continues to grow as solarpower blablba, Solar-Power is neato.")
found_matches = matcher(doc)

# formatted output
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(match_id, string_id, start, end, span.text)

# Can remove patterns from matcher w/ remove(...)

print('\n\n\n')
doc2 = nlp(u"Solar--Power is still solar power because of the OP")

found_matches = matcher(doc2)
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc2[start:end]
    print(match_id, string_id, start, end, span.text)

"""
THERE are a ton of different ways to create patterns to match with.
Google -> Spacy Matcher.
"""

#####################################################################
# --------------------------Phrase Matching-------------------------#
#####################################################################
print("\n\n\n\nPhrase Matching\n\n\n")
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)

# random text file included in folder - nlp wiki article
with open("randomTextFile_NLP.txt") as f:
    doc3 = nlp(f.read())

# List we want to find
phrase_list = ['part of speech', 'tokenized', 'tokenizer']

# convert each phrase into a nlp doc object
# the asterisk allows us to pass each argument in the list.
phrase_patterns = [nlp(text) for text in phrase_list]
matcher.add('PhraseMatcher', None, *phrase_patterns)

found_matches = matcher(doc3)

# formatted output
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc3[start:end]
    print(match_id, string_id, start, end, span.text)

