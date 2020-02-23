# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:10:45 2020

    Spacy - Vocabulary and Matching
    User defined pattern matching of vocab.
    Basically just a more involved way of pattern matching versus just
    using 'in' or equivalent code.

@author: tpvan
"""
"""
-------------------------------------------------------------------------------
imports
-------------------------------------------------------------------------------
"""
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import Matcher

"""
-------------------------------------------------------------------------------
Spacy - Matching:
    User defines patters as a list of dictionaries to match text.
-------------------------------------------------------------------------------
"""
#create matcher
matcher = Matcher(nlp.vocab)

# Solar Power
pattern1 = [ {'LOWER':'solarpower'} ]
# Solar-power - OP -> makes punctuation optional. * --> 0 or more times. Can be any punctuation
pattern2 = [ {'LOWER':'solar'}, {'IS_PUNCT':True, 'OP':'*'}, {'LOWER':'power'} ]

# Adding the patterns - 'None' for callbacks -- unexplained
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

# Just checking it works as advertised
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

"""
-------------------------------------------------------------------------------
Spacy - Phrase Matching:
    User defines a list of phrases to match.
-------------------------------------------------------------------------------
"""
print("\n\n\n\nPhrase Matching\n\n\n")

from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)

# random text file included in folder - NLP wiki article copypasta
with open("randomTextFile_NLP.txt") as f:
    doc3 = nlp(f.read())

# List of phrases we want to find
phrase_list = ['part of speech', 'tokenized', 'tokenizer']

# convert each phrase into a nlp doc object using list comprehension
phrase_patterns = [nlp(text) for text in phrase_list]

# the asterisk allows us to pass each argument in the list. 'List unpacking'
matcher.add('PhraseMatcher', None, *phrase_patterns)

found_matches = matcher(doc3)

# formatted output - the -5 and +5 adds extra tokens around the phrase to give context
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc3[ start -5 : end +5 ]
    print(match_id, string_id, start, end, span.text)

