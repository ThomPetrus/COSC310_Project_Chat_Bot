# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:49:05 2020

@author: tpvan

    This is the first part that uses spacy and not nltk!
    
    In anaconda cmd prompt with admin:
        
    conda install -c conda-forge spacy
    python -m spacy download en
    
    
    Spacy is just a NLP library - as far as I am aware no ML capability, but works
    much more naturally than NLTK so far. 
    
    Basically you load in the library, and create a nlp(..) document object
    on a piece of text and it more or less processes most of the basics for you.
    Meaning doc is kind of a list of tokens for each word in the text
    You can just call certain functions or access attributes, for example
    doc.sents for all the sentenes etc. Each token similarily is processed and
    has attributes and methods, for example token.pos for it's part of speech.
    Note there's a difference in appending an underscore to the attributes
    in some cases : .pos and .pos_ --->  the .pos_ will give the string representation
    vs an integer id used internally.
"""
"""
-------------------------------------------------------------------------------
Spacy Basics
-------------------------------------------------------------------------------
"""

import spacy

# This is a small version of the larger library -> _lg
nlp = spacy.load('en_core_web_sm')

print("~~~~BASICS~~~")
# the prefix u is not needed, indicates unicode string -- seems to work fine without.
doc = nlp(u'Tesla is looking to buy U.S. startup for $6 million')

# text, part of speach, syntactic dependency 
for token in doc:
    print(token.text, token.pos_, token.dep_)

print("\n\n\n")    
doc2 = nlp(u"Tesla isn't looking into startups anymore")
for token in doc2:
    print(token.text, token.pos_, token.dep_)

print("\n\n\n")
# Sentence tokenization
doc = nlp(u"This is the first. This is the second. This is the last.")
for sentence in doc.sents:
    print(sentence)

# check first word 
if(doc[0].is_sent_start):
    print("Yep!")
    
"""
-------------------------------------------------------------------------------
Spacy - Tokenization - more examples
-------------------------------------------------------------------------------
"""
print("\n\n\n")
print("~~~~TOKENIZATION~~~")
myString = '"We\'re creating a chatbot in Kelowna B.C. !"'
doc = nlp(myString)

for token in doc:
    print(token.text)
    
doc2 = nlp(u"We\'re here to help! Send e-mail to support@mail.ca or visit us at http://www.buttsex.info")
for token in doc2:
    print(token)
    
# nr of tokens - len
print(len(doc2))

# specific token
print(doc2[15])


print("\n\n\n")
"""
-------------------------------------------------------------------------------
Spacy - Basic Named Entity Recognition
-------------------------------------------------------------------------------
"""
print("~~~~ENTITY/NOUNT CHUNK RECOGNITION~~~")
doc3 = nlp(u"Apple to build a Hong Kong factory for $6 million. Will employ thousands of children.")
    
for entity in doc3.ents:
    print(entity)
    print(entity.label_)
    print(str(spacy.explain(entity.label_)))
    print('\n')
    
# noun chunk recognition
doc4 = nlp("Autonomous cars uprising. Government falls under Uber army. Insurance liability lies with Elon Musk. Thanks Obama.")
for chunk in doc4.noun_chunks:
    print(chunk)
    
"""
-------------------------------------------------------------------------------
Spacy - Visualization - Not super relevant immediately - could be cool later with a GUI
-------------------------------------------------------------------------------
""" 
print("~~~~VISUALIZATION~~~")
from spacy import displacy

# style = dependencies, serves at local host port 5000
# http://127.0.0.1:5000/
doc = nlp("Autonomous cars uprising. Government falls under Uber army. Insurance liability lies with Elon Musk. Thanks Obama.")
displacy.serve(doc, style='dep')


"""
-------------------------------------------------------------------------------
Spacy - Stemmin-- nope. As mentioned Spacy does not use Stemming and favours 
Lemmatization which is the next set of Notes.
-------------------------------------------------------------------------------
"""
print("~~~~STEMMING~~~")
# Stemming --------------------------------------------------------------------------------------
# Spacy does not have a built in stemmer. bc unreliable -- uses lematization instead.
print("See NLTK notes. Spacy uses lemmatization instead.")

