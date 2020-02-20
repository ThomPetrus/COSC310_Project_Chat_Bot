# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:26:04 2020

Install Instructions For NLTK found at:
https://www.nltk.org/install.html

All code derived from: 
1. https://pythonspot.com/category/nltk/
2. 


    The majority of the other notes use Spacy / scikit-learn / Keras
    Will come back to NLTK if need be.
    

@author: ThomPetrus
"""
import nltk
# word stem import
from nltk.stem import PorterStemmer

# basics import
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# import for classifying words
from nltk.tokenize import PunktSentenceTokenizer
from PIL import Image

# Name predict / Seniment analysis import
import nltk.classify.util
from nltk.corpus import names
from nltk.classify import NaiveBayesClassifier


data = "In computer science 310 we are building a chat bot! We are using python because why not. NLTK is the toolkit."
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Basics:

# Words seperated
words_tokenized = word_tokenize(data)
print("\nTokenize words:")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
print(words_tokenized)

# Sentences seperated
sentences_tokenized = sent_tokenize(data)
print("\nTokenize Sentences:")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
print(sentences_tokenized)

# Elimate Stop Words
stop_words = set(stopwords.words('english'))
words_filtered = []

for w in words_tokenized:
    if w not in stop_words:
        words_filtered.append(w)
    
print("\nFilter out the stop words:")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
print(words_filtered)

# finding the stem of words - generalizing topic at hand
ps = PorterStemmer()
related_words = ["learning", "learned", "learns", "algorithmic", "algorithmically","algorithms"]

print("\nFind the stem of the words:")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
for word in related_words:
    print(ps.stem(word))

# Does not work as well on complicated sentences
sentence = "We are learning about filtering words by their stem. This will not work as well on complicated sentences though";

print("\nWithout stop word filter:")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
words = word_tokenize(sentence)
for word in words:
    print(word + ":" + ps.stem(word))
    
print("\nWith stop word filter")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
words_filtered = []
for w in words:
    if w not in stop_words:
        words_filtered.append(w)

for w in words_filtered:
    print(w + ":" + ps.stem(w))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Speech tagging - i.e. recognizing verbs etc
document = 'I am writing this example sentence to be used with speech tagging. The intention is to learn about python NLTK.'
sentences = sent_tokenize(document)

# uncomment to open the picture w/ definitions
#image = Image.open('nltk-speech-codes.png')
#image.show()

print("\nPre Filter:")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
for s in sentences:
    print(nltk.pos_tag(word_tokenize(s)))


# We can then filter based on the type of word
print("\nPost Filter:")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
data = []
for s in sentences:
    data = data + nltk.pos_tag(word_tokenize(s))
    
for word in data:
    if 'VB' in word[1]:
        print(word)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
# Training and Predicting - example dataset includes names.
# You can define your own list of tuples for datasets
names =  ([(name, 'male') for name in names.words('male.txt')] +
         [(name, 'female') for name in names.words('female.txt')])


# Feature extraction - example given is last letter of names
# Function to extract last letter
def gender_features(word):
    return {'last_letter': word[-1]}

# Feature set - i.e for each tuple in names, last letter and gender
feature_set = [(gender_features(n), g) for (n, g) in names]
train_set = feature_set

# Train
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Predict
example_name = input("Enter a name to predict gender: \n")
print(classifier.classify(gender_features(example_name)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sentiment Analysis - or rather could be used to classify the particular topic discussed
def word_feats(words):
    return dict([(word, True) for word in words])

# data
positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)' , 'Beautiful', 'amazing', 'sweet', 'rad', 'cool']
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(', 'shit', 'garbage','awful', 'crap', ':/']
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not']

# tuples based on data
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

# Total dataset to train on
train_set = negative_features + positive_features + neutral_features
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Vars to track number of neg / pos words
neg = 0
pos = 0
sentence = input("Type a sentence to analyze its sentiment:\n")
sentence = sentence.lower()
words = sentence.split(' ')

"""
I could see this working on a larger pre-defined dataset on whichever topic we choose
If there's a way to create a tree structure of sorts starting general and progressively getting
more specialized with its questions / responses ? 
Definitely needs bigger data sets - with current training_set it classifies 'pretty' as negative.

"""

for word in words:
    class_result = classifier.classify(word_feats(word))
    if class_result == 'neg':
        neg = neg +1
        print(word + " neg")
    if class_result == 'pos':
        pos = pos+ 1
        print(word + " pos")

print('Positive: ' + str(pos))
print('Negative: ' + str(neg))

