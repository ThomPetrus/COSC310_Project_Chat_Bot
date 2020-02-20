# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:43:17 2020

Text Generation with Python and Keras.

Idea is to train a model, pass it 25 words, then predict the next word.

@author: tpvan
"""
"""
------------------------------------------------------------------------
Imports / initializations
------------------------------------------------------------------------
"""

import spacy

# Disabling some stuff to make it quicker...
nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])

# Setting a large max length - apparantly nlp complains sometimes. 
nlp.max_length = 1198623

"""
------------------------------------------------------------------------
Basic Functions
------------------------------------------------------------------------
"""

# Function to read file - self-explanatory
def read_file(filepath):
    with open(filepath) as f:
        text = f.read()
    
    return text

"""
The function below is equivalent to the following. This syntax is known as
list comprehension. Pretty neat!

list=[]
for token in nlp(doc_text):
    if token.text not in ...
        list.append(token.text)
    
    return list

This function and string at the end is provided by Keras for this very purpose.
"""

def seperate_punct(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n--!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n']

"""
As mentioned in header the idea here was was to read 25 words and predict the
next one. This next function just splits the tokens from above into sequences of 25
words, each sequence progressing one word at a time. 
syntax :
    for loop(startidx, endidx):
        current seq = tokens[current_beginidx : current_endidx]
        etc
"""

def seperate_sequences(tokens, train_len):
    text_sequences =[]
    for i in range(train_len, len(tokens)):
        current_seq = tokens[i-train_len:i]
        text_sequences.append(current_seq)
    
    return text_sequences

"""
------------------------------------------------------------------------
Main functionality:
    read file
    separate into tokens/words w/out punctuation
    seperate into sequences of desired size
    use keras to tokenize each token
    use keras to generate unique ids for each word
    -- to be continued tmr ...
Excuse the extraneous commented out code. It could be useful for reference later.
Just remove the block quotes to make it work obvs.
------------------------------------------------------------------------
"""
# read file and seperate into list w/out punctuation.
text = read_file('moby_dick_four_chapters.txt')
words = seperate_punct(text)

#print(tokens)

# seperate into desired sequence size:
train_len = 25+1
sequence_list = seperate_sequences(words, train_len)

"""
# if you want to display first sequence --> i.e. words 0-25
# note that [1] == 1-26
print(' '.join(sequence_list[0]))
"""

# import a tokenizer from keras - then fit to sequences
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequence_list)

# This generates and ID for each word in all the sequences
# and also orders them by how often theyre used -- i.e. 'the' is numero uno
sequences = tokenizer.texts_to_sequences(sequence_list)

"""
# Tokenizer has a ton of funcionality just check the method calls.
# i.e. print first sequence in order w/ associated index bc why not.
print('\n')
for i in sequences[0]:
    print(f"{i} : {tokenizer.index_word[i]}")
    
# print all the word counts :
print(tokenizer.word_counts)
"""

# get size of vocab
vocabulary_size = len(tokenizer.word_counts)

"""
Note that the type(sequences) is a list, with a list at each index.
The list at each index the ID of the words in that list. 
Simply put it is a matrix, and can be formatted as such using numpy.
"""
import numpy as np
sequences = np.array(sequences)
print("The matrix: rows are rows are sequences, columns are indexes of words in sequence")
print(sequences)
print("Once again notice each sequence moves ahead 1 word.")


