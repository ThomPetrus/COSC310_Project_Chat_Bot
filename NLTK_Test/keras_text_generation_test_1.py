# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:43:17 2020

Text Generation with Python and Keras.


    Idea is to train a model, pass it 25 words, then predict the next word.

    Considering this is for educational purposes and might not play
    a direct part in the chat bot both the number of neurons and the number of epochs are really low 
    to conserve time in favor of results. 
    
    reasonably number would be min of 150 nodes and 200 epochs.
 

@author: tpvan
All code based on lectures by Pierian Data, All notes and annotations by Thom.

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
    Set-up:
        read file
        separate into tokens/words w/out punctuation
        seperate into sequences of desired size
        use keras to tokenize each token
        use keras to generate unique ids for each word
    Building the LSTM model
    Training the model
    Saving the model
    Generating Text
        
Excuse the extraneous commented out code. It could be useful for reference later.
Just remove the block quotes to make it work obvs.
------------------------------------------------------------------------
"""

"""
------------------------------------------------------------------------
Set-up
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

# This generates an ID for each word in all the sequences
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

"""
The reason the matrix is needed is because this way we can access each column of the matrix easily
and do our test / train split and have the last column as our target X, and the rest as our features y.

The :,:-1 is slice notation to select everything except that last column.
Similarly :,-1 is only the last one.
Not completely clear to me why, but it works.
"""

from keras.utils import to_categorical
X = sequences[:,:-1]
y = sequences[:,-1]

# keras padding requires the plus 1 ...
y = to_categorical(y, num_classes=vocabulary_size+1)

# This could be hardcoded as 25, but this way it does not require changing if we adjust sequence length
# You can run the line below to see that X consists of 111312 rows and 25 columns
#print(X.shape)
seq_length = X.shape[1]

"""
------------------------------------------------------------------------
Building the LSTM model
    
    Change neuron count here~

    Note that for the purposes of completing this course I have kept the
    number of neurons really low. (2x)
    More neurons, better results ususally, longer time to train...

Embedding: 
    Turns positive integers (indexes) into dense vectors of fixed size
    Can only be used as the first layer in a model
    Embedding(Input dimension, output dimention, inputsize)
LSTM:
    Num of nodes/neurons should be a multiple of your input size - i.e 25.
    LSTM(num of nodes)
------------------------------------------------------------------------
"""

from tensorflow.python.keras import Sequential 
from tensorflow.python.keras.layers import Dense, LSTM, Embedding

def create_model(vocabulary_size, seq_length):
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_length, input_length=seq_length))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50, activation='relu'))
    
    model.add(Dense(vocabulary_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    
    return model

model = create_model(vocabulary_size+1, seq_length)

"""
------------------------------------------------------------------------
Training the Model

    Change Epoch Count Here~

    The batch size is how many sequences passed in at one time, too many will overload the network.
    only 2 epochs to conserve time - likely won't produce results - will try later on desktop with higerh neuron count / epochs.
------------------------------------------------------------------------
"""
"""
------------------------------------------------------------------------
Course model - trained for 300 epochs. ~60% accuracy.
------------------------------------------------------------------------
"""
from tensorflow.python.keras.models import load_model
from pickle import dump, load

model = load_model('course_decent_mobydick_model.h5')
tokenizer = load(open('course_decent_tokenizer', 'rb'))
"""
------------------------------------------------------------------------
Training a new model with parameters present in program:
------------------------------------------------------------------------
"""
#fit model
#model.fit(X, y, batch_size=128, epochs=2, verbose=1)

"""
------------------------------------------------------------------------
Saving the model

    Pickle is used to write and load output.
------------------------------------------------------------------------
"""

from pickle import dump, load
# save model
model.save('my_shitty_moby_dick_model.h5')
# save tokenizer
dump(tokenizer, open('my_simpletokenizer', 'wb'))

"""
------------------------------------------------------------------------
Generating Text using the model.
    I have included extra print statements to show the process if you're interested.
    
    The padding/truncating is not absolutely necessary here, but makes the function more
    robust. If we enter a seed text that is too short or too long it will ensure the max
    length is the desired length. I.e. the sequence length on which the model is trained.
    
------------------------------------------------------------------------
"""
from keras.preprocessing.sequence import pad_sequences

def generate_text(model, tokenizer, seq_length, seed_text, num_gen_words):
   
    output_text=[]
    
    input_text = seed_text
    for i in range(num_gen_words):
        
        #print("Input text : " + input_text)
        
        # remember texts_to_sequences generates the unique ID for the text as a sequence
        # returns a list, that in this case, only contains one sequence at [0]
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        
        #print("Encoded text : ")
        #print(encoded_text)
        
        # Pad / truncate if needed
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_length, truncating='pre')
        
        #print("Padded Encoded text : ")
        #print(pad_encoded)
        
        # model.predict_classes essentially assigns probabilities to all the words, where word at [0] has highest prob
        # remember each word has a unique index in vocab.
        predicted_word_index = model.predict_classes(pad_encoded, verbose=0)[0]
        
        # then use the index retrieved and the tokenizer to retrieve the actual word
        predicted_word = tokenizer.index_word[predicted_word_index]
        
        input_text+= ' ' + predicted_word
        
        output_text.append(predicted_word)
        
    return ' '.join(output_text)


print("\n\nACTUAL TEXT GENERATION\n\n")

first_sequence = sequence_list[0]
#print("First Sequence : "+ ' '.join(first_sequence))

# OR Random seed if you want - change seed number for diff sequence.
import random
random.seed(255)
random_pick = random.randint(0, len(sequence_list))
random_text = sequence_list[random_pick]

#seed_text = ' '.join(first_sequence)
seed_text = ' '.join(random_text)
print("Input Text:\n")
print(seed_text)

"""
------------------------------------------------------------------------
Output
------------------------------------------------------------------------
"""

predicted_text = generate_text(model, tokenizer, seq_length, seed_text=seed_text, num_gen_words=25)
print("\nOutput Text:\n")
print(predicted_text)