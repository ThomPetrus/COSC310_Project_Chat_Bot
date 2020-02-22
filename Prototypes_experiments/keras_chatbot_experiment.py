# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 19:22:39 2020

Chat bot Experiment - End to End Network

    This chat bot will answer questions based on a set amount of information
    for example a story.


@author: tpvan
"""
"""
-----------------------------------------------------------------
imports:
    pickel is used for serializing and deserializing binary data.
-----------------------------------------------------------------
"""

import pickle
import numpy as np

with open('train_qa.txt', 'rb') as f:
    train_data = pickle.load(f)
with open('test_qa.txt', 'rb') as f:
    test_data = pickle.load(f)

# Notice we have a 1:10 ratio of train to test data
print(len(train_data))
print(len(test_data))

# Example of one segment of data set
#print(train_data[0])
print(' '.join(train_data[0][0]))
print(' '.join(train_data[0][1]))
print(train_data[0][2])


"""
-----------------------------------------------------------------
Creating Vocabulary:
    Vocab is a set - in python a set is a unordered collection of 
    unique elements.
    Python set union creates new set of all unique words
    
    In this example the answers are limited to yes and no ...
    I hope that in using larger datasets with answers associated
    with particular questions this method will translate to that. 
-----------------------------------------------------------------
"""

all_data = test_data + train_data

vocab = set()

for story, question, answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))
    
vocab.add('no')
vocab.add('yes')

# we need padding - same as before - internally a 0 gets appended
vocab_len = len(vocab) + 1

# longest story
all_story_lengths = [len(data[0]) for data in all_data]

# max length 
max_story_len = max(all_story_lengths)

# max question length
max_question_len = max([len(data[1]) for data in all_data])

print(max_question_len)
print(max_story_len)

"""
-----------------------------------------------------------------
Vectorizing the data
-----------------------------------------------------------------
"""

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# No filters
tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

# all the indexes
print(tokenizer.word_index)

train_story_text = []
train_question_text = []
train_answers = []

# seperate the stories, question and answers
for story, question, answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)
    train_answers.append(answer)
    
# converts each word to its matching index
train_story_seq = tokenizer.texts_to_sequences(train_story_text)
#print(train_story_seq)

# Stories are not the same length unlike in text generator. Still need to pad!
def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_len, max_question_len=max_question_len) :
    
    # Stories
    X=[]    
    # questions
    Xq =[]
    # correct answers
    Y = []
    
    # creating vectors of all the indexes created above
    for story, question, answer in data:
        # for each story
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in question]
        
        # this I think would have to change to match the above if we have more complicated answers vs yes / no
        y = np.zeros(len(word_index)+1)
        y[word_index[answer]] = 1
        
        X.append(x)
        Xq.append(xq)
        Y.append(y)
        
        # return a tuple that can be unpacked - and pad each of the sequences!
    return (pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_question_len), np.array(Y))

stories_train, questions_train, answers_train = vectorize_stories(train_data)
stories_test, questions_test, answers_test = vectorize_stories(test_data)

#print(tokenizer.word_index['yes'])
#print(tokenizer.word_index['no'])
#print(sum(answers_test))

"""
-----------------------------------------------------------------
Building the network:
    - To understand how the network is built you should read the 
    End to End networks paper I inlcuded. The only relevant part 
    is the 'Approach' section that describes the process. I'll do
    my best annotation it here as well.
   
Step 1: Embedding Matrices - basically prepping the data :
    "Inputs {x1 ... xi} are converted into memory vectors by embedding
    ach xi into a continues space" in simplest case with embedding Matrix.
    vectorized and embedded in an Embedding Matrix. Which is a matrix
    that essentially maps all the vectors from a 'one-of-n-vectors' 
    space/size to a smaller more managable size and some other fancy maths.
    Same thing happens to output Vectors (C) an Questions (B). 
    
    I believe this embedding process is known as word2vec algorithm.
    At least I keep seeing that term pop up.
-----------------------------------------------------------------
"""
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM

# Two sperate inputs - stories and questions - need a placeholder for now 
# Input(shape(maxlenght, batch_size)) - tuple w empty spot
input_sequence = Input((max_story_len, ))
question = Input((max_question_len,))

# Vocab + that padding
vocab_len = len(vocab) + 1

# ------------------------- A or m ------------------------------------
# Input encoder M -- for all the inputs (M_i in paper) == {X1....Xi}
# output dimension 64 - unexplained - dim for matrix I suppose.
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_len, output_dim=64))

# This randomly turns off 30% ( to 50%) of neurons off in this layer - this is supposed to help with overfitting.
input_encoder_m.add(Dropout(0.3))
# output -> (samples, story_max_len, embedding_dim)

# ------------------------- C ----------------------------------------
# Input encoder C - Each input vector x has a corresponding output vector c 
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_len, output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))
# output -> (samples, story_max_len, max_question_len)

# ------------------ Question Encoder (B) ----------------------------
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_len, output_dim=64, input_length=max_question_len))
question_encoder.add(Dropout(0.3))
# output -> (samples, question_max_len, embedding_dim)

# ---------- pass inputs into the appropriate encoders ----------------
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

"""
-----------------------------------------------------------------
Step 2: Finding the probability vector over the inputs
    
    The embedded matrices A & B are referred to as internal state 'u'
    In that 'memory space' / internal space we compute the match between
    u and EACH memory vector. Not as complicated as it sounds.
    
    This is done by taking the inner product (or dot product) or simply
    transpose of a matrix multiplied by the other matrix.
    
    Paper says the following 
    probability = p_i = Softmax(uT * m_i)
    Code here just takes the dot product of the encoded inputs A and the
    encoded questions. Lower left of Figure 1.
    
    The softmax function takes in a vector and normalizes all
    the vectors components, such that each of the vector components
    are on the interval of (0, 1) and sum to 1. 
-----------------------------------------------------------------
"""
match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)

"""
-----------------------------------------------------------------
Step 3: Generating the answer / prediction:
    
    Take a weighted sum of the input C and our match.
        i.e. top left of figure 1 a):
            o = sigma(p_i * ci) 
    Then permute the matrix to be of desired size...
-----------------------------------------------------------------
"""
response = add([match, input_encoded_c])
# Permutes matrix to desired shape / dimensions. -- see next step
response = Permute((2, 1))(response)
"""
-----------------------------------------------------------------
Step 4: Generating the answer / predictions -- cont'd :
    
    add B and o (calculated in last step), then softmax the answer.
    The calculating mentioend a final weight matrix W.
    Only logical part of the code that represents this is dropout part.
    
        i.e. top right of figure 1 a):
            a^ = softmax(W(o + u))
-----------------------------------------------------------------
"""

answer = concatenate([response, question_encoded])

# Weighting -- W?
answer = LSTM(32)(answer)
answer = Dropout(0.5)(answer)
answer = Dense(vocab_len)(answer)
answer = Activation('softmax')(answer)

"""
-----------------------------------------------------------------
Step 5: Compile model and print Summary
-----------------------------------------------------------------
"""

model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

"""
-----------------------------------------------------------------
Step 6: Fit and Train network -- NOTE 200 epochs take a little while. 20 mins. can cancel.
    
    Currently set to not trian the model. 
    I've included and load a 200 epoch model in step 9.
    It works really well on test data.
    If you wan't to see the actual plot you'll have to actually train the model.
    It's pretty neat.
-----------------------------------------------------------------
"""

#history = model.fit([stories_train, questions_train], answers_train, batch_size=32, epochs=200, validation_data=([stories_test, questions_test], answers_test))

"""
-----------------------------------------------------------------
Step 7: Plotting model accuracy
-----------------------------------------------------------------
"""

"""
import matplotlib.pyplot as plt
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'Test'], loc='lower left')
plt.show()
"""

"""
-----------------------------------------------------------------
Step 8: Saving model -- Not currently saving -- overwrites prev training data
-----------------------------------------------------------------
"""

filename = 'chat_bot_experiment_v1.h5'
#model.save(filename)

"""
-----------------------------------------------------------------
Step 9 (opt): Loading Course model -- actually less epochs than mine 100 v 200
-----------------------------------------------------------------
"""
# course
#model.load_weights('chatbot_10.h5')

model.load_weights('chat_bot_experiment_v1.h5')

"""
-----------------------------------------------------------------
Step 10: Prediction - based on test data
-----------------------------------------------------------------
"""

pred_results = model.predict(([stories_test, questions_test]))

# prints all the probabilities for every word to be the answer. 
print(pred_results)

max_probability = np.argmax(pred_results[200])

# print first story, question answer ad prediction
print(test_data[200][0])
print(test_data[200][1])
print(test_data[200][2])

for key, val in tokenizer.word_index.items():
    if val == max_probability:
        k = key

print(k)


"""
-----------------------------------------------------------------
Step 11: Prediction - based on User input:
    
    The model ONLY understands the set vocabulary that the network
    is trained on.
    
    Currently need to put spaces around the punctuation.
    This could of course be changed by using some preprocessing techniques.
    
    
-----------------------------------------------------------------
"""

my_story = "John left the kitchen . Sandra dropped the football in the garden ."

my_question = "Is the football in the garden ?"

# This is the process for adding new data to train / test dataset.
mydata = [(my_story.split(), my_question.split(), 'yes')]
my_story, my_question, my_answer = vectorize_stories(mydata)


pred_results = model.predict(([my_story, my_question]))
val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == max_probability:
        k = key

print(k)


"""
OK.
    The way this would work if we wanted to pick our own topic is 
    to create new test and training data as well as process the answers in
    the same way as the questions and inputs. Rather than just yes/no.
    
    Which I suppose I'll try next. If we wanted to use a different data set 
    that is a must.
    
"""
