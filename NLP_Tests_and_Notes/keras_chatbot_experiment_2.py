# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 20:42:07 2020

    Keras Chat Bot - Prototype 2:
        
        Improvements from the previous prototype will involve
        changing the answer possibilities from just yes / no to include
        any number of appropriate answers. As well as implementing an internal
        adjustable train / test split versus the pre split data found and used in
        the previous prototype based on the Pierian Data tutorial.
        
        Will still be using the End-to-End Network described in the paper by (...)
        
        Instead of stories. I think it would be neat if we present the article headers.
        Make the user pick one and then they can ask some questions on that topic.
        
        
            
        
        Broken - After analysis of the data types and structures used a conversion
        script was written to convert the data to match that of prototype 1.
        
        Continued in prototype 3.
        
    
    

@author: tpvan
"""
"""
--------------------------------------------------------------------------------------------------
Basic Imports:
    
    Spacy: NLP library - Not used for alot beyond tokenization during the building
           of the vocabulary.
           Potentially include Lemmatization / Named Entity Recognition etc (?)
    
    Pickle : Used for serializing and deserializing python objects.
    
    Pandas : Library for reading in csv files or tsv files. -- Not used rn.
    
    Numpy : Python Math library, contains functions for processing large matrices
            As well as other fancy pants mathematical functions.
            
--------------------------------------------------------------------------------------------------
"""

import spacy
nlp = spacy.load('en_core_web_sm')
import pandas as pd
import pickle
import numpy as np

"""
--------------------------------------------------------------------------------------------------
Basic Functions:
    
    Read File: 
        Not really necessary - left in for future convenience.
    
    Seperate Punctuation:
        Added an alternative function that takes in a Spacy doc object.
        Potentially remove if affects results. (?)
        
        
--------------------------------------------------------------------------------------------------
"""

def read_file(filepath):
    with open(filepath) as f:
        text = f.read()
    
    return text

"""
The function below is equivalent to the following. This syntax is known as
list comprehension. Ok python. I'll give you this one, that's nice.

list=[]
for token in nlp(text):
    if text not in ...
        list.append(text)
    
    return list

This function and string at the end is provided by Keras for this very purpose.

"""

def seperate_punct_doc(doc):
    return [token.text.lower() for token in doc if token.text not in '\n\n \n\n\n--!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n']

def seperate_punct_text(text):
    return [text.lower() for word in text if text not in '\n\n \n\n\n--!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n']
            
"""
--------------------------------------------------------------------------------------------------
Step 1 : Read in data set and clean that data set

    Q & A Dataset [cite?] is a tsv file structured as follows:
        ArticleTitle, Question, Answer, DifficultyFromQuestioner, DifficultyFromAnswerer, ArticleFile
        We only really need Question and Answer, and possible Article Title.
        
       I've used excel to trim the extra colunms, indexed it and saved as a new tsv file.
       Much more smart like.
       
       Scratch that pandas automatically indexes it.
       Pandas.intelligence() > Thom.smrts()
       
--------------------------------------------------------------------------------------------------
"""
# Returns tuples representation of tsv file.
data_frame = pd.read_csv('q_a_test.tsv', sep='\t')

# here's the head, first 5 Q and A's
# print(data_frame.head())

"""
--------------------------------------------------------------------------------------------------
'Clean' data
--------------------------------------------------------------------------------------------------
"""
print('\n')
print('Before Cleaning NULL data ...')
# check for missing data - I saw a fair few NULLS in original file.
print(data_frame.isnull().sum())
print('\n')
# remove missing data
data_frame.dropna(inplace=True)
print('After Cleaning NULL data ...')
print(data_frame.isnull().sum())
print('\n')

# remove blank lines -- isspace method.
blanks=[]

for index, article, question, answer in data_frame.itertuples():
    if article.isspace():
        blanks.append(index)
    if question.isspace():
        blanks.append(index)
    if answer.isspace():
        blanks.append(index)
    
print('Number of Blank Lines before:')
print(len(data_frame))
# drop the blank lines
data_frame.drop(blanks, inplace=True)
print('Number of Blank Lines after:')
print(len(data_frame))
print('\n')

"""
--------------------------------------------------------------------------------------------------
Step 2: Create a Vocabulary for our model:
    
    We need to create a Vocabulary for the model to learn from, essentially
    just a giant unordered set of the words with no duplicates.
    
    We're also converting the data_frame return by pandas to a managable format.
    
    Then article, question and answer are converted to lower case and
    have their punctuation removed.
    This can be changed if we feel punctuation is needed, but it still has to 
    return a list of the words in order to build the vocabulary
    
--------------------------------------------------------------------------------------------------
"""
# Convert to 2d list.
df_list = [list(x) for x in data_frame.to_records(index=False)]

# Iterate over the df list and for answer and question return a lis of the words in each without punctuation.
# Possibly my favourite messy code I've ever written. Python man.
for i in range(len(df_list)):
    df_list[i][0] = [word.lower() for word in seperate_punct_doc(nlp(df_list[i][0]))]
    df_list[i][1] = [word.lower() for word in seperate_punct_doc(nlp(df_list[i][1]))]
    df_list[i][2] = [word.lower() for word in seperate_punct_doc(nlp(df_list[i][2]))]
         
# Create the vocab set
vocab = set()

# Perform a union on a set representation of articles, questions and answers.
for i in range(len(df_list)):
    vocab = vocab.union(set(df_list[i][0]))
    vocab = vocab.union(set(df_list[i][1]))
    vocab = vocab.union(set(df_list[i][2]))
    

# print(vocab)    
"""
--------------------------------------------------------------------------------------------------
Step 3: Splitting our Data into train and test data
    
    For training the model we have to have a set of data to compare our predictions to.
    We can experiment with how large our train / test split should be but we scan start with 30/70%.
    
    ROAD BLOCK -- the way the data was split before worked because the stories were composed of
    the same words as the questions and answers - in this case the articles are super limited in their words.
    I will nonetheless try to do a split as before in the text generation experiment. 
    
    If that yields no results I will manually split the date per article in about 70/30 split manually..
    
    Attempt 1 - messing around with arrays - abandoned
    Attemp 2 - Using the scikit train_test_split - must be a better way
    
    The possible error in this method is due to the specificity and relatively small size of each article
    some articles may not have made it into the test data. Which could be prevented by more selective splitting
    either with code or manually...
    
--------------------------------------------------------------------------------------------------
"""
from sklearn.model_selection import train_test_split

# There absolutely must be a better way of doing this.
# X is articles and questions
X=[]
# y is answers
y=[]

# append articles, questions and answers to X or y respectively -- X always represents 'features' and y is 'targets'
for i in range(len(df_list)):
    X.append([df_list[i][0], df_list[i][1]])
    y.append([df_list[i][2]])

# Use the sklean train_test_split method - 70/30 and shuffles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
# Sopy the appropriate answers from y_train to X_train to create 70% test data portion
for i in range(len(X_train)):
    X_train[i].append(y_train[i][0])

# Same thing for the 30% split
for i in range(len(X_test)):
    X_test[i].append(y_train[i][0])

train_data = X_train
test_data = X_test

df_train_data = pd.DataFrame(train_data)
df_test_data = pd.DataFrame(test_data)

#print(len(df_list))
#print(len(test_data))
#print(len(train_data))

"""
--------------------------------------------------------------------------------------------------
Step 4: Vectorizing the data.
    
    For the model to understand any of the text data we have to vectorize the articles, answers and questions.
    Using our unique set of words called vocab we give each word its own unique index.
    We then use these indexes to represent the sentences as vectors consisting of the corresponding values.
    
    Because we don't have identical size article, questions and answers we need to determine the
    largest of each and pad every other vector with 0's to match that size.
    
--------------------------------------------------------------------------------------------------
"""

# We need padding +1 - explain ... * (apparantly internally 0 gets appended throughout the process.... vague)
vocab_len = len(vocab) + 1

# longest article
all_article_lengths = [len(item[0]) for item in df_list]
# Max article length
max_article_len = max(all_article_lengths)
# Same for question and answer
max_question_len = max([len(item[1]) for item in df_list])
max_answer_len = max([len(item[2]) for item in df_list])

#print(max_article_len)
#print(max_question_len)
#print(max_answer_len)

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# No filters for tokenizer
tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

# all the indexes
#print(tokenizer.word_index)


def vectorize_data(data, word_index=tokenizer.word_index, max_article_len=max_article_len, max_question_len=max_question_len):
    
    # Article Names
    X=[]    
    # Questions
    Xq =[]
    # Correct Answers
    Y = []
    
    # creating vectors of all the indexes created above
    for index, article, question, answer in data.itertuples():
        # for each story
        x = [word_index[word.lower()] for word in article]
        xq = [word_index[word.lower()] for word in question]
        y = [word_index[word.lower()] for word in answer]
        
        X.append(x)
        Xq.append(xq)
        Y.append(y)
        
        
        # return a tuple that can be unpacked - and pad each of the sequences!
    return (pad_sequences(X, maxlen=max_article_len), pad_sequences(Xq, maxlen=max_question_len),  np.array(Y))

articles_train, questions_train, answers_train = vectorize_data(df_train_data)
articles_test, questions_test, answers_test = vectorize_data(df_test_data)


"""
-----------------------------------------------------------------
Step 5: Building the network - End-to-End Network:
    
    -Based on the End-to-End Network referenced in the paper by (cite?) 
   
    Step 5 a): Embedding Matrices - basically prepping the data :
        "Inputs {x1 ... xi} are converted into memory vectors by embedding
        ach xi into a continues space" in simplest case with embedding Matrix.
        vectorized and embedded in an Embedding Matrix. Which is a matrix
        that essentially maps all the vectors from a 'one-of-n-vectors' 
        space/size to a smaller more managable size and some other fancy maths.
        Same thing happens to output Vectors (C) an Questions (B). 
    
-----------------------------------------------------------------
"""
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM

# Two sperate inputs - articles and questions - need a placeholder for now 
# Input(shape(maxlenght, batch_size)) - basically a tuple with empty spot.
input_sequence = Input((max_article_len, ))
question = Input((max_question_len,))

# ------------------------- A or m ------------------------------------
# Input encoder M -- for all the inputs (M_i in paper) == {X1....Xi}
# Output dimension 64 - dim for embedding matrix.
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_len, output_dim=64))

# This randomly turns off 30% (up to 50% recommended) of neurons off in this layer - this is supposed 
#to help with overfitting -> when a model is too trained on specific data and does not handle new data well.
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
Step 5 b): Finding the probability vector over the inputs
    
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
Step 5 c): Generating the answer / prediction:
    
    Take a weighted sum of the input C and our match.
        i.e. top left of figure 1 a):
            o = sigma(p_i * ci) 
            
    Then permute the matrix to be of desired size this is for the next
    step - matrix addition is only possible of matrices of the same dimensions.
    
-----------------------------------------------------------------
"""
response = add([match, input_encoded_c])
# Permutes matrix to desired shape / dimensions. -- see next step
response = Permute((2, 1))(response)

"""
-----------------------------------------------------------------
Step 5 d): Generating the answer / predictions - continued :
    
    Another weighted addition of o from the last step and u.
    More specifically adding the o and B (questions encoded).
    Top right of figure 1 a).
    
    Add B and o (calculated in last step), then softmax the answer.
    
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
Step 6: Compile model and print Summary
-----------------------------------------------------------------
"""

model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

"""
-----------------------------------------------------------------
Step 7: Fit and Train network -- NOTE 200 epochs take a little while. Can cancel.
    
    Once trained I will include a the model in the repo.
    If you wan't to see the actual plot you'll have to actually train the model.
    It's pretty neat.
-----------------------------------------------------------------
"""



# changed batch size from 32 - 64

history = model.fit([articles_train, questions_train], answers_train, batch_size=64, epochs=200, validation_data=([articles_test, questions_test], answers_test))

"""
-----------------------------------------------------------------
Step 7: Plotting model accuracy
-----------------------------------------------------------------
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
-----------------------------------------------------------------
Step 8: Saving model -- Not currently saving -- overwrites prev training data
-----------------------------------------------------------------
"""

filename = 'chat_bot_experiment_200_v2.h5'
model.save(filename)

"""
-----------------------------------------------------------------
Step 9 (opt): Loading Course model -- actually less epochs than mine 100 v 200
-----------------------------------------------------------------
"""

#model.load_weights('chat_bot_experiment_200_v2.h5')


"""
-----------------------------------------------------------------
Step 10: Prediction - based on test data
-----------------------------------------------------------------
"""

pred_results = model.predict(([articles_test, questions_test]))

# prints all the probabilities for every word to be the answer. 
print(pred_results)

max_probability = np.argmax(pred_results[200])

# print an article, question answer and prediction
print(test_data[200][0])
print(test_data[200][1])
print(test_data[200][2])

for key, val in tokenizer.word_index.items():
    if val == max_probability:
        k = key

print(k)
