# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 20:18:02 2020

    Prototype 4 - New Data set!
        
        Next up: 
    
            incorproate the GUI.
            figure out intent filter.
            Potentially include Lemmatization / Named Entity Recognition etc (?)
            
            
            Include some of the previous data set for variety?
            There is a question intent ! 
            If we can get the intent thing to work and it recognizes those
            prompts like "Can I ask a question?" etc
            It would be awesome to draw upon that second data set to answer questions!



            








    
@author: tpvan
"""

"""
--------------------------------------------------------------------------------------------------
Basic Imports:
    
    Spacy: NLP library - Not used for alot beyond tokenization during the building
           of the vocabulary.
           Potentially include Lemmatization / Named Entity Recognition etc (?)
    
    Pickle : Used for serializing and deserializing python objects.
    
    Pandas : Library for reading in csv files or tsv files.
    
    Numpy : Python Math library, contains functions for processing large matrices
            As well as other fancy pants mathematical functions.
            
--------------------------------------------------------------------------------------------------
"""

import pickle
import numpy as np
import spacy
nlp = spacy.load('en_core_web_sm')

"""
------------------------------------------------------------------------------------------------
Step 1: Load in all data, test data and train data:
    
    Train and Test data are both serialized objects created by the conversion script we wrote, 
    included in the folder. 
    The format is [(int:idx, [intent], [question], [answer])]
    
------------------------------------------------------------------------------------------------
"""

# full data frame
with open('qa_df.txt', 'rb') as f:
    all_data = pickle.load(f)

# train data if split performed in conversion script
with open('qa_train_df_v2.txt', 'rb') as f:
    train_data = pickle.load(f)

# test data if split performed in conversion script
with open('qa_test_df_v2.txt', 'rb') as f:
    test_data = pickle.load(f)    
    
# list of all indexed the answers used to print out answer at run time.
with open('qa_indexed_ans_v2.txt', 'rb') as f:
    idx_ans_list = pickle.load(f)
    

# For the current dialogu data set there is no real point in using a test / train split
#print("Testing to Training Data ratio : " + str(len(test_data) / len(train_data)))

train_data = all_data

"""
------------------------------------------------------------------------------------------------
Step 2: Create a vocabulary for our model to learn on
    
    Train and Test data are are both cleaned up and formatted as desired in the conversion script
    so essentially all that has to be done is create a giant unordered set. Meaning no duplicates.
    
------------------------------------------------------------------------------------------------
"""

# Changed the vocab to only include the train data
#all_data = test_data + train_data

vocab = set()

if input("Create new vocab ? (Y / N)").lower() in "yes":
    
    for index, article, question, answer in all_data:
        vocab = vocab.union(set(str(index)))
        vocab = vocab.union(set(article))
        vocab = vocab.union(set(question))
        vocab = vocab.union(set(answer))
    # Added padding -- It seems internally a 0 get's appended a throughout the process.
   
    with open('dialogue_vocab_v1.txt', 'wb') as fp:
            pickle.dump(vocab, fp, protocol=4)        
else :
    print("Loading Vocab ...")
    with open('dialogue_vocab_v1.txt', 'rb') as fp:
            vocab = pickle.load(fp)
    

vocab_len = len(vocab) + 1


"""




I think I figured out why it appears the model is not loading properly.
The vocab and indexes are recreated if you exit out of spyder and re run this script.
Causing the answers to be wrong when looked up during the printing.

If that is the issue which I will confirm later then I will create a new script that will serve as the 
actual launching point for the conversation with gui and all that. This can be the training script for 
the main data set. The new script should simply load the model, vocab and other required parts.





"""


#print(vocab)

"""
------------------------------------------------------------------------------------------------
Step 3: Vectorizing the data
    
    For the model to understand any of the text data we have to vectorize the articles, answers and questions.
    Using our unique set of words called vocab we give each word its own unique index.
    We then use these indexes to represent the sentences as vectors whocse components consist
    of the corresponding index values.
    
    Because we don't have identical size article, questions and answers we need to determine the
    largest of each and pad every other vector with 0's to match that size.
    
------------------------------------------------------------------------------------------------
"""

# longest intent Name
all_intent_lengths = [len(data[0]) for data in all_data]

# max length 
max_intent_len = max(all_intent_lengths)

# max question length
max_question_len = max([len(data[1]) for data in all_data])
# max answer length
max_answer_len = max([len(data[2]) for data in all_data])

# imports for Tokenizer and padding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# No filters
tokenizer = Tokenizer(filters=[])
# This creates all the individual indexes for each word in the vocab.
tokenizer.fit_on_texts(vocab)

# all the indexes -- if you're curious.
#print(tokenizer.word_index)

# The following method actually performs the vectorization.
def vectorize_data(data, word_index=tokenizer.word_index, max_intent_len=max_intent_len, max_question_len=max_question_len):
    
    # Articles
    X=[]    
    # Questions
    Xq =[]
    # Correct Answers
    Y = []
    
    # Creating vectors of all the indexes created above
    for index, intent, question, answer in data:
        
        # For each intent, Question and Answer find their corresponding index, 
        # use list comprehension to create a vector of each index.
        x = [word_index[word.lower()] for word in intent]
        xq = [word_index[word.lower()] for word in question]
        
        # create a vector of all 0's and 1 one at answers index
        y = np.zeros(len(word_index)+1)
        y[word_index[str(answer[0])]] = 1
       
        # Then add that vector to the corresponding main list.
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    
        # return a tuple that can be unpacked - and pad each of the sequences!
    return (pad_sequences(X, maxlen=max_intent_len), pad_sequences(Xq, maxlen=max_question_len), np.array(Y))

# Actual method call and unpacking of the tuples to set each respective variable below.
articles_train, questions_train, answers_train = vectorize_data(train_data)

# As mentioned test data is currently not used.
#articles_test, questions_test, answers_test = vectorize_stories(test_data)

"""
-----------------------------------------------------------------
Step 4 - Building the network:
    
    To understand how the network is built you should read the 
    End to End networks paper I inlcuded. The only relevant part 
    is the 'Approach' section that describes the process. I'll do
    my best annotation it here as well.
    
    In particular Figure 1 a).
   
    Step 1 a): 
    
        Embedding Matrices - basically prepping the data :
        "Inputs {x_1 ... x_i} are converted into memory vectors by embedding
        each x_i into a continues space in simplest case with embedding Matrix."
        and Embedding matrix is a matrix that essentially maps all the vectors from 
        a 'one-of-n-vectors' space/size to a smaller more managable size and some 
        other fancy maths. You can imagine if one of the vectors we create is 7 words long, 
        meaning 7 indexes and another is 234 words long, the 7 word one will have a lot of 0's
        in it. The Embedding takes all these vectors in matrix representation and transforms
        them to a smaller space. That is my understanding at least.
        Same thing happens to output Vectors (C) an Questions (B). 
    
-----------------------------------------------------------------
"""

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM


"""

    All dropouts currently not commented out.
    Will try with out drop out next.

"""



# Two sperate inputs - intentss and questions - need a placeholder for now 
# Input(shape(maxlenght, batch_size)) - tuple w empty spot
input_sequence = Input((max_intent_len, ))
question = Input((max_question_len,))

# ------------------------- A or m ------------------------------------
# Input encoder M -- for all the inputs (M_i in paper) == {X1....Xi}
# output dimension 64 - dim for matrix.
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_len, output_dim=64))

# This randomly turns off 30% ( to 50%) of neurons off in this layer - this is supposed to help with overfitting.
# overfitting is when a model is too trained on specific data and does not handle new data well.
input_encoder_m.add(Dropout(0.3))
# output -> (samples, intent_max_len, embedding_dim)

# ------------------------- C ----------------------------------------
# Input encoder C - Each input vector x has a corresponding output vector c 
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_len, output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))
# output -> (samples, intent_max_len, max_question_len)

# ------------------ Question Encoder (B) ----------------------------
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_len, output_dim=64, input_length=max_question_len))
question_encoder.add(Dropout(0.3))
# output -> (samples, intent_max_len, embedding_dim)

# ---------- pass inputs into the appropriate encoders ----------------
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)


"""
-----------------------------------------------------------------
Step 4 b): Finding the probability vector over the inputs
    
    The embedded matrices A & B are referred to as internal state 'u'
    In that 'memory space' / internal space we compute the match between
    u and EACH memory vector. Not as complicated as it sounds.
    
    This is done by taking the inner product (or dot product) or simply
    transpose of a matrix multiplied by the other matrix.
    
    Paper says the following 
    probability = p_i = Softmax(uT * m_i)
    Code here just takes the dot product of the encoded inputs A and the
    encoded questions. Lower left of Figure 1 a).
    
    The softmax function takes in a vector and normalizes all
    the vectors components, such that each of the vector components
    are on the interval of (0, 1) and sum to 1. 
-----------------------------------------------------------------
"""

match = dot([input_encoded_m, question_encoded], axes=(2, 2))

match = Activation('softmax')(match)

"""
-----------------------------------------------------------------
Step 4 c): Generating the answer / prediction:
    
    Take a weighted sum of the input C and our match.
        i.e. top left of figure 1 a):
            o = sigma(p_i * ci) 
    Then permute the matrix to be of desired size.
    Matrix addition is only possible of matrices of the same dimensions.
-----------------------------------------------------------------
"""

response = add([match, input_encoded_c])
# Permutes matrix to desired shape / dimensions. -- see next step
response = Permute((2, 1))(response)

"""
-----------------------------------------------------------------
Step 4 d): Generating the answer / predictions -- cont'd :
    
    Another weighted addition of o from the last step and u.
    More specifically adding the o and B (questions encoded).
    Top right of figure 1 a).
    
    Add B and o (calculated in last step), then softmax the answer.
    
            a^ = softmax(W(o + u))
-----------------------------------------------------------------
"""

answer = concatenate([response, question_encoded])

# Weighting
answer = LSTM(64)(answer)
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
Step 6: Fit and Train network     / Step 7 : Plot Accuracy

    Currently set to validate on training data - 
    Might try to do a 70/30 next out of curiosity -- due to the nature of this data set it might still work.
    
-----------------------------------------------------------------
"""
import matplotlib.pyplot as plt

if input("Train model? (Y/N)").lower() in "yes":
    history = model.fit([articles_train, questions_train], answers_train, batch_size=128, epochs=250, validation_data=([articles_train, questions_train], answers_train))
    
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'Test'], loc='lower left')
    plt.show()

    model.save('chat_bot_experiment_500_128_dialogue_dropout_validated_on_train_v1.h5')

"""
-----------------------------------------------------------------
Step 8 (opt): Saving model -- Not currently saving -- Overwrites prev training data.
-----------------------------------------------------------------
"""

    

"""
-----------------------------------------------------------------
Step 9 (opt): Loading Model
-----------------------------------------------------------------
"""
from tensorflow.keras.models import load_model

if input("Load model? (Y/N)").lower() in "yes":
    print("Loading Model ...")
    model = load_model('chat_bot_experiment_500_128_dialogue_dropout_validated_on_train_v1.h5')

"""
-----------------------------------------------------------------
Step 10: Prediction - Based on test data - Not required for this data set.

    This current model is trained on all of data set.
    Considering the nature of the data set maybe I will try to 
    do a 70/30 split and see if it still does its thing.
    
-----------------------------------------------------------------
"""

"""
pred_results = model.predict(([articles_test, questions_test]))

# Following prints all the probabilities for each article / question combo
#print(pred_results)

# For example for the first of the results at index 0 - find the highest probability answer
max_probability_answer = np.argmax(pred_results[12])

# print the actual article, question and answer 
print(test_data[12][0])
print(test_data[12][1])
print(test_data[12][2])

# Then using the max_probability find the answer using the indexes from tokenizer.
for key, val in tokenizer.word_index.items():
    if val == max_probability_answer:
        k = key

# There's an off by one error we should fix due to indexing for answers starting at 1 ... 
print(k)

"""

"""
-----------------------------------------------------------------
User Input :
-----------------------------------------------------------------
"""
def seperate_punct_doc(doc):
    return [token.text.lower() for token in doc if token.text not in '\n\n \n\n\n--!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n']

counter = 5
while(counter > 0):
    # possibly change this to remove punctuation and lower case it after printing the next message... 
    my_article_text = input('Pick the intent : \n')
    my_article = seperate_punct_doc(nlp(my_article_text))
    my_article = [word for word in my_article if word in vocab]
    my_question_text = input('Say Something related to ' + my_article_text + ' : \n')
    my_question = seperate_punct_doc(nlp(my_question_text))
    my_question = [word for word in my_question if word in vocab]
    
    my_data = [(['0'], my_article, my_question, ['work'])]
    my_article, my_question, my_answer = vectorize_data(my_data)
    
    pred_results = model.predict(([my_article, my_question]))
    val_max = np.argmax(pred_results[0])
    
    for key, val in tokenizer.word_index.items():
        if val == val_max:
            k = key


    """
    Currently prints the k-1 and k -- just debuggin - harder to tell 
    if there is still a off by one.
    """

    
    print(k)
    if(str(k).isdigit()):
        print(' '.join(idx_ans_list[int(k)-1][1]))
        print(' '.join(idx_ans_list[int(k)][1]))
        print("Number of Questions left: " + str(counter))
        counter-=1


