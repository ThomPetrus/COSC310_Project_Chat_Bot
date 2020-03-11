# -*- coding: utf-8 -*-
"""
Updated on Mon Mar  7 1:45:46 2020

        Added more universal global path directories
        Added randomized answers when all the vocab in the input is unknown
        Added key bind for enter key to send the input

"""

"""
--------------------------------------------------------------------------------------------------
Basic Imports:
    
    Keras: Library used to build the ANN/RNN. Requires the keras libary to be installed.
    
    Spacy: Requires the spacy library to be installed.
           NLP library - Not used for alot beyond tokenization during the building
           of the vocabulary.
    
    Pickle : Used for serializing and deserializing python objects.
    
    Pandas : Library for reading in csv files or tsv files.
    
    Numpy : Python Math library, contains functions for processing large matrices
            As well as other fancy pants mathematical functions.
    
    Tkinter : GUI
            
--------------------------------------------------------------------------------------------------
"""
from tkinter import *
from tkinter import scrolledtext, INSERT, Button, Label, Entry, END, Tk
import pickle
import numpy as np
import spacy
import os
import string
import random
nlp = spacy.load('en_core_web_sm')


# Get previous directory
this_dir = os.path.dirname(os.path.dirname(__file__))
print(this_dir)
split_dir = this_dir.split('\\')
prev_dir = ""
for i in range(len(split_dir)):
    prev_dir = prev_dir + split_dir[i] + '/'
    if split_dir[i] == 'ChatBot_v2':
        break

# Global Variables for convenience
model_load_name = prev_dir + 'data/chat_bot_experiment_5000_128_dialogue_dropout_validated_on_train_v2.h5'
model_save_name = prev_dir + 'data/chat_bot_experiment_5000_128_dialogue_dropout_validated_on_train_v2.h5'
intents_model_load_name = prev_dir + 'data/chat_bot_experiment_5000_128_dialogue_dropout_validated_on_train_INTENTS.h5'
intents_model_save_name = prev_dir + 'data/chat_bot_experiment_5000_128_dialogue_dropout_validated_on_train_INTENTS.h5'


data_frame_load_name = prev_dir + 'data/qa_df.txt'
train_data_frame_load_name = prev_dir + 'data/qa_train_df_v2.txt'
test_data_frame_load_name = prev_dir + 'data/qa_test_df_v2.txt'

indexed_ans_list_load_name = prev_dir + 'data/qa_indexed_ans_v2.txt'

vocab_save_name = prev_dir + 'data/dialogue_vocab_v2.txt'
vocab_load_name = prev_dir + 'data/dialogue_vocab_v2.txt'
tokenizer_save_name = prev_dir + 'data/dialogue_tokenizer_v2.txt'
tokenizer_load_name = prev_dir + 'data/dialogue_tokenizer_v2.txt'

"""
------------------------------------------------------------------------------------------------
Step 1: Load in all data, test data and train data:
    
    Train and Test data are both serialized objects created by the conversion script we wrote, 
    included in the folder. 
    The format is [([int:idx], [intent], [question], [answer])]
    
------------------------------------------------------------------------------------------------
"""

def load_data_frames():
    # full data frame
    with open(data_frame_load_name, 'rb') as f:
        all_data = pickle.load(f)
    
    # train data if split performed in conversion script
    with open(train_data_frame_load_name, 'rb') as f:
        train_data = pickle.load(f)
    
    # test data if split performed in conversion script
    with open(test_data_frame_load_name, 'rb') as f:
        test_data = pickle.load(f)    
           
    # For the current dialogu data set there is no real point in using a test / train split
    # print("Testing to Training Data ratio : " + str(len(test_data) / len(train_data)))
    
    return all_data, train_data, test_data;


def load_idx_ans_list():
    # list of all indexed the answers used to print out answer at run time.
    with open(indexed_ans_list_load_name, 'rb') as f:
        idx_ans_list = pickle.load(f)
        
        return idx_ans_list
    
# Seperates punctuation, to lowercase and tokenizes input.
def seperate_punct_doc(doc):
    return [token.text.lower() for token in doc if token.text not in '\n\n \n\n\n--!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n']

 
"""
------------------------------------------------------------------------------------------------
Step 2: Create a vocabulary for our model to learn on
    
    Train and Test data are are both cleaned up and formatted as desired in the conversion script
    so essentially all that has to be done is create a giant unordered set. Meaning no duplicates.
    
------------------------------------------------------------------------------------------------
"""

# imports for Tokenizer and padding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

def create_new_vocab(all_data):
    # Changed the vocab to only include the train data
    #all_data = test_data + train_data
    
    vocab = set()
        
    for index, intent, question, answer in all_data:
        vocab = vocab.union(set(str(index)))
        vocab = vocab.union(set(intent))
        vocab = vocab.union(set(question))
        vocab = vocab.union(set(answer))
        # Added padding -- It seems internally a 0 get's appended a throughout the process.
       
        with open(vocab_save_name, 'wb') as fp:
                pickle.dump(vocab, fp, protocol=4)        
    

    return vocab

# Alternative to creating new vocab -- load in previously created vocab set.
def load_vocab(filename):
    print("Loading Vocab ...")
    with open(filename, 'rb') as fp:
            vocab = pickle.load(fp)
    
    return vocab

# Adds padding to the length of the vocab.
def vocab_len(vocab):
    return len(vocab) + 1

# Used to create unique indexes for the words in the vocab to be used in vectorization of sentences.
def create_tokenizer(vocab):
    # No filters
    tokenizer = Tokenizer(filters=[])
    # This creates all the individual indexes for each word in the vocab.
    tokenizer.fit_on_texts(vocab)
    
    with open(tokenizer_save_name, 'wb') as fp:
                pickle.dump(tokenizer, fp, protocol=4)        
    
    return tokenizer

# Alternative to creating a new tokenizer, previously used tokenizer can be loaded in.
def load_tokenizer():
    with open(tokenizer_load_name, 'rb') as fp:
              tokenizer =  pickle.load(fp)     
              
              return tokenizer
             

"""
------------------------------------------------------------------------------------------------
Step 3: Vectorizing the data
    
    For the model to understand any of the text data we have to vectorize the intents, answers and questions.
    Using our unique set of words called vocab we give each word its own unique index.
    We then use these indexes to represent the sentences as vectors whocse components consist
    of the corresponding index values.
    
    Because we don't have identical size intent, questions and answers we need to determine the
    largest of each and pad every other vector with 0's to match that size.
    
------------------------------------------------------------------------------------------------
"""

# The following method actually performs the vectorization FOR the intent, answers and questions.
def vectorize_qa_data(data, word_index, max_intent_len, max_question_len):
    
    # intent
    X=[]    
    # Questions
    Xq =[]
    # Correct Answers
    Y = []
    
    # Creating vectors of all the indexes created above
    for index, intent, question, answer in data:
        
        # For each word in intent, Question and Answer find their corresponding index, 
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


# Vectorization FOR the questions and intents only - Somewhat redundant considering above function.
def vectorize_intents_data(data, word_index, max_question_len):
    
    # Questions
    X =[]
    # Correct Answers - intents
    Y = []
    
    # Creating vectors of all the indexes created above
    for index, intent, question, answer in data:
        
        # For each word in Question and Answer find their corresponding index, 
        # use list comprehension to create a vector of each index.
        x = [word_index[word.lower()] for word in question]
        
        # create a vector of all 0's and 1 one at answers index - intent is one word - i.e. smalltalk.agent.beautiful
        y = np.zeros(len(word_index)+1)
        y[word_index[str(intent[0])]] = 1
       
        # Then add that vector to the corresponding main list.
        X.append(x)
        Y.append(y)
    
        # return a tuple that can be unpacked - and pad each of the sequences!
    return (pad_sequences(X, maxlen=max_question_len), np.array(Y))


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
    
    
    The dropout is used to prevent overfitting of a model on a particular data set - currently not used.
    
-----------------------------------------------------------------
"""

from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, LSTM

# Possibly could be refactored into smaller functions but considering our use case of the model I think this is fine - it builds the model. concise.
def build_model(max_intent_len, max_question_len, vocab_len):
    # Two separate inputs - intents and questions - need a placeholder for now:
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
    #input_encoder_m.add(Dropout(0.3))
    # output -> (samples, intent_max_len, embedding_dim)
    
    # ------------------------- C ----------------------------------------
    # Input encoder C - Each input vector x has a corresponding output vector c 
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_len, output_dim=max_question_len))
    #input_encoder_c.add(Dropout(0.3))
    # output -> (samples, intent_max_len, max_question_len)
    
    # ------------------ Question Encoder (B) ----------------------------
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_len, output_dim=64, input_length=max_question_len))
    #question_encoder.add(Dropout(0.3))
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
    #answer = Dropout(0.5)(answer)
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
    
    return model

"""
-----------------------------------------------------------------
Step 6: Fit and Train network / Step 7 : Plot Accuracy

    Currently set to validate on training data - meaning "test" graph is of little use.
    This is done due to the nature of the dataset not requiring the network to predict 
    new answers but rather pick answers based on what it was trained on.
    
    These are more of the semi-redundant functions. For the time being the code is
    easier to understand this way. Naturally the code could be refactored to have arguments
    such as "input1" if space and concise code were a concern.
    
-----------------------------------------------------------------
"""

import matplotlib.pyplot as plt

def plot_training_model_accuracy(history):
    # Print out accuracy graph
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'Test'], loc='lower left')
    plt.show()


def train_intents_model(intents_model, questions_train, intents_train):
    
    # Fit model to data - in this case only the questions, 
    history = intents_model.fit([questions_train, questions_train], intents_train, batch_size=128, epochs=2000, validation_data=([questions_train, questions_train], intents_train))
    
    # Plot accuracy 
    plot_training_model_accuracy(history)
    
    # Save model - file name at top of program.
    intents_model.save(intents_model_save_name)
    
    return intents_model
    
# Trains the created / loaded model based on two inputs and the desired output
def train_qa_model(model, intents_train, questions_train, answers_train):
    
    # Fit model - in this case using intents, questions and answers.
    history = model.fit([intents_train, questions_train], answers_train, batch_size=128, epochs=2000, validation_data=([intents_train, questions_train], answers_train))
    
    plot_training_model_accuracy(history)
    
    # Save current training data
    model.save(model_save_name)

    return model    

"""
-----------------------------------------------------------------
Step 9 : Loading Models
    Alternatively the previously trained models can be loaded in.
-----------------------------------------------------------------
"""

from tensorflow.keras.models import load_model

# Loads a previously trained model
def load_prev_model(filename):
    print("Loading Model ...")
    model = load_model(filename)
    
    return model

"""
-----------------------------------------------------------------
Step 10: Setup Function

    Setup function refactored into several more concise functions:
    For convenience in main method still used as a collection point for the other method calls.
    
-----------------------------------------------------------------
"""

# The max lengths of all inputs and outputs are required to pad the sequences appropriately (See vectorize data function.)
def determine_max_lengths(all_data):
       # longest intent Name
    all_intent_lengths = [len(data[0]) for data in all_data]
    # max length 
    max_intent_len = max(all_intent_lengths)
    # max question length
    max_question_len = max([len(data[1]) for data in all_data])
    # max answer length
    max_answer_len = max([len(data[2]) for data in all_data])
    
    return all_intent_lengths, max_intent_len, max_question_len, max_answer_len


# Either creates a new vocab, tokenizer and vocab_length or loads previous one.
def retrieve_vocab(all_data):
    
    # Simple input prompt - User will never access to this script
    if input("Create New Vocab? (Y/N)").lower() in "yes":
        vocab = create_new_vocab(all_data)
        tokenizer = create_tokenizer(vocab)
        vocab_length = vocab_len(vocab)
    else:
        vocab = load_vocab(vocab_load_name)
        tokenizer = load_tokenizer()   
        vocab_length = vocab_len(vocab)
        
    return vocab, tokenizer, vocab_length

# Either builds and trains a new main QnA model or loads previous model - model name at top of script
def retrieve_qa_model(max_intent_len, max_question_len, vocab_length, intents_train, questions_train, answers_train):
    if input("Create New QA Model? (Y/N)").lower() in "yes":
        model = build_model(max_intent_len, max_question_len, vocab_length)
        model = train_qa_model(model, intents_train, questions_train, answers_train)
    else:
        model = load_prev_model(model_load_name)
        
    return model

# Either builds and trains a new intents model or loads previous model - model name at top of script
def retrieve_intents_model(max_questions_len, vocab_length, intents_model_questions_train, intents_model_intents_train):
    if input("Create New Intents Model? (Y/N)").lower() in "yes":
        intents_model = build_model(max_questions_len, max_questions_len, vocab_length)
        intents_model = train_intents_model(intents_model, intents_model_questions_train, intents_model_intents_train)
    else:
        intents_model = load_prev_model(intents_model_load_name)
        
    return intents_model


# Collection point for the above functions used to set up the data required to train or run the model.
def setup():
    
    # The data loaded depends on what data was processed and saved by the conversion script
    all_data, train_data, test_data = load_data_frames()

    # Same with the indexed answers' list
    idx_ans_list = load_idx_ans_list()
    
    # Since we're not using the train/test here - for clarity the following statement is used.
    train_data = all_data    
    
    # Retrieve Vocab and tokenizer - either creates new or loads previous data
    vocab, tokenizer, vocab_length = retrieve_vocab(all_data)
    
    # Retrive all the maximum lenghts of the arguments for the model - used to pad sequences.
    all_intent_lengths, max_intent_len, max_question_len, max_answer_len = determine_max_lengths(all_data)
    
    # Actual method call and unpacking of the tuples to set each respective variable below.
    intents_train, questions_train, answers_train = vectorize_qa_data(train_data, tokenizer.word_index, max_intent_len, max_question_len)
    
    # Possibly a bit too verbose lol, but at least its clear.
    intents_model_questions_train, intents_model_intents_train = vectorize_intents_data(train_data, tokenizer.word_index, max_question_len)
    
    model = retrieve_qa_model(max_intent_len, max_question_len, vocab_length, intents_train, questions_train, answers_train)
    intents_model = retrieve_intents_model(max_question_len, vocab_length, intents_model_questions_train, intents_model_intents_train)
      
    return model, intents_model, vocab, tokenizer, idx_ans_list, all_data

"""
-----------------------------------------------------------------
Step 11: GUI - 

    Ian I've done my best to refactor it into concise methods with
    one or two functions - I can only get it to print as it should
    with the funky function def going on in the main method rn.
    Feel free to undo all the refactoring on the GUI of course, I was
    not as confident in all its functionality as with the code I wrote obviously.
    
-----------------------------------------------------------------
"""
def get_intent_prediction(intents_model, my_question, tokenizer, max_question_len):
    
    # Strange format is because the same vectorization method from training is used - which takes all four arguments.
    my_intent_data = [(['0'], my_question, my_question, ['work'])]
    my_intent_question, my_intent_answer = vectorize_intents_data(my_intent_data, tokenizer.word_index, max_question_len)
    
    # Predict the most likley intent based on question
    pred_results = intents_model.predict(([my_intent_question, my_intent_question]))
    val_max = np.argmax(pred_results[0])
    
    # retrieve the value
    for key, val in tokenizer.word_index.items():
        if val == val_max:
            k = key
    
    # k is a string - this converts it back to the same format used in training.
    my_intent = seperate_punct_doc(nlp(k))
    
    return my_intent
   
def get_bot_answer(model, my_intent, my_question, tokenizer, max_intent_len, max_question_len):
    my_data = [(['0'], my_intent, my_question, ['work'])]
    my_intent, my_question, my_answer = vectorize_qa_data(my_data, tokenizer.word_index, max_intent_len, max_question_len)
    
    pred_results = model.predict(([my_intent, my_question]))
    val_max = np.argmax(pred_results[0])
    
    for key, val in tokenizer.word_index.items():
        if val == val_max:
            k = key
    
    return k

def print_bot_answer(bot_answer, hst, txt, response, idx_ans_list):
    if(str(bot_answer).isdigit()):
        #response.configure(text = ' '.join(idx_ans_list[int(bot_answer)-1][1]))
        #hst.insert(INSERT, "Chatbot: " + ' '.join(idx_ans_list[int(bot_answer)-1][1]) + "\n")
        response.configure(text = ' '.join(idx_ans_list[int(bot_answer)][1]))
        hst.insert(INSERT, "Chatbot: " + ' '.join(idx_ans_list[int(bot_answer)][1]) + "\n")
            
    txt.delete(0, END)
       
# Similar to set up function this function servers as a collection point for the other method calls.
def generate_answer(txt, hst, response, model, intents_model, vocab, tokenizer, idx_ans_list, max_intent_len, max_question_len):
                
    # Retrieve text from text box
    global my_intent
    my_question_text = txt.get()
                
    # Tokenize and remove punctuation
    my_question = seperate_punct_doc(nlp(my_question_text))
            
    
    
    """
    The following line works as advertised but in the case that the final question contains 0 words in vocab 
    an index out of bounds error is thrown in vectorizing the data - b/c question / intent will be empty list.
    This is a prime spot to implement either an intent to print "Can't answer that question" or straight up print it.
    The program does not break - the index out of bounds is handled by the GUI, it simply does not say anything back rn
    for example try saying "sassy" to the bot.
    """
    
    # Remove words not currently in vocab -
    my_question = [word for word in my_question if word in vocab]

    # Insert user's original question text in chat window.
    hst.insert(INSERT, "User: " + my_question_text + "\n")

    # If no vocab is found
    if not my_question:

        unknown_answers = ["Don't know.",  "What??", "I don't understand what you're trying to say.", "Let's talk about something else.", "Was that English?", "Can you try wording that differently?", "I'm not sure what that means."]

        random_index = random.randint(0, len(unknown_answers) - 1)

        hst.insert(INSERT, "Chatbot: " + unknown_answers[random_index] + "\n")
    
    else:
    
        # Predict the Intent based on the question
        my_intent = get_intent_prediction(intents_model, my_question, tokenizer, max_question_len)
         
        # Predict the Answer based on predicted intent and question / prompt
        bot_answer = get_bot_answer(model, my_intent, my_question, tokenizer, max_intent_len, max_question_len)
           
        print_bot_answer(bot_answer, hst, txt, response, idx_ans_list)
            
#Enter key event
def enter_hit(event):
    process_input()

def init_GUI(model, intents_model, vocab, tokenizer, idx_ans_list, max_intent_len, max_question_len):
      
    # Initialize window
    window = Tk()
    window.title("Chatbot")

    #Bind the enter key with pressing the send button
    window.bind('<Return>', enter_hit)
    
    # Adding widgets to the main GUI window  
    # Response will display all answers and prompts from the chatbot
    response = Label(window, text = "Enter Message:")
    response.grid(column = 0, row = 0)
    
    # txt is where the user will query the chatbot
    txt = Entry(window, width = 30)
    txt.grid(column = 0, row = 1)
    
    # hst is a scrollable history of the conversation between the user and the chatbot
    hst = scrolledtext.ScrolledText(window, width = 100, height = 10)
    hst.grid(column = 0, row = 4)
    
    # Add button widgets to main GUI window and attach functions to them.
    btn = Button(window, text = "Send", command = process_input)
    
    btn.grid(column = 0, row = 3)
    
  
    return window, txt, hst, response

         

def process_input():
        generate_answer(txt, hst, response, model, intents_model, vocab, tokenizer, idx_ans_list, max_intent_len, max_question_len)
        
"""
---------------------------------------------------------------------------------
Main Program Loop for Testing - 

    Ian I've done my best refactoring the GUI's methods, 
    I can only get it to process the input correctly this way?
    
---------------------------------------------------------------------------------
"""
if __name__ == '__main__':
   
    # Initialize model and GUI
    model, intents_model, vocab, tokenizer, idx_ans_list, all_data = setup()
    all_intent_lengths, max_intent_len, max_question_len, max_answer_len = determine_max_lengths(all_data)    
    window, txt, hst, response = init_GUI(model, intents_model, vocab, tokenizer, idx_ans_list, max_intent_len, max_question_len)
    
    window.mainloop()
    
    
