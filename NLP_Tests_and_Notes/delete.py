# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:10:42 2020

@author: tpvan
"""



import pickle
import numpy as np
import spacy
nlp = spacy.load('en_core_web_sm')


with open('qa_df.txt', 'rb') as f:
    all_data = pickle.load(f)

with open('qa_train_df_v2.txt', 'rb') as f:
    train_data = pickle.load(f)

# Not actually used for the answers from Wiki.
with open('qa_test_df_v2.txt', 'rb') as f:
    test_data = pickle.load(f)    
    
with open('qa_indexed_ans_v2.txt', 'rb') as f:
    idx_ans_list = pickle.load(f)
    

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
Step 9 (opt): Loading Model
-----------------------------------------------------------------
"""
from tensorflow.keras.models import load_model
model = load_model('chat_bot_experiment_1000_128_dialogue_dropout_validated_on_train_v1.h5')


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


