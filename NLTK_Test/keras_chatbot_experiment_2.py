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
    
    The actual split is fairly easy and performed by the scikit library in this case.
    
    ROAD BLOCK -- the way the data was split before worked because the stories were composed of
    the same words as the questions and answers - in this case the articles are super limited in their words.
    I will try to do a split as before in the text generation experiment. 
    
    If that yields no results I will manually split the date per articlee in about 70/30 split.
    !
    !
    !
    !
    !
    !
    !
    !
    !
    !
    !
    !
    !
    !
    !
    !
    
    That sounds like a tomorrow thing. It's late as balls.
--------------------------------------------------------------------------------------------------
"""
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

# Create seperate lists for article, questions and answers
train_article_text = []
train_question_text = []
train_answer_text = []

# Seperate the data frame list into seperate lists
for i in range(len(df_list)):
    train_article_text.append(df_list[i][0])
    train_question_text.append(df_list[i][1])
    train_answer_text.append(df_list[i][2])
    
    
def vectorize_data(data, word_index=tokenizer.word_index, max_article_len=max_article_len, max_question_len=max_question_len):
    
    # Article Names
    X=[]    
    # Questions
    Xq =[]
    # Correct Answers
    Y = []
    
    # creating vectors of all the indexes created above
    for article, question, answer in data:
        # for each story
        x = [word_index[word.lower()] for word in article]
        xq = [word_index[word.lower()] for word in question]
        y = [word_index[word.lower()] for word in answer]
        
        X.append(x)
        Xq.append(xq)
        Y.append(y)
        
        # return a tuple that can be unpacked - and pad each of the sequences!
    return (pad_sequences(X, maxlen=max_article_len), pad_sequences(Xq, maxlen=max_question_len), pad_sequences(Y, maxlen=max_answer_len))



stories_train, questions_train, answers_train = vectorize_data(train_data)
stories_test, questions_test, answers_test = vectorize_data(test_data)