# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:37:17 2020

    Needed to reformat the data from the data set to match the
    data set format from the previous chat bot prototype.

    Converts the tsv file to the desired list of tuples. Each index of the
    tuple consists of a list of the words in the article, question and answer.

    There's likely a more efficient way of doing this but it works.

    Also splits the data into training data and test data.

@author: tpvan
"""
"""
--------------------------------------------------------------------------------------------------------
Imports
--------------------------------------------------------------------------------------------------------
"""
import pandas as pd
import pickle
import spacy
nlp = spacy.load('en_core_web_sm')

"""
--------------------------------------------------------------------------------------------------------
Read file - tsv
--------------------------------------------------------------------------------------------------------
"""
def read_file(filepath):
    with open(filepath) as f:
        text = f.read()
    
    return text

# Returns pandas DataFrame
data_frame = pd.read_csv('chat_data.tsv', sep='\t')

"""
--------------------------------------------------------------------------------------------------
'Clean' data
--------------------------------------------------------------------------------------------------
"""
if(input("Clean Data? (Y / N)") in "y"):
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
Process Data into desired structure
--------------------------------------------------------------------------------------------------
"""

# Converts to a list of lists
df_list = [list(x) for x in data_frame.to_records(index=True)]

# Used to remove all the punctuation and turn every string into a list of words.
def seperate_punct_doc(doc):
    return [token.text for token in doc if token.text not in '\n\n \n\n\n--"#$%&()*+,-/:;<=>@[\\]^_`{|}~\t\n']
            
print("Loading ... ")
# Ensure data is a string, then tokenize, then remove punctuation, returns a list.
for i in range(len(df_list)):
    idx = df_list[i][0]
    df_list[i][0] = [word.lower() for word in seperate_punct_doc(nlp(str(df_list[i][0])))]
    df_list[i][1] = [word.lower() for word in seperate_punct_doc(nlp(str(df_list[i][1])))]
    df_list[i][2] = [word.lower() for word in seperate_punct_doc(nlp(str(df_list[i][2])))]
    df_list[i][3] = [str(idx)] + [word.lower() for word in seperate_punct_doc(nlp(str(df_list[i][3])))]
          
   
# Convert the lists consisting of [Article, Question, Asnwer] to a tuple
df_tuple = [tuple(x) for x in df_list]

# Serialize object.
with open('qa_df.txt', 'wb') as fp:
           pickle.dump(df_tuple, fp, protocol=4) 
            
# Ensure output is as desired.
with open('qa_df.txt', 'rb') as f:
    df_loaded = pickle.load(f)
    
print(df_loaded)
    
"""
--------------------------------------------------------------------------------------------------------
Train Test Split


    Note that the current bot does not use a train test split.
    We could try an implement a proper test split and all that considering this dialogue data set is more 
    suitable for trying a proper test, but it does not matter. The validation/test data for the fitting does 
    not affect the training of the model.


--------------------------------------------------------------------------------------------------------
"""

from sklearn.model_selection import train_test_split


# There absolutely must be a better way of doing this.
# X is articles and questions
X=[]
# y is answers
y=[]

indexed_answers = []

# append articles, questions and answers to X or y respectively -- X always represents 'features' and y is 'targets'
for i in range(len(df_list)):
    X.append([df_list[i][0], df_list[i][1], df_list[i][2]])
    y.append([df_list[i][3]])
    indexed_answers.append([df_list[i][0], df_list[i][3]])


#print(indexed_answers)

# Use the sklean train_test_split method - and shuffles if include 'randomstate = 42'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
# Copy the appropriate answers from y_train to X_train to create train data portion
# This will likely need refactoring -
for i in range(len(X_train)):
    X_train[i].append(y_train[i][0])


# Same thing for the test split
for i in range(len(X_test)):
    X_test[i].append(y_test[i][0])

# for clarity
train_data = X_train
test_data = X_test

indexed_train_answers =[]
# append articles, questions and answers to X or y respectively -- X always represents 'features' and y is 'targets'
for i in range(len(train_data)):
    indexed_train_answers.append([train_data[i][2][0], ' '.join(train_data[i][2][1:])])

# Convert the lists consisting of [Article, Question, Asnwer] to a tuple.
df_train = [tuple(x) for x in train_data]
df_test = [tuple(x) for x in test_data]

print(df_train)

# Serialize objects / write actual text to file.
# training data
with open('qa_train_df_v2.txt', 'wb') as fp:
            pickle.dump(df_train, fp, protocol=4) 

# print to Ensure output is as desired.
with open('qa_train_df_v2.txt', 'rb') as f:
    df_train_loaded = pickle.load(f)
    
# test data
with open('qa_test_df_v2.txt', 'wb') as fp:
            pickle.dump(df_test, fp, protocol=4) 

with open('qa_test_df_v2.txt', 'rb') as f:
    df_test_loaded = pickle.load(f)


# indexed answers
with open('qa_indexed_ans_v2.txt', 'wb') as fp:
            pickle.dump(indexed_answers, fp, protocol=4)

with open('qa_indexed_ans_v2.txt', 'rb') as f:
    df_idx_ans = pickle.load(f)
            
# indexed answers - train data
with open('qa_train_indexed_ans_v2.txt', 'wb') as fp:
            pickle.dump(indexed_train_answers, fp, protocol=4)

# indexed train answers text representation            
with open('qa_train_indexed_ans_text_v2.txt', 'w') as f:
    for i in range(len(indexed_train_answers)):
            f.write("Article : " + ' '.join(df_train[i][0]) +", Question: "+ ' '.join(df_train[i][1]) + ", Answer: " + ' '.join(df_train[i][2]) + '\n')
            





   