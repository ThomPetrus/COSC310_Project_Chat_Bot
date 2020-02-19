# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 19:21:17 2020

ChatBot Prototype 1 

@author: tpvan
"""
# main imports - nltk currently unused.
import nltk
# spacy nlp library
import spacy
nlp = spacy.load('en_core_web_sm')
# regular expressions - python
import re

#####################################
# Basic Functions for preprocessing #
#####################################
"""
  Remove punctuation - using regular expressions - no library - core python.
  input is string literal
  outputs is a list 
"""
def remove_punctuation(user_input_text):
    pattern = r"[^!.?, ]+"
    filtered_words_list = re.findall(pattern, user_input_text)
    return filtered_words_list
"""
 Spacy - basic stop words
 input is a regular list of string literals
 output is also a list

"""
def remove_stop_words(user_input_list):
   filtered_words =[]
   for word in user_input_list:
       if not (nlp.vocab[word].is_stop):
            filtered_words.append(word)            
   return filtered_words
"""
 Spacy - lemmatization 
 input is list of string literals
 outputs a list of string literals
"""
def lemmatization(user_input_list):
    doc = []
    lemmas = []
    for token in user_input_list:
        doc.append(nlp(token))
    for token in doc:
        lemmas.append(token[0].lemma_)
    return lemmas


"""
Pre-Process function

This function should be the entry point for using all the other pre-processing 
functions prior to the processing of the actual question or statements.
We'll figure out what works best later I guess. Currently this just removes punctuation,
stop words and finds the lemmatization of each word. Does filter a lot out.

I think the stop words should probably be altered, 
see : print(nlp.Defaults.stop_words) for a list of all of them, can be easily altered.

potentially keep question marks in as well for indicating questions?
"""
def pre_process(user_input_text):
    user_input_list = remove_punctuation(user_input_text)
    user_input_list = remove_stop_words(user_input_list)
    user_input_list = lemmatization(user_input_list)
    return user_input_list
        
# Main Loop
if __name__ == "__main__":
    input_count=0
    
    print("Hello! Welcome to the first iteration of our ChatBot")
    
    # Change to keyword quit or something.
    while(input_count<3):
        input_count+=1
        
        # retrieve input
        user_input = input("What is your question?\n")
        
        # pre-process - punctuation / stopwords / lemmas - returns list
        user_input = pre_process(user_input)
        
        print(user_input)
        

