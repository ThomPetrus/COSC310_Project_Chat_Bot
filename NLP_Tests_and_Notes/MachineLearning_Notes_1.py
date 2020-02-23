# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:47:12 2020

SciKit - Text Classification - ML

Data sets? 
https://lionbridge.ai/datasets/15-best-chatbot-datasets-for-machine-learning/

movie dialogue:
    http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

ML Notes :
    Using supervised learning:
        algos trained using labeled examples, input and desired output is known
        learning algo receives set of input & outputs, then learns
        by comparing the actual ouput with correct ouput to find errors
        then modifies the model accordingly.
        
        Used where historical data can predict likely future events.
        
        -Get data, data cleanining, data is split in test data, and training data
        training set is usually 70% of data the other 30% is used to test after data ->model testing
        Then can iterate over the last couple steps to tune system
        
        - This tutorial uses a pre labelled data set of spam email vs reg email.
        
    Evaulation metrics:
        accuracy, recall, precision, f1-score (recall+precision kinda)
        fundamentally a model only has binary output - correct vs incorrect
        We pass the test data into the trained model and compare the prediction to what we
        want as output and label them correct v incorrect. 
        
        One metric is usually not sufficient however, considering an answer kind of exists on a spectrum
        We can organize the predicted values compared to the real values in a confusion matrix
        
        Accuracy - number of correct predictions / total nr of predictions
            - good metrix with balanced classes or outcomes - i.e. spam vs regular mail
              not good for unbalanced classes.
        
        recall - the ability of a model to find all the relevant cases within a dataset
                the number of true positives / (number of true positives + false negatives)
        
        Precision - ability of a model to identify only the relevant data points
                    true positives / (number of true positives + number of false positives)
        
        F1 - optimal blend between recall an precision.
            The harmonic mean of precision and recall:
                F1 = 2 * (precision * recall) / (precision + recall)
            Harmonic means punishes extreme values. i.e. plug in 1 and 0 , mean is 0.5, h.mean is 0
            
        confusion 'matrix' is just a graph with all possible conditions
                                   predicted 
                            predict ham    predict spam     i.e. :
        real | real ham  |    true pos   |   false neg          50 | 10
             | real spam |    false pos  |   true neg           5  | 100
        
        
        google: generalized confusion matrix you'll see a better examples 
        
        hopefully this spam vs reg email will translate to good response vs bad response
        
        Actual values for prediction, recall and accuracy vary and depending on the situation one of the
        metrics are prioritized. Ideally you want high numbers for all but not necessary.
        
        I realize there's likely better algorithms than classification. We'll see what the rest of 
        this course brings.
        
        
    Vectorizing data:
        from raw text to vectorize format - a numerical matrix.
    
@author: tpvan
"""



