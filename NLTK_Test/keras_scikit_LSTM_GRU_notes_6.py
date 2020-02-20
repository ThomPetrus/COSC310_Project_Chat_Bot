# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:33:14 2020

Hey fellas, most of this is not really required knowledge as this occurs under the hood 
but it is super interesting if you ask me so I've left it in.

Recurrent Neural Networks
    -Best suited for sequenced data:
     time, sentences, audio, music etc
    -Unlike the previous notes section recurrent neurons send output back to themselves
     as one of the inputs + current timestep inputs.
    -Cells that are a function of inputs from prev time steps are called memory cells
    -We can create entire recurrent layers of neurons
    
    -Many different structures:
        -sequence to sequence, sequence to vector ...
        vector to sequence : example - passing in a word, and returning a sequence of probability phrases
        that would actually come out (probably best suited for chat?)
     
    -Long short term memory cells :
        We need long term memory to ensure initial input are not forgotten. LSTM.
        I found a diagram and included it in the folder.
        
    -at time step 't': let _ indicate subscript.
        -first step : f_t - forget gate layer - what do we throw away:
          input is a linear combination of h_t-1 (output from prev neuron), and x_t (current input) with some weights W
          which is passed into the sigmoid function (the curvey one w/ output between 0-1) == 1 keep, 0 forget
        -second step : what new information should be stored in the cell state 
            i_t : input gate layer - sigmoid function - same linear combination of h_t-1, and x_t
            C_t~ : hyperbolic tangent layer - same linear combination but passed into the hyperbolic tangent function
                  returns a vector called new candidate values.
        -third step : new cell state is C_t:
            combines the output of the previous steps to create the cell state:
                C_t = f_t * C_t-1 + i_t*C_t~
                where f_t is stuff to be rememner or forgotten, 
                C_t-1 is prev cell state,
                i_t is multiplied by C_t~ 
    
    This is likely gibberish without looking at the diagram and associated mathemcatical functions.
    
    
    
@author: tpvan
"""

