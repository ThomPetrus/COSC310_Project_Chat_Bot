# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:33:14 2020

Hey fellas, most of this is not really required knowledge as this occurs under the hood 
but it is super interesting if you ask me so I've left it in. Also won't really make sense without the image.
If you open the image and find the functions associated with each step and read the notes below it should be 
fairly simple to understand. It's just a bunch of linear combinations of inputs and prev input passed into sigmoid or hyperbolic tangent 
functions as a way to determine what to store, pass on, etc.

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
        -fourth step - final decision - h_t
            filtered version of cell state.
            same linear combination of h_t-1 and x_t passed into sigmoid function multiplied by
            tanh(C_t), C_t was calculated in last step .
            tanh is hyperbolic tangent function.
            
    This is likely gibberish without looking at the diagram and associated mathemcatical functions.
    
    There's different variants:
        peephole variant: adds 'peepholes' at each gate to allow each gate (f_t, i_t, o_t)
            to see the previous cell state C_t-1
        Gated recurrent Unit (GRU): combines the forget and input gate into an update gate.
            also merges the cell state and hiden state. Not much more info here.
            
FORTUNATELY KERAS HAS BUILT IN APIS FOR LSTM AND RNNs woopwoop.

@author: tpvan
"""

