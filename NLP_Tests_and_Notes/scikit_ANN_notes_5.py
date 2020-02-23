# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:56:11 2020
@author: tpvan

!! This uses keras !!
Anaconda prompt with admin :
    conda install -c conda-forge keras

Deep Learning - NLP - Overview :
    recurrent neural networks - RNN
    Long, short term memory - LSTM
    Use LSTM to generate text from source corpus (set of docs)
    Create a QA chatbot - from here we'll build I guess if it works.
    
Before that : 
    ANN:
        Perceptrons:
        -use perceptrons to mimic biological neurons.
        -each perceptron has inputs and outputs
        -each input is weighted to account for importance
        -each perceptron has an activation function and only fires if the inputs satisfy
         that function. There are tons of different activation functions. For simplicity
         say the function is 1 when pos, 0 when neg
        -we can introduce a bias term (b) to account for 0 input - i.e. 0 * weight is always 0
            z = sigma(Wi * Xi + bias)...
        
        Structure - connected perceptrons through input and outputs
        -Input layers, hidden layers and output layer - Im sure you've seen the structure before
         first and last are input and output respectively and anything in between is hidden.
        -Hidden layers don't see the input or outputs.
        -More than 3 hidden layers is a 'deep network'
        
        Activation Functions:
        -Naturally a functions as described above is binary in output
         we can have more complicated dynamic functions, such as a smooth line as opposed to a step.
             google activation function if you're unsure, picture-ish : 
                 
             1|    ______
              |    |
             0|____|______ vs curve
        
        -The curve is called the sigmoid function: 1/(1+e^(-(x)))
        -Others include the Hyperbolic Tangent function - same curve ish but ranges from -1 to 1 
        -Rectified linaer unit (ReLU) - function: max(0, z).. 0 until positive then linear line x=y -ish, z = wx+b
         THIS one tends to have the best performance - most libraries have them built in.
         
         Using the iris dataset ( built in ) - Flowers dataset.
"""
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()

# load data
X = iris.data
y = iris.target

# convert to categorical vector - just cuz built in 
from keras.utils import to_categorical
y = to_categorical(y)

# split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scale / standardize data -- there's other methods, this one basically just divides values by max
from sklearn.preprocessing import MinMaxScaler
scaler_object = MinMaxScaler()

# fit to train data
scaler_object.fit(X_train)

# scale / standardize
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

# imports for ANN
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


"""
Input is number of units / nodes - should be a multiple of number of features / inputs
followed by input dimensions, in this case 4 - b/c four features compared
finally add the activation function, in this case wer're using that ReLU described above
Adding two dense layers.
"""
model = Sequential()
model.add(Dense(8,input_dim=4, activation='relu'))
model.add(Dense(8,input_dim=4, activation='relu'))
#output layer has diff activation function:
model.add(Dense(3, activation='softmax'))

# compile - loss depends on what we're performing - in this case categorical
#           optimizer - idk I guess it's called adam lol, metrics defines what were training for
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# quick summary of current model
print(model.summary())

# fit data as before - epochs = nr ot iterations, verbose = info, 0 is nothing, 1 is progress bar, 2 is extra info
model.fit(scaled_X_train, y_train, epochs=150, verbose=2)

# test model against test data
print(model.predict(scaled_X_test))

prediction = model.predict_classes(scaled_X_test)

# print data same same
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test.argmax(axis=1), prediction))
print(classification_report(y_test.argmax(axis=1), prediction))
print(accuracy_score(y_test.argmax(axis=1), prediction))

# SAVING THE MODEL AFTER TRAINING!!!!
model.save('myfirstmodel.h5')

# loading :
#from keras.models import load_model
# new_model = load_model('myfirstmodel.h5')