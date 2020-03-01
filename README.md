# COSC310_Project_Chat_Bot
 COSC 310 Python Chat Bot

Required Libraries:
This Python Chat bot was built in the Spyder IDE. More importantly it requires the Spacy library as well as the Keras library to run properly. These should be installed using pip or anaconda prompt depending on preference and what environment used.

The current chat bot iteration requires entry of the particular intent as well as the question. The qa_train_indexed_ans_text_v2 text file contains all the intents, question and associated answers for testing. Once you run the program simply say no to the prompts askign to create new vocab / new model and the GUI will run.

Repository Structure:
The Repository consists of a NLP notes folder containing a variety of experiments mainly using Spacy, NLTK, Keras and SciKit used primarily to understand the platform better.

There is also a data sets folder containing the data sets used during development.
The primary dataset currently used is the chat_data.tsv. It is a dialogue dataset from DialogueFlow. 

The prototypes folder contains the first 4 iterations of the chat bot as well as the two papers referenced. In particular the paper 'End-To-End Memory Networks' by Sainbayar Sukhbaata from Dept. of Computer Science Courant Institute, New York University and Arthur Szlam, Jason Weston and Rob Fergus from Facebook AI Research New York. 

ChatBot Structure:
Provided the libraries are installed properly the only thing required to run it is the actual keras_chatbot_prototype_# python file. The other main file is a data set conversion script which takes in the tsv or csv files and formats it in such a way the program can use it. The other files store the datasets processed in various ways including indexed answers as well as train and test data split by the conversion file.


 
Edit History :
Project Documentation
Feb 4:
- Set up the repository and created a test file using the Python NLTK.
- Here's a link with all the howto's:
  https://www.nltk.org/howto/

Feb 17-19:
Completed 
Credits to supplementary material and information goes to Pierian Data and their Udemy Course on NLP, code and annotations produced by Thomas Van De Crommenacker based on said material.
No copyright infringement intended and all used for educational purposes.
-Not all the info is relevant to the creation of the chatbot, though all is related to NLP in some form or another. Simply used to get familiar with the material, platforms and frameworks we want to use for the project.

Feb 20:
-Started Experimentation with a library called Keras for creating RNN models to generate text and chatbots.
 Chat bot experiment with Keras will employ a End to End network based on a paper referenced in the course. 
 
 Here's the paper's names. I read the first one with reasonable success, the second one I barely touched.
 The paper is called 'End-To-End Memory Networks' by Sainbayar Sukhbaata from Dept. of Computer Science Courant Institute, New York University and Arthur Szlam, Jason Weston and Rob Fergus from Facebook AI Research New York. 
 As well as 'Memory Networks' by Jason Weston, Sumit Chopra & Antoine Bordes from Facebook AI Research.

Feb 21:
-Finished the basic Keras chat bot based on Pierian Data tutorial and the End to End network paper. The Chat bot answers
really basic yes or no questions. It works really well for the basic structure of the data set and answers provided. 
 Now the goal is to extrapolate what we've learned to a more complicated network that will process more complicated questions and more importantly more specific answers. 

-On the link Andy found there was a wikipedia article Q.A. that we'll try to use next.
These data were collected by Noah Smith, Michael Heilman, Rebecca Hwa, Shay Cohen, Kevin Gimpel, and many students at Carnegie Mellon University and the University of Pittsburgh between 2008 and 2010. (http://www.cs.cmu.edu/~ark/QA-data/)

We're not writing a research paper but as far as documentation goes:

Please cite this paper if you write any papers involving the use of the data above:

    Question Generation as a Competitive Undergraduate Course Project
    Noah A. Smith, Michael Heilman, and Rebecca Hwa
    In Proceedings of the NSF Workshop on the Question Generation Shared Task and Evaluation Challenge, Arlington, VA, September 2008. 
    
Feb 22 : Finished the Wiki QA Bot prototype.

Feb 23 : Experiment with different parameter values for the network trying to get an 
optimal result for this current data set and application.

Considering our use case if the network I get the highest degree of accuracy with a 50/50 split, and training the network and validating it against the training data (see the model.fit line). It is around 80%. Obviously this is not exactly how this should be used but as long as it finds the right answer for each question. I have printed the corresponding train text questions, answers and articles to a seperate file. When you ask the appropriate question it has a fairly high success rate. Will work on making sure user can enter any question without throwing an error next. From there we can make it more conversational in verious ways. I had a couple ideas:
Obvs Andy's idea with making the bot determine the topic rather than the list.
having it print the list regardless if asked?
Asking whether it was the right answer?
Apologize if wrong? Store and write correct answers to a file?

Feb 29 : 
Prototype 4 uploaded - incorporates Ians GUI and refactored the main script to be legible. Some functions could require further refactoring.

