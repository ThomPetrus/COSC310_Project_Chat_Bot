# COSC310_Project_Chat_Bot
 COSC 310 Python Chat Bot
 ***************************************************************************
 Authors:
 
 Andres Escobedo
 
 Ian Heales
 
 Thomas Van De Crommenacker
***************************************************************************
A2 - Script with Chat Bot only functionality is in :
/Prototypes_experiments/ChatBot_v2_A2_DELIVERABLE/Bot and Training Script/

The .py File called:
"keras_chatbot_experiment_6_intent_experiment_refactored_only_bot" in :

All possible question and answer combinations can be found in:
Prototypes_experiments/ChatBot_v2_A2_DELIVERABLE/Data/

The text file called:
"qa_train_indexed_ans_text_v3.txt"

All other files and directory structure explained below.
***************************************************************************

Required Libraries:

IMPORTANT - For compilation and running the chatbot.

This Python Chat bot was built in the Spyder IDE. More importantly it requires the Spacy library as well as the Keras library to run properly. These should be installed using pip or anaconda prompt depending on preference and what environment used.

Links to both:

https://anaconda.org/conda-forge/keras

https://spacy.io/

It also uses Pickle for serializing and deserializing objects, Pandas for reading csv and tsv files, Numpy the math libraby and Tkinter for the GUI. If you don't have any of these for whatever reason they will also need to be installed.

***************************************************************************
Bot Description:

The current chat bot iteration interprets the intent and answers questions based on the dialogueFlow dataset. The qa_train_indexed_ans_text_v3 text file contains all the intents, question and associated answers for testing. The current prototype has not been fully refactored into its own script.
Therefore, once you run the program simply say no to the prompts asking to create new vocab / new model and the GUI will run. The model trained is a single layer of the End to End network from the paper by by Sainbayar Sukhbaata from Dept. of Computer Science Courant Institute, New York University and Arthur Szlam, Jason Weston and Rob Fergus from Facebook AI Research New York. This paper was referenced in an online course on NLP and Python by Pierian data, which was used as reference during the early stages of development.

***************************************************************************
Repository Structure:

The Repository consists of a NLP notes folder containing a variety of experiments mainly using Spacy, NLTK, Keras and SciKit used primarily to understand the platform better. These were used to communicate ideas and techniques between the team members.

There is also a data sets folder containing the data sets used during development.
The primary dataset currently used is the chat_data.tsv. It is a dialogue dataset from DialogueFlow. 

The prototypes folder contains the first 5 iterations of the chat bot as well as the two papers referenced. In particular the paper 'End-To-End Memory Networks' by Sainbayar Sukhbaata from Dept. of Computer Science Courant Institute, New York University and Arthur Szlam, Jason Weston and Rob Fergus from Facebook AI Research New York. 

Each protoptype should have what is required to run it in each individual folder. This also includes a simple conversion script we wrote used to convert the CSV and TSV files into the desired Python data structures required to process the data sets. For most it also includes one or more trained versions of the models to use, though only in later iterations can you switch with user prompt or even yield any results to speak of.

***************************************************************************
Edit History :

Feb 4 - NLTK? :
Set up the repository and created a test file using the Python NLTK.

Here's a link with all the howto's:
  https://www.nltk.org/howto/

Feb 17-19 - Completed Course - Continueing Research.
Credits to supplementary material and information goes to Pierian Data and their Udemy Course on NLP, code and annotations produced by Thomas Van De Crommenacker based on said material.
No copyright infringement intended and all used for educational purposes.
Not all the info is relevant to the creation of the chatbot, though all is related to NLP in some form or another. Simply used to get familiar with the material, platforms and frameworks we want to use for the project.

Feb 20 - Library Experiments - Read Papers:
Started Experimentation with a library called Keras for creating RNN models to generate text and chatbots.
Chat bot experiment with Keras will employ a End to End network based on a paper referenced in the course. 
 
Here are the paper's names. I read the first one with reasonable success, the second one I barely touched. The paper is called 'End-To-End Memory Networks' by Sainbayar Sukhbaata from Dept. of Computer Science Courant Institute, New York University and Arthur Szlam, Jason Weston and Rob Fergus from Facebook AI Research New York. 
As well as 'Memory Networks' by Jason Weston, Sumit Chopra & Antoine Bordes from Facebook AI Research.

Feb 21 - Finished Yes/No Bot - Continue research on how to evolve this.
Finished the basic Keras chat bot based on Pierian Data tutorial and the End to End network paper. The Chat bot answers really basic yes or no questions. It works really well for the basic structure of the data set and answers provided. 
Now the goal is to extrapolate what we've learned to a more complicated network that will process more complicated questions and more importantly more specific answers. 

Feb 22 - DataSet for Wikipedia QA ? 
On the link Andy found there was a wikipedia article Q.A. that we'll try to use next.
These data were collected by Noah Smith, Michael Heilman, Rebecca Hwa, Shay Cohen, Kevin Gimpel, and many students at Carnegie Mellon University and the University of Pittsburgh between 2008 and 2010. (http://www.cs.cmu.edu/~ark/QA-data/)

We're not writing a research paper but as far as documentation goes:

Please cite this paper if you write any papers involving the use of the dataset above:

    Question Generation as a Competitive Undergraduate Course Project
    Noah A. Smith, Michael Heilman, and Rebecca Hwa
    In Proceedings of the NSF Workshop on the Question Generation Shared Task and Evaluation Challenge, Arlington, VA, September 2008. 
    
Feb 23: Finished the Wiki QA Bot prototype.

Feb 23 : Experiment with different parameter values for the network trying to get an 
optimal result for this current data set and application.

Considering our use case if the network I get the highest degree of accuracy with a 50/50 split, and training the network and validating it against the training data (see the model.fit line). It is around 80%. Obviously this is not exactly how this should be used but as long as it finds the right answer for each question. I have printed the corresponding train text questions, answers and articles to a seperate file. When you ask the appropriate question it has a fairly high success rate. Will work on making sure user can enter any question without throwing an error next. 

Feb 29 - Incorporate GUI : 
Prototype 4 uploaded - incorporated Ians GUI and refactored the main script to be legible. Some functions could require further refactoring.

Mar 1 - Intent Filter - Needs refactoring now : 
Trained the model on sctrictly the questions in the data set with the intents as answers. Bot now uses that model to identify intents and based on the intent produces an answer. Works well with the questions in the dataset, considering our use case of the model it can not predict what to say based on new input. Refactoring next for legibility. Likely write seperate script for model, no point in keeping it in the script that trains models.

Mar 4 - No vocab checking :
Added the functionality of generating a randomized answer from a list stating that the chat bot didn't know how to respond. This is called whenever the user enters vocab which is not in our data set at all.

Mar 7 :
Added some more universal dirrectory handling so that it will work if you are not in exactly the right directory and it should also work on a server. Added a key binding to the enter key so you can press enter to send your message.

Mar 10 :
Created a dataset conversion py script to convert some yaml files to csv files so that we can add them to our current data set. The script puts them into the same format as how we want them (with intent, question, and answer).

Mar 17 :
Finished the client and server prototypes.

Mar 27:
Updated the GUI so that the input text box is at the bottom and there is space for the chat bot picture representing its emotions.

Mar 30:
Incorporated the bot picture which changes according to which intent is output. 

April 1st : 
Created the server chatbot. This chatbot does away with the GUI in favor of simply running inside of a 'while True' loop.
This allows for our chatbot to 'talk' with itself, with one instance being a client and another being a server.

April 2nd: 
Created the client chatbot. I might have to change the GUI for it later, but for now it works perfectly. The user has a choice when they 
run the chatbot program to run it in 'client-mode,' since I have to load in the GUI differently.

April 2nd:
Added some chatterbot datasets to our dataset.

April 1st:
Added Part of Speech and Named Entity Recognition tagging, prefix or postfix * for POS and & for NER.

# COSC310_Features
***************************************************************************
# Sockets Implementation:
Allows a client chatbot to create a connection with a server chatbot. Through this connection, the client and server can then send and recieve data until the connection is closed. Since the module we used was the multiprocessing module, this was a strict upgrade over the sockets module because we could run the GUI concurrently.

Sockets GUI:
Gave the user a history of the conversation between the client chatbot and the server chatbot. This GUI also allows the user to safely close the connection at any time. This is an improvement over closing the GUI window, which can throw a ConnectionClosed exception.
Sample Dialogue :
>Server: how your day is going  
>Client: couldn't be better.  
>Server: great !  
>Client: terrific !  
>Server: How are you getting on?  
>Client: okay . i'm here.  

# New GUI
Now includes a dynamically changing image of our chatbot that reacts differently based on input. While it does not directly affect the conversational ability of the chat bot this really adds a personality to the application.
See the report for examples.

# Improved Data Set
The chat bot is now trained on 1929 different question and answer pairs to improve the chatbot's ability to respond to the user's input.
There is a variety of new topics included now including movies, literature, AI, food and many more.

>User: Heya!
>Chatbot: howdy .
>User: can you help me?
>Chatbot: my pleasure .
>User: What is illuminatus?
>Chatbot: a weird sci fi trilogy written by robert anton wilson and robert shea about conspiracies competing to control the world .
>User: awesome!
>Chatbot: thanks i try .
>User: Have you read homer?
>Chatbot: he wrote the iliad and the odyssey did n't he ?
>User: you are right!
>Chatbot: of course i am .
User: do androids dream of electric sheep?
Chatbot: that is a good story by philip k. dick . have you read valis or the man in the high castle ?

# Part of Speech and Named Entity Recognition
Due to the method used to train our models we have not yet fully integrated POS or NER tagging into determining the output in conversation directly, you can however ask the chat bot to define the POS for any word or ask what a certain entity is if it is recognized.

>User: * Hey
>Chatbot: hi there friend !
# >Chatbot: Oh btw, You placed a Part of Speech Pointer on "Hey", Hey is a interjection
>User: What is * working ?
>Chatbot: i always have time to help you out . what can i do for you ?
# >Chatbot: Oh btw, You placed a Part of Speech Pointer on "working?", working? is a verb, gerund or present participle
>User: What is * working ?
>Chatbot: i always have time to help you out . what can i do for you ?
# >Chatbot: Oh btw, The Part of Speech for "working" is : verb, gerund or present participle
>User: cool
>Chatbot: i 'm glad you think so .
>User: what is &Apple ?
>Chatbot: indeed .
# >Chatbot: Oh btw, NAMED ENTITY DETECTED. lol jk, but "Apple" is a Companies, agencies, institutions, etc.
>User: who is &Trump ?
Chatbot: indeed .
# Chatbot: Oh btw, You placed a Named Entity Recognition pointer on "Trump", its certainly something but I don't know what it is
>User: What is &Tokyo ?
>Chatbot: indeed .
# >Chatbot: Oh btw, Uhm... "Tokyo" is a : Countries, cities, states


