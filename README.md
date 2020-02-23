# COSC310_Project_Chat_Bot
 COSC 310 Python Chat Bot
 
Project Documentation
Feb 4:
- Set up the repository and created a test file using the Python NLTK.
- Here's a link with all the howto's:
  https://www.nltk.org/howto/

Feb 17-19:
-Exploring other libraries and online course material on Udemy.
Updated notes on Spacy, Scikit-Learn, and NLTK. 
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