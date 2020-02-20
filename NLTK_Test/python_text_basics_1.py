# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 20:34:30 2020
Python Text Basics - Udemy Course
COSC 310 - NLP
@author: tpvan
"""

"""
-------------------------------------------------------------------------------
Text Basics - Python 
-------------------------------------------------------------------------------
"""
# F String literals - formatting text 
person="Thom"
print(f"My name is {person}")

# List of tuples
library = [('Author', 'Topic', 'Pages'), ('Twain', 'Rafting', 601), ('Hamilton', 'Blalbasdvasdvasdvasda', 400)]

# Tuple unpacking & fstring literals
for author, topic, pages in library:
    print(f"{author} {topic} {pages}")
    
# Tuple unpacking and formatting w/ fstring literals
for author, topic, pages in library:
    print(f"{author:{10}} {topic:{30}} {pages:{10}}")

# Auto-align ( > )
for author, topic, pages in library:
    print(f"{author:{10}} {topic:{30}} {pages:>{10}}")
    
# Auto-align & padding w/ characters ( +> )
for author, topic, pages in library:
    print(f"{author:{10}} {topic:{30}} {pages:+>{10}}")    
    
# dates / strftime -- Ton of different directives - googleable - not super useful
from datetime import datetime

# i.e. %B is full month etc
today = datetime(year=2019, month=2, day=27)
print(f"{today:%B %d %Y}")

"""
-------------------------------------------------------------------------------
Files Basics - Python 
-------------------------------------------------------------------------------
"""
myfile = open("C:/Users/tpvan/OneDrive/Documents/UBCO/Second Year/COSC 310/Project/Udemy Course/test.txt")

print(myfile.read())
# note that the following wont print -- same as iostreams in java
print(myfile.read())
# can reset cursor :
myfile.seek(0)
print(myfile.read())

myfile.seek(0)
contents = myfile.read()

# Same as java -- close ur dang iostreams
myfile.close()

# read lines -- returns list
myfile = open("C:/Users/tpvan/OneDrive/Documents/UBCO/Second Year/COSC 310/Project/Udemy Course/test.txt")
content_as_list = myfile.readlines()

print(content_as_list)

# only 1st word
for line in content_as_list:
    print(line.split()[0])

myfile.close()

# Writing to files. -- NOTE the 'w', if you 'w+' or 'w' it'll overwrite
myfile = open("C:/Users/tpvan/OneDrive/Documents/UBCO/Second Year/COSC 310/Project/Udemy Course/test.txt", 'w')
myfile.write('whats up txt file')
myfile.close()

# 'a+' is append to -- if file does not exist, new one is created.
myfile = open("C:/Users/tpvan/OneDrive/Documents/UBCO/Second Year/COSC 310/Project/Udemy Course/test.txt", 'a+')
myfile.write('This was appended to the previous')
myfile.close()


"""
-------------------------------------------------------------------------------
File Basics - Just use this fr - Python 
-------------------------------------------------------------------------------
"""
# Using the context-manager, kind of like try-w-resources in java -- auto closes.
with open("C:/Users/tpvan/OneDrive/Documents/UBCO/Second Year/COSC 310/Project/Udemy Course/test.txt", 'w') as mynewfile:
    myvariable = mynewfile.readlines()
    
# Note it is possible to read pdfs with extractable text -- i.e. those created by a word processor but it not really relevant to us.
    


    