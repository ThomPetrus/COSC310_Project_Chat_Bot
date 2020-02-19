# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:00:38 2020

Session 2 - NLTK - Udemy
@author: tpvan
"""

# Basic Expressions --

# in keyword -- like 'contains()'
if("408-555-1234" in "Her phone is 408-555-12345"):
    print("yep");
    
"""
Using patterns - w/ regular expressions
Can use regular strings as well as patterns

i.e.
find 3 digits, then -, then 3 digits, then - etc. :
r"\d{3}-\d{3}-\d{4}"


\d - digit, \w - alphanumeric, \s - whitespace,
\D - non-digit, \W - non-alphanumeric (punctuation etc)

+ one or more, {3} 3 times, {2,5} 2 - 5 times
{3, } 3 or more, * 0 or more,  ? once or more

note the import is needed.
"""
import re

text = "The phone number of the agent is 408-555-1234."
# note the following are equivalent.
pattern = r"\d\d\d-\d\d\d-\d\d\d\d"
pattern =  r"\d{3}-\d{3}-\d{4}"

# first match / exists
first_match = re.search(pattern, text)

# all matches
matches = re.findall(pattern, text)
print(matches)

# grouping matches - i.e. only area code
pattern =  r"(\d{3})-(\d{3})-(\d{4})"
mymatch = re.search(pattern, text)
print(mymatch.group(1))

# --- wildcard matches 
# one or the other
text = "The cat in the hat sat flat on the mat5"
pattern = r"cat|fat"
match = re.search(pattern, text)
print(match)

# part of the string - . replaces missing letter
matches = re.findall(r".at", text)
print(matches)

# \d$ -ends with number, ^ - starts with. OR its exclude
pattern = r"[^\d]+"
text = "12 bunch of numbers 14 and text 13 yeah!"
matches = re.findall(pattern, text)
print(matches)





#can be used to REMOVE PUNCTUATION





text = "This is a String!, with a, bunch of , stupid punctuation!!"
pattern = r"[^!.?, ]+"
matches = re.findall(pattern, text)
print(matches)