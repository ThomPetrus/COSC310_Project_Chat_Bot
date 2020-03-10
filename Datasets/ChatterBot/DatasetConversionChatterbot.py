import pandas as pd
import csv

f = open("literature.txt", "r")

counter = 0
input = ""
output = ""

#open original file
with open('literature.csv', 'w', newline = '\n') as lit:
    
    #write to the csv file
    writer = csv.writer(lit, delimiter = '\t')

    #for all the lines in the text file
    for line in f:

        counter = counter + 1
        
        #line is an input
        if("- -" in line):
            input = line.strip()[4:]
 
        #line is an output
        else:
            output = line.strip()[2:]

        if(counter % 2 == 0):
            writer.writerow(["literature", input, output])
