# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:11:38 2020

@author: iheal
"""

import socket #import socket module

s = socket.socket() #create a socket object
host = socket.gethostname() #get the local machine name
port = 12345 #reserve a port
s.bind((host, port)) 

s.listen(5)
while True:
    c, addr = s.accept() #establish connection with client
    print ('Got connection from ', addr)
    c.send('Thank you for connecting!')
    s.close() #close the connection