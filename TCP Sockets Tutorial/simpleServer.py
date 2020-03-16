# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:11:38 2020

@author: iheal
"""
"""
import socket #import socket module

s = socket.socket() #create a socket object
host = socket.gethostname() #get the local machine name
port = 12345 #reserve a port
s.bind((host, port)) 

s.listen(5)                 # Now wait for client connection.
while True:
   c, addr = s.accept()     # Establish connection with client.
   print ('Got connection from', addr)
   c.send('Thank you for connecting')
   c.close()                # Close the connection
s.close()
"""

import socket

s = socket.socket() #Create a server socket
#By default it will be TCP 
print('Socket Created')

s.bind(('localhost', 9999))

s.listen(3) #CAn handle up to 3 connections
print('waiting for connections')

while True:
    c, addr = s.accept() #Returns client socket and address
    
    name = c.recv(1024).decode()
    
    print('Connected with ', addr, name)
    
    c.send(bytes('Welcome to the server!', 'utf-8'))
    
    c.close()

