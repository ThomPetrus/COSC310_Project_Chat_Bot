# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:43:42 2020

@author: iheal
"""

import socket

c = socket.socket() #Creating a client socket
#For shorthand, s = server socket, c = client socket

c.connect(('localhost', 9999))

while True:
    query = input("Ask a question: ")
    c.send(bytes(query, 'utf-8'))
    print(c.recv(1024).decode())