# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:19:07 2020

@author: iheal
"""
"""
import socket

s = socket.socket()
host = socket.gethostname()
port = 12345

s.connect((host, port)) #initiate TCP server connection
print (s.recv(1024))
s.close()
"""
import socket

c = socket.socket() #Creating a client socket

name = input("Enter your name: ")
c.send(bytes(name, 'utf-8'))

c.connect(('localhost', 9999))

print(c.recv(1024).decode())