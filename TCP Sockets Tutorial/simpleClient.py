# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:19:07 2020

@author: iheal
"""

import socket

s = socket.socket()
host = socket.gethostname()
port = 12345

s.connect((host, port)) #initiate TCP server connection
print (s.recv(1024))
s.close()