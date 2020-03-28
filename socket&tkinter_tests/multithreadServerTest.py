# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:59:03 2020

@author: iheal
"""

from multiprocessing.connection import Listener
from array import array

#Family is deduced to be 'AF_INET'
address = ('localhost', 6000)

#Wrapper for socket referred to by address
with Listener(address) as listener:
    #Returns a Connection object
    with listener.accept() as conn:
        print('connection accepted from', listener.last_accepted)
        
        while 1:
            conn.send('Wahoo')
            
            rst = conn.recv()
            
            if (rst == 'stop'):
                conn.send('stop')
                break
            print(rst)