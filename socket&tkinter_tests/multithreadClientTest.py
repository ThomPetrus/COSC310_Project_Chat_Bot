# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:08:50 2020

@author: iheal
"""

from multiprocessing.connection import Client
import tkinter as tk

def sendMessage():
    conn.send('Hello!')
    answer = conn.recv()
    label = tk.Label(frame, text = answer)
    label.pack()
    

address = ('localhost', 6000)

#The root frame
root = tk.Tk()
apps = []

#Create a canvas to set the size of a frame
canvas = tk.Canvas(root, height=700, width=700, bg="#263D42")

#Attach the canvas to the root frame                   
canvas.pack()

frame = tk.Frame(root, bg="white")
#Attach the new frame to the root frame and set its size relative to the parent frame 
#Sort of like CSS
frame.place(relwidth = 0.8, relheight = 0.8, relx = 0.1, rely = 0.1)

#Create a button
sendMessage = tk.Button(root, text="Send Message", 
                     padx=10, pady=5, fg="white", bg="#263D42", command = sendMessage)
sendMessage.pack()

with Client(address) as conn:
    root.mainloop()
    
