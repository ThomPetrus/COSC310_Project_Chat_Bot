import tkinter as tk
import socket

root = tk.Tk()

s = socket.socket()

s.bind(('localhost', 8888))

#s.listen()

#s.close()