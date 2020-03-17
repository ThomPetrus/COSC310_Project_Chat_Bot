import socket

c = socket.socket() #Creating a client socket
#For shorthand, s = server socket, c = client socket

c.connect(('localhost', 8888))

name = input("Enter Your Name: ")

c.send(bytes(name, 'utf-8'))

print(c.recv(1024).decode())