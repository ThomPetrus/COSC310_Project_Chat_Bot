import socket

c = socket.socket() #Creating a client socket
#For shorthand, s = server socket, c = client socket

c.connect(('localhost', 7777))

print(c.recv(1024).decode())