import socket

s = socket.socket() #Create a server socket, TCP by default
#For shorthand, s = server socket, c = client socket
print('Socket Created')

s.bind(('localhost', 8888))#Pass the IP address of the machine and the port number
#Don't use port numbers that are in the low 1000s, those are usually busy
s.listen(3) #Can handle up to 3 connections

print('waiting for connections')

while True:
    c, addr = s.accept() #Returns client socket and address, accepts connection

    name = c.recv(1024).decode()

    print("Connected with " + str(addr) + " " + str(name))
    
    c.send(bytes('Welcome to Telusko', 'utf-8'))
    
    c.close()

