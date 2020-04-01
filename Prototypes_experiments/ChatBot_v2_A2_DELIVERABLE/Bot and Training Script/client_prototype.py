#import socket
from multiprocessing.connection import Client

#c = socket.socket() #Creating a client socket
#For shorthand, s = server socket, c = client socket

#IP of the localhost, port number is arbitrary but should be out of the low 1000s
#c.connect(('localhost', 9999))
remote = Client(("", 25000), authkey=b"selam")

#Repeatedly query the server 
while True:
    query = input('Please ask a question: ')
    
    #Sends a byte object with utf08 encoding
    remote.send_bytes(query.encode('utf8'))
    
    listener = remote.accept()
    
    #Decodes a byte object into a string
    answer = listener.rec_bytes()
    
    if (answer == 'stop'):
        break
    
    print(answer)