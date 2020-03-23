import socket

c = socket.socket() #Creating a client socket
#For shorthand, s = server socket, c = client socket

c.connect(('localhost', 9999))

while True:
    query = input('Please ask a question: ')
    c.send(bytes(query, encoding = 'utf8'))
    answer = c.recv(1024).decode()
    if (answer == 'stop'):
        break
    print(answer)