import socket

HOST = socket.gethostname()  # Standard loopback interface address (localhost)
# Only processes on the host will be able to connect to the server
PORT = 12345       # Port to listen on (non-privileged ports are > 1023)

'''
socket.socket() creates a socket object that supports the context manager type.
This allows you to use the socket object without calling close().
The arguments specify the address family and socket type.

'''
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#AF_INET is the IPv4 internet address family
#SOCK_STREAM is the socket type for TCP
    s.bind((HOST, PORT)) # Associates socket with specified network interface (localhost) and port number (65432)
    s.listen() #Wait to accept() connections
    conn, addr = s.accept() #Blocks and waits for incoming connection
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            if not data:
                break
            print(data)