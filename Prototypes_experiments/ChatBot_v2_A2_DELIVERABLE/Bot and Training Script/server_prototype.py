import socket

host=socket.gethostname()
port=(8000)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

def connect():
    s.bind((host,port))
    s.listen(2)
    print("Server listening")
    conn,addr=s.accept()
    print("Connected")
    send(conn)

def send(conn):
    while 1:
        data=input("Input data to send: ")
        encoded_data=data.encode('UTF-8')
        conn.send(encoded_data)
        
connect()