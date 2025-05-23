import json
import socket

from src.typings.general import ChatHistoryItem


class Server:
    def __init__(self, port, workers, host="localhost"):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        while True:
            try:
                self.socket.bind((self.host, self.port))
                break
            except OSError as e:
                self.port += 1
        self.socket.listen(workers + 2)
        self.log = {}
        self.status = {}

    async def start(self, folder, session):
        log_file = []
        print(111)
        client_socket, client_address = self.socket.accept()
        print(222)
        while True:
            data = client_socket.recv(1000000).decode()
            if data == "":
                self.stop(client_socket)
                break
            elif data.startswith("#[ERROR]"):
                status = int(data[-1])
                self.status[folder] = status
            else:
                try:
                    session.history = json.loads(data)
                    session.history = [ChatHistoryItem(**item) for item in session.history]
                    log_file.append({"role": "user", "content": data})
                    ret = await session.action()
                    if ret.content is None:
                        self.status[folder] = 3
                        self.send_message(client_socket, "### LLM ERROR EXIT ###")
                        break
                    else:
                        ret = ret.content
                    print("\n######\n")
                    print(ret)
                    log_file.append({"role": "agent", "content": ret})
                    print("sending message")
                    self.send_message(client_socket, ret)
                    print("message sent")
                except json.decoder.JSONDecodeError:
                    log_file.append({"role": "agent", "content": ""})
                    print("except sending")
                    self.send_message(client_socket, "")
                    print("except message sent")
        self.log[folder] = log_file

    def send_message(self, client_socket, message):
        client_socket.sendall(message.encode())

    def stop(self, client_socket):
        client_socket.close()
