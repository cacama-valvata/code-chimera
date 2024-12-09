# Holds the server program and utility functions to be used by the client as well


import select
import socket
import pickle
import copy

import util

# Some socket guidance from:
# https://medium.com/hackernoon/socket-programming-in-python-client-server-and-peer-examples-a25c9782b584
# https://docs.python.org/3/library/socket.html
# https://stackoverflow.com/questions/5308080/python-socket-accept-nonblocking

def connectToServer(ip = "0.0.0.0", port = 52021):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    return sock

def sendInfo(info: dict, socket: socket.socket):
    msg = pickle.dumps(info)
    msgLength = len(msg)

    msgLength = msgLength.to_bytes(2, byteorder="big", signed=False)
    socket.sendall(msgLength)
    socket.sendall(msg)

def sendToAllClients(info: dict, allSockets, excludeIndex = None):
    for i, client in enumerate(allSockets):
        if i == 0 or i == excludeIndex:
            continue
        sendInfo(info, socket)

def recvMsg(sock: socket.socket, buffer: bytearray):
    headerLen = 2
    try:
        received = sock.recv(4096)
    except socket.timeout: 
        return "EOF"

    if not received:
        return "EOF"

    buffer.extend(received)

    result = []
    while True:
        msgLength = int.from_bytes(buffer[0:headerLen], byteorder="big", signed=False)
        if len(buffer) >= headerLen+msgLength:
            data = pickle.loads(buffer[headerLen:headerLen+msgLength])
            for i in range(headerLen+msgLength):
                buffer.pop(0)
            result.append(data)
        else:
            break

    if len(result) > 0:
        return result
        
    

def updateClientStates(state, allSockets, excludeIndex = None):
    for i, client in enumerate(allSockets):
        if i == 0 or i == excludeIndex:
            continue

        clientState = copy.deepcopy(state)
        removeClientFromState(clientState, i, ["health"])
        sendInfo(clientState, client)

    toRemove = []
    for key, value in state.items():
        if "fired" in key:
            toRemove.append(key)

    for key in toRemove:
        state.pop(key)

    print("updated clients state")

def updateState(state, allSockets, data, i):
    for key, value in data.items():
        if key[0].isdecimal():
            state[key] = value
        else:
            stateKey = str(i) + key
            state[stateKey] = value
        
def removeClientFromState(state, i, skips = []):    
    keys = list(state.keys())
    for key in keys:
        if key.startswith(str(i)) and not key[1:] in skips:
            state.pop(key)

def runServer():
    addr = ("", 52021)
    if socket.has_dualstack_ipv6():
        serv = socket.create_server(addr, family=socket.AF_INET6, dualstack_ipv6=True)
    else:
        serv = socket.create_server(addr)

    serv.listen(5)
    print(f"Server started on port {addr[1]}")

    state = dict()

    maybeReadables = [serv]
    addresses = ["0.0.0.0"]
    buffers = [bytearray()]
    while True:
        updateClientStates(state, maybeReadables)
        readables, _, _ = select.select(maybeReadables, [], [], 60)
        for readable in readables:
            i = maybeReadables.index(readable)
            if readable is serv:
                client, clientAddr = serv.accept()
                updateClientStates(state, [client])
                maybeReadables.append(client)
                addresses.append(clientAddr)
                buffers.append(bytearray())
                print(f"{clientAddr[0]} connected.")
            else:
                result = recvMsg(readable, buffers[i])
                print(result)
                # Disconnect on EOF
                if result == "EOF":
                    readable.close()
                    print(f"{addresses[i][0]} disconnected.")
                    addresses.pop(i)
                    maybeReadables.pop(i)
                    buffers.pop(i)
                    removeClientFromState(state, i)
                elif isinstance(result, list):
                    for data in result:
                        updateState(state, maybeReadables, data, i)
                        updateClientStates(state, maybeReadables)


if __name__ == '__main__':
    runServer()