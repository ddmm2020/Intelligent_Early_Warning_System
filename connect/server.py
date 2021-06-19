# -*- coding: utf-8 -*-
# @File  : server.py
# @Author: ddmm
# @Date  : 2021/3/23
# @Desc  :

import sys
import socket


class NetServer(object):
    def __init__(self, host='localhost', port=9527):
        self.host = host
        self.port = port

    def tcpServer(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((self.host, self.port))  # 绑定套接字到IP与端口
        sock.listen(5)  # 开始监听连接
        print("waiting for call .....")

        while True:
            # 不断接受客户端的连接请求
            clientSock, (remoteHost, remotePort) = sock.accept()
            print("[%s:%s] connect" % (remoteHost, remotePort))  # 接收客户端的ip, port

            # 接收传来的数据，并发送给对方数据
            revcData = clientSock.recv(1024)
            sendDataLen = clientSock.send("Hello Client,I am server".encode())
            print("revcData: ", revcData)
            print("sendDataLen: ", sendDataLen)

            # 传输完毕
            clientSock.close()


if __name__ == "__main__":
    netServer = NetServer()
    netServer.tcpServer()
