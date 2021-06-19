# -*- coding: utf-8 -*-
# @File  : client.py
# @Author: ddmm
# @Date  : 2021/3/23
# @Desc  :

import sys
import socket
import os

class NetClient(object):
    def __init__(self, host='localhost', port=9527):
        self.host = host
        self.port = port

        self.clientSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientSock.connect((self.host, self.port))

    def get_cmd(self):
        print("ip: {} port: {} connecting.......".format(self.host,self.port))
        # clientSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # clientSock.connect((self.host, self.port))

        print("sending.....")
        # sendDataLen = clientSock.sendto("liuweijian client link succeed!".encode(),(self.host,self.port))
        cmd = self.clientSock.recv(1024)
        print(cmd)
        print("client link succeed")
        # clientSock.close()
        os.system(cmd)



    def tcpclient(self):
        print("ip: {} port: {} connecting.......".format(self.host,self.port))
        clientSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientSock.connect((self.host, self.port))

        print("sending.....")
        sendDataLen = clientSock.sendto("liuweijian client link succeed!".encode(),(self.host,self.port))
        recvData = clientSock.recv(1024)
        print(recvData)
        print("client link succeed")
        clientSock.close()

    def send_json(self,json_info):
        # clientSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # clientSock.connect((self.host, self.port))

        self.clientSock.sendto(json_info.encode(),(self.host,self.port))
        # recvData = clientSock.recv(1024)
        # print("sendDataLen: ", sendDataLen)
        # print("recvData: ", recvData)

        # clientSock.close()



if __name__ == "__main__":
    netClient = NetClient("192.168.1.102",6666)
    netClient.tcpclient()
