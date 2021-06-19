# -*- coding: utf-8 -*-
# @File  : remote_control.py
# @Author: ddmm
# @Date  : 2021/5/16
# @Desc  :
import socket
import os

if __name__ == '__main__':
    # 创建一个socket对象
    host = "192.168.1.102"
    port = 6666
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))  # 绑定套接字到IP与端口
    sock.listen(5)  # 开始监听连接
    print("waiting for call .....")

    while True:
        # 不断接受客户端的连接请求
        clientSock, (remoteHost, remotePort) = sock.accept()
        print("[%s:%s] connect" % (remoteHost, remotePort))  # 接收客户端的ip, port

        # 接收传来的数据，并发送给对方数据
        cmd = clientSock.recv(1024).decode()
        sendDataLen = clientSock.send("Hello Client,I am server".encode())
        print(cmd)
        os.system(cmd)
        # msg = "发送成功"
        # 发送数据，需要进行编码
        # clientSock.send(msg.encode("utf-8"))