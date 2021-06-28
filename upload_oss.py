# -*- coding: utf-8 -*-
# @File  : upload_oss.py
# @Author: ddmm
# @Date  : 2021/5/16
# @Desc  :


import argparse
import os
import zipfile
import oss2
import time
import paramiko

parser = argparse.ArgumentParser(description = 'Test for connect')
parser.add_argument('--file', '-n',default = "demo.py",help='执行文件')
parser.add_argument('--jsondir', '-jd',default = "./json_file/",help='视觉信息存储文件')
parser.add_argument('--start_time', '-ss',default = "2021_05_16_21_56_01", help ='开始时间')
parser.add_argument('--end_time', '-t',default="2021_05_16_21_56_27",help ='结束时间')
parser.add_argument('--show', '-sh',default = "img",help='数据质量')
args = parser.parse_args()

def file2zip(zip_file_name: str, file_names: list):
    """ 将多个文件夹中文件压缩存储为zip

    :param zip_file_name:   /root/Document/test.zip
    :param file_names:      ['/root/user/doc/test.txt', ...]
    :return:
    """
    # 读取写入方式 ZipFile requires mode 'r', 'w', 'x', or 'a'
    # 压缩方式  ZIP_STORED： 存储； ZIP_DEFLATED： 压缩存储
    with zipfile.ZipFile(zip_file_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in file_names:
            parent_path, name = os.path.split(fn)
            # print(parent_path,name)

            # zipfile 内置提供的将文件压缩存储在.zip文件中， arcname即zip文件中存入文件的名称
            # 给予的归档名为 arcname (默认情况下将与 filename 一致，但是不带驱动器盘符并会移除开头的路径分隔符)
            zf.write(fn, arcname=name)

            # 等价于以下两行代码
            # 切换目录， 直接将文件写入。不切换目录，则会在压缩文件中创建文件的整个路径
            # os.chdir(parent_path)
            # zf.write(name)


def sftp_upload_file(host, user, password, server_path, local_path, timeout=10):
    """
    上传文件，注意：不支持文件夹
    :param host: 主机名
    :param user: 用户名
    :param password: 密码
    :param server_path: 远程路径，比如：/home/sdn/tmp.txt
    :param local_path: 本地路径，比如：D:/text.txt
    :param timeout: 超时时间(默认)，必须是int类型
    :return: bool
    """
    try:
        t = paramiko.Transport((host, 22))
        t.banner_timeout = timeout
        t.connect(username=user, password=password)
        sftp = paramiko.SFTPClient.from_transport(t)
        sftp.put(local_path, server_path)
        t.close()
        return True
    except Exception as e:
        print(e)
        return False

def upload_oss_file(key):
    endpoint = 'oss-cn-beijing.aliyuncs.com'

    auth = oss2.Auth('L****Ff**Konug**********', 'R****3D0nh**MJ8iPF4c**********')
    bucket = oss2.Bucket(auth, endpoint, 'edu-squash')
    current_fold = time.strftime('%Y-%m-%d', time.localtime())
    current_file_path = "69/"+key
    file_path =  key
    # 上传
    # remote = "https://" + bucketName + "." + endpoint + "/" + fileName;
    url = "https://" + "edu-squash" + "." + endpoint + "/" + current_file_path

    bucket.put_object_from_file(current_file_path, file_path)
    return url

if __name__ == '__main__':
    start_time = args.jsondir + args.start_time + ".json"
    end_time = args.jsondir + args.end_time + ".json"
    file_list = [args.jsondir + i for i in os.listdir(args.jsondir)]
    file_start_idx = file_list.index(start_time)
    file_end_idx = file_list.index(end_time)
    zip_name = "test.zip"
    file2zip(zip_name,file_list[file_start_idx:file_end_idx])
    url = upload_oss_file(zip_name)
    print(url)
