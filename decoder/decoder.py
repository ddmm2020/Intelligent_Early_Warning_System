# -*- coding: utf-8 -*-
# @File  : decoder.py
# @Author: ddmm
# @Date  : 2021/3/23
# @Desc  :

import cv2
import numpy as np
import json
from visual import draw_mask
from connect.client import NetClient
from action_analyse.model import cv2AddChineseText


def rle2mask(rle_code):
    # print(rle_code[0])
    # key_value_pair = rle_code.split(',')

    for key_value in rle_code:
        mask = np.zeros((10,))
        key_value = key_value.split(',')
        for kv in key_value[:-1]:
            kv = kv.split(':')
            key, value = int(kv[0]), int(kv[1])
            tmp = np.zeros((value,)) if key == 0 else np.ones((value,))
            mask = np.append(mask, tmp)
        mask = mask[10:].reshape(720, 1280)

    background = cv2.imread("../demo/background_69.jpeg")

    frame = draw_mask.mask_only(background, mask)
    cv2.imwrite("./mask_testttt.jpg", frame)


def rle2mask_test(rle_code, bbox):
    # print(rle_code[0])
    # key_value_pair = rle_code.split(',')

    for key_value in rle_code:
        mask = np.zeros((10,))
        key_value = key_value.split(',')
        for kv in key_value[:-1]:
            kv = kv.split(':')
            key, value = int(kv[0]), int(kv[1])
            tmp = np.zeros((value,)) if key == 0 else np.ones((value,))
            mask = np.append(mask, tmp)
        mask = mask[10:].reshape(720, 1280)

    background = cv2.imread("../demo/background_69.jpeg")
    frame = draw_mask.mask_only(background, mask)
    for b in bbox:
        cv2.rectangle(frame, (np.int32(b[0]), np.int32(b[1])),
                      (np.int32(b[2]), np.int32(b[3])), (0, 255, 0), 1, 8, 0)
    cv2.imwrite("./mask_testttt.jpg", frame)

def add_one_person(background,box,rle_code,color,score,action_name,img = None ,use_img = False):
    x1, y1, x2, y2 = box[0],box[1],box[2],box[3]


    img = np.stack(img)
    h,w,c = img.shape

    mask = np.zeros((10,))
    key_value = rle_code.split(',')
    for kv in key_value[:-1]:
        kv = kv.split(':')
        key, value = int(kv[0]), int(kv[1])
        tmp = np.zeros((value,)) if key == 0 else np.ones((value,))
        mask = np.append(mask, tmp)
    mask = mask[10:].reshape(h, w)
    # print(mask.shape)
    mask_full = np.zeros((background.shape[0],background.shape[1]),dtype=np.int8)
    mask_full[y1:y2,x1:x2] = mask
    if use_img :
        background[y1: y2, x1: x2] = img
        frame = background
    frame = draw_mask.mask_only(background, mask_full)
    print(action_name)
    frame = cv2AddChineseText(frame, action_name, (x1, y1), (255, 0, 0), 60)

    cv2.rectangle(frame, (np.int32(box[0]), np.int32(box[1])),
                  (np.int32(box[2]), np.int32(box[3])), (0, 255, 0), 1, 8, 0)
    return frame

if __name__ == '__main__':
    f = open('D:\\CVWORK\\PersonMask\\json_file\\2021_05_17_10_48_34.json', 'r')
    content = f.read()
    json_info = json.loads(content)
    f.close()

    # netClient = NetClient("192.168.1.104", 10086)
    # netClient.tcpclient()
    # netClient.send_json(json_info)

    for k, v in json_info.items():
        print("key: {}".format(k))

    person_cnt = json_info['cnt']
    person_bbox = json_info['bbox']
    person_mask = json_info['mask']
    person_idx = json_info['colors_id']
    person_scores = json_info['scores']
    person_imgs = json_info['img']
    person_actions = json_info['action']

    background = cv2.imread("../demo/background_69.jpeg")

    for idx in range(person_cnt):
        bbox,mask,color,score,person_img,person_action = person_bbox[idx], person_mask[idx], person_idx[idx], person_scores[idx], person_imgs[idx],person_actions[idx]
        background = add_one_person(background,bbox,mask,color,score,person_action,person_img)
    cv2.imwrite("test.jpg",background)


    # for i in range(mask_cnt):
    #     key = "person"
    #     mask = json_info[key]
    #     # rle2mask(mask)
    #     rle2mask_test(mask, bbox)
    # print(type(a))
    # print(a)

    # mask = json.load()
    # print(mask)
