# -*- coding: utf-8 -*-
# @File  : inference.py
# @Author: ddmm
# @Date  : 2021/3/21
# @Desc  :

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import os
import cv2
from visual import draw_mask
import time
import numpy as np
import json
from encoder.rle import generate_json
from connect.client import NetClient
from person.person import Person
from person import person


if __name__ == '__main__':
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # 使用GPU
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        model.cuda()

    cap = cv2.VideoCapture("./data/test2.avi")
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    video_writer = cv2.VideoWriter('./test.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    background_raw = cv2.imread("./data/background_69.jpeg")
    # print(background.shape)

    # netClient = NetClient("123.56.150.226", 10086)
    # netClient = NetClient("192.168.1.101", 8800)
    # netClient.tcpclient()

    # test = cv2.imread("./test.jpg")
    cnt = 0
    person_list = []
    color_id = 0
    while cap.isOpened():
        background = background_raw.copy()
        ret, frame = cap.read()
        blob = transform(frame)
        c, h, w = blob.shape
        input_x = blob.view(1, c, h, w)

        output = model(input_x.cuda())[0]

        boxes = output['boxes'].cpu().detach().numpy()
        scores = output['scores'].cpu().detach().numpy()
        labels = output['labels'].cpu().detach().numpy()
        masks = output['masks'].cpu().detach().numpy()

        threshold = 0.6

        for idx in range(len(masks)):
            if scores[idx] > threshold and labels[idx] == 1:
                mask,bbox = np.squeeze(masks[idx]), boxes[idx]
                mask[np.where(mask > 0.5)] = 1
                if person.match_person(mask,bbox,person_list):
                    person_list.append(Person(color_id, mask, bbox,0.35))
                    color_id = (color_id+ 1) % 10
        person.clear_person(person_list)
        background = draw_mask.replace_pic(background,frame,person_list)
        cv2.putText(background,"person: {}".format(len(person_list)),(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        cv2.imshow("demo", background)
        # cv2.imwrite("./my_test21.png",background)

        video_writer.write(background)
        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            video_writer.release()
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
