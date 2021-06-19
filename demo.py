# -*- coding: utf-8 -*-
# @File  : demo.py
# @Author: ddmm
# @Date  : 2021/5/16
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
from action_analyse.action import Action
from action_analyse.model import PoseDetection
import random
import datetime


if __name__ == '__main__':
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # 使用GPU
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        model.cuda()

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # video_writer = cv2.VideoWriter('./test_421.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    background_raw = cv2.imread("./demo/background_HP.jpg")
    print(background_raw.shape)

    netClient = NetClient("123.56.150.226", 10086)
    netClient.tcpclient()

    # test = cv2.imread("./test.jpg")
    cnt = 0
    person_list = []
    color_id = 0

    Pose = PoseDetection()
    A = Action()

    while cap.isOpened():
        background = background_raw.copy()
        ret, frame = cap.read()

        if random.random() < 0.2:
            keypoints, bboxs_kp, _ = Pose.detection_one_image(frame)
            actions_list = [A.get_action(k, b) for k, b in zip(keypoints, bboxs_kp)]
            # print(actions_list)
            person.action_match_person(bboxs_kp, actions_list, person_list)

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
                mask, bbox = np.squeeze(masks[idx]), boxes[idx]
                mask[np.where(mask > threshold)] = 1
                if person.match_person(mask, bbox, person_list):
                    person_list.append(Person(color_id, mask, bbox, 0.35))
                    color_id = (color_id + 1) % 10
        person.clear_person(person_list)

        json_info = draw_mask.gen_json_file_v3(frame, person_list, use_img=True)

        cnt += 1
        json_send = json.dumps(json_info)
        if cnt % 5 != 0:
             continue
        # c = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
        now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))
        file_name = "json_file/"+now+".json"
        with open(file_name, 'w') as f:
            f.write(json_send)
        netClient.send_json(json_send)

    cap.release()
    # video_writer.release()
    cv2.destroyAllWindows()
