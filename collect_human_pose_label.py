# -*- coding: utf-8 -*-
# @File  : collect_human_pose_label.py
# @Author: ddmm
# @Date  : 2021/5/11
# @Desc  :

import torchvision
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import math
import torch
import PIL
import pandas as pd
from PIL import Image
from action_analyse.action import Action
import os
from torchvision.transforms import transforms
from action_analyse.action import Action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()
A = Action(data_file="./data/test_clear_v2.csv")


def image_loader(image_name):
    """
    读取图像
    :param image_name: 图像名称
    :return:
    """
    image = Image.open(image_name).convert('RGB')
    image = loader(image)
    return image


def map_coco_to_personlab(keypoints):
    permute = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    return keypoints[:, permute, :]


def plot_poses(img, skeletons, save_name='pose1.jpg'):
    EDGES = [(0, 14), (0, 13), (0, 4), (0, 1), (14, 16), (13, 15), (4, 10), (1, 7),
             (10, 11), (7, 8), (11, 12), (8, 9), (4, 5), (1, 2), (5, 6), (2, 3)]
    NUM_EDGES = len(EDGES)

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')

    canvas = img.copy()

    for i in range(17):
        rgba = np.array(cmap(1 - i / 17. - 1. / 34))
        rgba[0:3] *= 255
        for j in range(len(skeletons)):
            cv2.circle(canvas, tuple(skeletons[j][i, 0:2].astype('int32')), 2, colors[i], thickness=-1)

    to_plot = cv2.addWeighted(img, 0.3, canvas, 0.7, 0)
    fig = matplotlib.pyplot.gcf()

    stickwidth = 2

    skeletons = map_coco_to_personlab(skeletons)
    for i in range(NUM_EDGES):
        for j in range(len(skeletons)):
            edge = EDGES[i]
            if skeletons[j][edge[0], 2] == 0 or skeletons[j][edge[1], 2] == 0:
                continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        # cv2.imwrite("./kp_data/" + save_name, canvas)


def pose_detection(img, model,name,detect_threshold=0.8, keypoint_score_threshold=2,
                   vis=False):
    """
    人体17个部分的关键点检测
    :param img: 输入RGB图像，建议采用PILLOW 读取图像，如果是opencv图像需要转换颜色域
    :param model_cfg: 模型名称
    :param detect_threshold: 人员的confidence
    :param keypoint_score_threshold: 人体point的confidence
    :param vis: 可视化
    :return:
    """
    points_num = 17

    try:
        prediction = model([img.to(device, torch.float)])
    except AttributeError:
        raise Exception(f"img should be tensor! But got {type(img)}")


    keypoints = prediction[0]['keypoints'].cpu().detach().numpy()
    bboxs = prediction[0]['boxes'].cpu().detach().numpy()
    scores = prediction[0]['scores'].cpu().detach().numpy()
    keypoints_scores = prediction[0]['keypoints_scores'].cpu().detach().numpy()
    idx = np.where(scores > detect_threshold)
    keypoints = keypoints[idx]
    keypoints_scores = keypoints_scores[idx]
    bboxs = bboxs[idx]
    file_names = []

    csv_info = []
    if vis:
        for j in range(keypoints.shape[0]):
            for num in range(points_num):
                if keypoints_scores[j][num] < keypoint_score_threshold:
                    keypoints[j][num] = [0, 0, 0]
        img = img.cpu().mul(255).permute(1, 2, 0).byte().numpy()
        # print(keypoints.shape)
        for i in range(keypoints.shape[0]):
            file_names.append( name.replace(".","_{}.".format(i)) )
            plot_poses(img, np.expand_dims(keypoints[i],axis=0), save_name=name.replace(".","_{}.".format(i)))
            csv_info.append([name.replace(".","_{}.".format(i)),bboxs[i],keypoints[i],keypoints_scores[i]])
            # csv_info.append([name.replace(".","_{}.".format(i)),-1,])
    return keypoints, keypoints_scores,csv_info


if __name__ == '__main__':
    base_path = "./kp_image"
    imgs = os.listdir(base_path)

    model = torchvision.models.detection.__dict__['keypointrcnn_resnet50_fpn'](num_classes=2,
                                                             pretrained=True)
    model.to(device)
    model.eval()
    cnt = 0
    for img_name in imgs:
        cnt += 1
        img = image_loader(os.path.join(base_path,img_name))
        kp,kps,csv_info = pose_detection(img,model,img_name, detect_threshold=0.55, vis=True)

        df = pd.DataFrame(csv_info, columns=['file_name','bbox','keypoint','keypoint_score'])
        # df = pd.DataFrame(csv_info, columns=['file_name','action'])
        if cnt ==0:
            df.to_csv("test_v2.csv", index=False, mode='a+', header=True)
        else:
            df.to_csv("test_v2.csv", index=False, mode='a+', header=False)


