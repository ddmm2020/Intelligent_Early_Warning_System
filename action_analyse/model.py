# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: ddmm
# @Date  : 2021/5/14
# @Desc  :

import torchvision
import torch
import numpy as np
import cv2
import math
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import transforms
from action_analyse.action import Action

loader = transforms.Compose([
    transforms.ToTensor()])
unloader = transforms.ToPILImage()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.__dict__["keypointrcnn_resnet50_fpn"](num_classes=2, pretrained=True).to(device)
model = model.eval()

action_names = ["未识别","走动", "站立", "坐", "掌舵", "玩手机", "瞭望", "打盹"]
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=60):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    if isinstance(text,int):
        text = action_names[text]
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


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


class PoseDetection:
    def __int__(self, model_cfg="keypointrcnn_resnet50_fpn", points_num=17):
        self.points_num = 17

    def detection_one_image(self, img, detect_threshold=0.8):
        if type(img) is np.ndarray:
            img = loader(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        prediction = model([img.to(device, torch.float)])
        keypoints = prediction[0]['keypoints'].cpu().detach().numpy()
        bboxs = prediction[0]['boxes'].cpu().detach().numpy()
        scores = prediction[0]['scores'].cpu().detach().numpy()
        keypoints_scores = prediction[0]['keypoints_scores'].cpu().detach().numpy()
        idx = np.where(scores > detect_threshold)
        keypoints = keypoints[idx]
        keypoints_scores = keypoints_scores[idx]
        bboxs = bboxs[idx]
        return keypoints, bboxs, keypoints_scores

    def plot_poses(self, img, skeletons, save_name='pose1.jpg'):
        EDGES = [(0, 14), (0, 13), (0, 4), (0, 1), (14, 16), (13, 15), (4, 10), (1, 7),
                 (10, 11), (7, 8), (11, 12), (8, 9), (4, 5), (1, 2), (5, 6), (2, 3)]
        NUM_EDGES = len(EDGES)

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
                  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        # cmap = matplotlib.cm.get_cmap('hsv')

        canvas = img.copy()

        for i in range(17):
            # rgba = np.array(cmap(1 - i / 17. - 1. / 34))
            # rgba[0:3] *= 255
            for j in range(len(skeletons)):
                cv2.circle(canvas, tuple(skeletons[j][i, 0:2].astype('int32')), 2, colors[i], thickness=-1)

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
                # canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        # canvas = unloader(canvas)
        cv2.imwrite(save_name, canvas)
        return canvas


if __name__ == '__main__':
    pass
    PoseD = PoseDetection()
    A = Action()
    # img = image_loader('./p2_00024.jpeg')
    # img_np = cv2.imread("./p2_00024.jpeg")
    # img=loader(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

    cap = cv2.VideoCapture("./test2.avi")
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    video_writer = cv2.VideoWriter('./final_pose.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, size)

    while cap.isOpened():
        ret, img_np = cap.read()
        img = loader(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        keypoints, bboxs,keypoints_scores= PoseD.detection_one_image(img)
        # img_np = PoseD.plot_poses(img_np, keypoints, save_name='test.jpg')
        # print(bboxs)
        for i in range(len(keypoints)):
            max_l2, action_name = A.get_action(keypoints[i], bboxs[i])
            if isinstance(action_name, int):
                action_name = action_names[action_name]
            print(max_l2, action_name)
            if max_l2 < 0.1:
                idx = np.argmax(keypoints_scores)
                x1, y1, x2, y2 = int(bboxs[i][0]), int(bboxs[i][1]), int(bboxs[i][2]), int(bboxs[i][3])
                x = int(x1 * 0.5 + x2 * 0.5)
                y = int(y1 * 0.5 + y2 * 0.5) - 20
                # print((x1 -10,y1-10),(x2+10,y2+10))
                # cv2.rectangle(img_np, (240, 0), (480, 375), (0, 255, 0), 2)
                half = 70
                cv2.line(img_np, (x, y-2), (x + half, y-2), (23,13,227), 8)
                cv2.line(img_np, (x-2, y), (x-2, y + half), (23,13,227), 8)
                cv2.line(img_np, (x + half * len(action_name), y + half), (x + half * len(action_name), y), (23,13,227),8)
                cv2.line(img_np, (x + half * len(action_name), y + half), (x + half * len(action_name) - half, y + half),
                         (23,13,227), 8)
                cv2.rectangle(img_np, (x, y), (x + half * len(action_name), y + half), (99,71,255), 1)
                img_np = cv2AddChineseText(img_np, action_name, (x, y), (127,255,0), 70)
                img_np = cv2AddChineseText(img_np, "L2:" + str(max_l2)[:5], (x, y-35), (227,23,13), 30)
        # cv2.imwrite("test.jpg", img_np)
        cv2.imshow("demo", img_np)
        # cv2.imwrite("./my_test21.png",background)

        video_writer.write(img_np)
        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            video_writer.release()
            break

    cap.release()
    # video_writer.release()
    cv2.destroyAllWindows()

    # img = img.cpu().mul(255).permute(1, 2, 0).byte().numpy()
    # PoseD.plot_poses(img_np, keypoints, save_name='test.jpg')
