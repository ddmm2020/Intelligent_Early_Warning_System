# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: ddmm
# @Date  : 2021/5/11
# @Desc  :

import cv2
import numpy as np
import time
from config.config import Config

Config = Config()


TEST_PATH = "../demo/background_69.jpeg"


if __name__ == '__main__':
    background = cv2.imread(TEST_PATH)
    h,w,c = background.shape
    corrd = []
    x1 = Config.start_x
    # for x in range(x1,w,150):
    #     cv2.line(background,(x,0),(x,h),(0,0,255),5)
    # for y in range(0,h,240):
    #     cv2.line(background,(0,y),(w,y),(0,0,255),5)

    for x in range(x1,w,Config.step_x):
        for y in range(Config.start_y,h,Config.step_y):
            corrd.append([x,y])

    print(corrd)
    for idx,(i) in enumerate(corrd):
        cv2.putText(background,str(idx),(i[0],i[1]+50),color=(0,255,255),fontFace= cv2.FONT_HERSHEY_PLAIN,fontScale=3)
        cv2.circle(background,(i[0],i[1]),radius=2,color=(0,0,255))
    cv2.imwrite("./test.png", background)
# video_writer.release()
cv2.destroyAllWindows()