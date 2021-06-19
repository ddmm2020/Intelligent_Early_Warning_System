# -*- coding: utf-8 -*-
# @File  : test_cam.py
# @Author: ddmm
# @Date  : 2021/5/15
# @Desc  :

import cv2
import time

capture = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output{}.avi'.format(time.time()),fourcc, 30.0, (640,480))

while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame,1)
    cv2.imshow("video", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
out.release()
cv2.destroyAllWindows()