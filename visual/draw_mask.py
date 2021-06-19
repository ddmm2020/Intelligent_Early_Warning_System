# -*- coding: utf-8 -*-
# @File  : draw_mask.py
# @Author: ddmm
# @Date  : 2021/3/22
# @Desc  :

import numpy as np
import cv2
from encoder.rle import rle_encoder
import time
import json
from config.config import Config

colors = [
    (255, 0, 255), (255, 222, 173), (255, 114, 86), (238, 130, 238), (255, 127, 36), (250, 128, 114), (255, 106, 106),
    (125.0, 0.0, 255.0), (125.0, 0.0, 0.0), (255, 255, 0), (0.0, 0.0, 125.0), (255, 187, 255), (0, 255, 215),
    (0.0, 255.0, 255.0), (125.0, 255.0, 0.0), (255.0, 0.0, 0.0), (0.0, 0.0, 255.0)]

coco_names = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
              6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
              11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
              16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
              22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
              28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
              35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
              40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
              44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
              70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
              77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
              82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
              88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def add_color(frame, mask, color_id):
    frame[np.where(mask == 1)] = colors[color_id]
    return frame


def show_person(frame, person_list):
    color_mask = np.zeros_like(frame, dtype=np.uint8)
    for p in person_list:
        if p.confidence < 0.5:
            continue
        cv2.putText(frame, "person:{}".format(p.confidence), (np.int32(p.bbox[0]), np.int32(p.bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.rectangle(frame, (np.int32(p.bbox[0]), np.int32(p.bbox[1])),
                      (np.int32(p.bbox[2]), np.int32(p.bbox[3])), (0, 255, 0), 1, 8, 0)
        color_mask = add_color(color_mask, p.mask, p.color_id)
    frame = cv2.addWeighted(frame, 0.5, color_mask, 0.5, 0)
    return frame

def replace_pic(background,frame,person_list):
    for p in person_list:
        if p.confidence < 0.5:
            continue
        # cv2.putText(frame, "person:{}".format(p.confidence), (np.int32(p.bbox[0]), np.int32(p.bbox[1]) - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        x1 = int(p.bbox[0])
        y1 = int(p.bbox[1])
        x2 = int(p.bbox[2])
        y2 = int(p.bbox[3])
        background[y1:y2,x1:x2] = frame[y1:y2,x1:x2]
    return background

def show_person_ans_gen_json(frame, person_list, gen_json=False):
    send_info = {
        "time": time.time(),
    }
    person_mask = []
    colors_id = []
    person_bbox = []
    person_conf = []
    color_mask = np.zeros_like(frame, dtype=np.uint8)

    for p in person_list:
        if p.confidence < 0.5:
            continue
        cv2.putText(frame, "person:{}".format(p.confidence), (np.int32(p.bbox[0]), np.int32(p.bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        x1 = int(p.bbox[0])
        y1 = int(p.bbox[1])
        x2 = int(p.bbox[2])
        y2 = int(p.bbox[3])
        cv2.rectangle(frame, (x1, y1),
                      (x2, y2), (0, 255, 0), 1, 8, 0)
        frame = add_color(frame, p.mask, p.color_id)
        if gen_json:
            person_mask.append(rle_encoder(p.mask))
            person_bbox.append([x1, y1, x2, y2])
            colors_id.append(p.color_id)
            person_conf.append(p.confidence)

    if gen_json:
        send_info["cnt"] = len(person_mask)
        send_info["person"] = person_mask
        send_info["colors_id"] = colors_id
        send_info["bbox"] = person_bbox
        send_info["scores"] = person_conf
    json_send = json.dumps(send_info)
    # frame = cv2.addWeighted(frame, 0.5, color_mask, 0.5, 0)
    return frame, json_send


def gen_json_file_v3(frame, person_list, use_img=False):
    send_info = {}
    person_mask = []
    person_bbox = []
    colors_id = []
    person_conf = []
    person_imgs = []
    person_actions = []
    for p in person_list:
        if p.confidence < 0.5:
            continue
        x1 = int(p.bbox[0])
        y1 = int(p.bbox[1])
        x2 = int(p.bbox[2])
        y2 = int(p.bbox[3])

        # person_mask.append(rle_encoder(p.mask))
        person_mask.append(rle_encoder(p.mask[y1:y2,x1:x2]))
        person_bbox.append([x1, y1, x2, y2])
        colors_id.append(p.color_id)
        person_conf.append(p.confidence)
        person_actions.append(p.action)
        if use_img:
            person_imgs.append(frame[y1:y2,x1:x2].astype(np.int8).tolist())
            # print(x1, y1, x2, y2)
            # cv2.imwrite("./sub_img.jpg", frame[y1:y2,x1:x2])

    send_info["cnt"] = len(person_mask)
    send_info["boat"] = Config.boat_num
    send_info["time"] = time.time()
    send_info["mask"] = person_mask
    send_info["colors_id"] = colors_id
    send_info["bbox"] = person_bbox
    send_info["scores"] = person_conf
    send_info["action"] = person_actions
    if use_img:
        send_info["img"] = person_imgs
    return send_info


def mask_only(frame, mask):
    color_mask = np.zeros_like(frame, dtype=np.uint8)
    point = cv2.split(color_mask)
    # print(np.sum(np.where(mask == 1)))
    mask = np.squeeze(mask > 0.5)
    point[2][mask == 1], point[1][mask == 1], point[0][mask == 1] = colors[0]
    color_mask = cv2.merge(point)
    # back_ground = np.zeros_like(frame)
    result = cv2.addWeighted(frame, 0.5, color_mask, 0.5, 0)
    return result


def draw(frame, boxes, scores, labels, masks):
    index = 0
    color_mask = np.zeros_like(frame, dtype=np.uint8)
    point = cv2.split(color_mask)
    send_masks = []
    bboxs = []
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        if idx >= 5:
            break
        if scores[idx] > 0.5 and labels[idx] == 1:
            bbox = boxes[index].tolist()
            bbox = list(map(int, bbox))
            cv2.rectangle(frame, (np.int32(x1), np.int32(y1)),
                          (np.int32(x2), np.int32(y2)), (0, 255, 0), 3, 8, 0)
            mask = np.squeeze(masks[index] > 0.5)
            send_masks.append(mask)
            bboxs.append(bbox)

            np.random.randint(0, 256)
            point[2][mask == 1], point[1][mask == 1], point[0][mask == 1] = colors[idx]

            label_id = labels[index]
            label_txt = coco_names[label_id]
            cv2.putText(frame, label_txt, (np.int32(x1), np.int32(y1 - 5)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 1)
        index += 1

    color_mask = cv2.merge(point)
    # back_ground = np.zeros_like(frame)
    result = cv2.addWeighted(frame, 0.5, color_mask, 0.5, 0)
    # cv2.imwrite("./test_mask67.png",result)
    return result, send_masks, bboxs
