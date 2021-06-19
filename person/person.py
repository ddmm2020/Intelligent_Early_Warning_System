# -*- coding: utf-8 -*-
# @File  : person.py
# @Author: ddmm
# @Date  : 2021/4/9
# @Desc  :

import numpy as np
import collections
from action_analyse.action import Action

Action = Action(data_file="./data/test_clear_v2.csv")

def match_person(mask,bbox,person_list):
    for p in person_list:
        if p.mask_iou(mask) > p.threshold:
            p.confidence = round(np.random.randint(80,100)/100,2)
            p.mask = mask
            p.bbox = bbox
            return False
    return True

def action_match_person(bbox_list,action_list,person_list,max_l2_dis = 10):
    if len(person_list) < 1:
        return
    for i,(l2_dis,action_name) in enumerate(action_list):
        if l2_dis > max_l2_dis:
            continue
        if i > len(person_list):
            return
        idx = np.argmin(np.sum((np.array([p.bbox[:2] for p in person_list]) - bbox_list[i][:2])**2,axis= 1))
        person_list[idx].action = action_name



def clear_person(person_list,threshold = 0.3):
    delete = []
    for p in person_list:
        if not p.refresh:
            p.confidence -= 0.05
            p.confidence = round(p.confidence, 2)
        if p.confidence < threshold:
            delete.append(p)
    for delete_person in delete:
        person_list.remove(delete_person)


class Person:
    def __init__(self, color_id, mask,bbox,confidence = 1.0,threshold = 0.5):
        self.color_id = color_id
        self.mask = mask
        self.bbox = bbox # [x1,y1,x2,y2]
        self.confidence = confidence
        self.threshold = threshold
        self.refresh = False
        self.location = -1
        self.action = 0
        self.action_l2 = -1


    def mask_iou(self,mask):
        union_mask = self.mask + mask
        union_area = union_mask[np.where(union_mask >= 1)]
        intersect_area = union_mask[np.where(union_mask > 1)]
        iou = len(intersect_area) / len(union_area)
        return iou

    def update(self,masks,bboxs):
        for mask,bbox in zip(masks,bboxs):
            mask = mask[0]
            mask[np.where(mask > 0.5)] = 1
            if self.mask_iou(mask) > self.threshold:
                self.confidence = np.random.randint(8,10)/10
                self.mask = mask
                self.bbox = bbox
                self.refresh = True
                break

    def update_location(self):
        self.location = Action.get_loc()

    def update_action(self):
        pass

