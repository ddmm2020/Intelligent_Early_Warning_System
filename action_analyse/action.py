# -*- coding: utf-8 -*-
# @File  : action.py
# @Author: ddmm
# @Date  : 2021/5/10
# @Desc  :

from config.config import Config as CFG
import pandas as pd
import re
import numpy as np

# 0站立，1坐，2走，3掌舵，4跌倒，5玩手机，6睡觉，7俯身

class Action(object):
    def __init__(self, data_file="./test_clear_v2.csv", action_names=["站立"], loc_coord=[[0, 0]], l2_thresh=10):
        self.action_name = action_names
        self.loc_coord = loc_coord
        self.l2_thresh = l2_thresh
        self.loc_label = [[372, 0], [372, 240], [372, 480], [522, 0], [522, 240], [522, 480],
                          [672, 0], [672, 240], [672, 480], [822, 0], [822, 240], [822, 480],
                          [972, 0], [972, 240], [972, 480], [1122, 0], [1122, 240], [1122, 480],
                          [1272, 0], [1272, 240], [1272, 480]]

        self.id2action = dict(zip(range(len(self.action_name)), self.action_name))
        self.action_name = ["未识别","走动", "站立", "坐", "掌舵", "玩手机", "瞭望", "打盹"]

        # self.action_label = pd.read_csv(CFG.action_label)
        # self.action_data = pd.read_csv(CFG.action_data)
        # self.df = pd.merge(self.action_data,self.action_label, how='right', left_on='file_name', right_on='file_name')
        # self.df = pd.concat([self.action_data,self.action_label])
        # self.df.to_csv("test.csv",header=True,index=False)
        self.df = pd.read_csv(data_file)
        self.action_template_kp = [[] for i in range(CFG.loc_num)]
        self.action_template = [[] for i in range(CFG.loc_num)]
        self.gen_action_loc_table()

    def normalize(self, keypoint, bbox):
        [x1, y1, w, h] = bbox
        # h = y2 - y1 + 1
        # w = x2 - x1 + 1
        keypoint -= [x1, y1, 0]
        keypoint = keypoint.astype(np.float16) * [1 / w, 1 / h, 1]
        return keypoint

    def gen_action_loc_table(self):
        locations = self.df["bbox"].values
        actions = self.df["action"].values
        keypoints = self.df["keypoint"].values
        # kp_scores = self.df["kp_score"].values
        for i in range(len(actions)):
            # print(locations[i])
            boxes = re.findall(r"\b\d+\b", locations[i])
            boxes = [int(b) for b in boxes]
            loc_id = self.get_loc(boxes)
            kps = re.findall(r"\b\d+\b", keypoints[i])
            kps = [int(kp) for kp in kps]
            kps = np.array(kps).reshape(-1, 3)
            kps = self.normalize(kps, [self.loc_label[loc_id][0], self.loc_label[loc_id][1], abs(boxes[0] - boxes[1]),
                                       abs(boxes[1] - boxes[3])])
            self.action_template[loc_id].append(int(actions[i]))
            self.action_template_kp[loc_id].append(kps)

    def get_loc(self, boxes):
        """
        :param boxes: [x1,y1,x2,y2]
        :return:
        """
        x1, y1 = boxes[0], boxes[1]
        return int(max(0, (x1 - CFG.start_x)) / CFG.step_x) * 3 + int(y1 / CFG.step_y)

    def get_l2(self, keypoint1, tmplate_keypoint):
        keypoint1 = np.array(keypoint1)
        tmplate_keypoint = np.array(tmplate_keypoint)
        kps1 = keypoint1[:, :2]
        taplate_kps = tmplate_keypoint[:, :2]
        mask = np.logical_and(tmplate_keypoint[:, -1], keypoint1[:, -1])
        reg = np.sum(mask)
        return (np.sum(np.sum(np.abs(kps1 - taplate_kps), axis=1) * mask)) / reg

    def get_action(self, keypoints, bbox):
        idx = self.get_loc(bbox)
        keypoints = self.normalize(keypoints, [self.loc_label[idx][0], self.loc_label[idx][1], abs(bbox[0] - bbox[1]),
                                               abs(bbox[1] - bbox[3])])
        t_kps = self.action_template_kp[idx]
        t_action = self.action_template[idx]
        max_l2 = 99999
        action_idx = -1
        for i, (t_kp) in enumerate(t_kps):
            l2 = self.get_l2(keypoints, t_kp)
            if l2 < max_l2:
                action_idx = i
                max_l2 = l2
        # return max_l2, self.action_name[t_action[action_idx]]
        return max_l2, t_action[action_idx]


if __name__ == '__main__':
    A = Action()
    # print(A.get_loc([573,720,0,0]))
    print(A.get_l2([[2, 4, 1], [2, 2, 1]], [[2, 4, 1], [2, 1, 1]]))
