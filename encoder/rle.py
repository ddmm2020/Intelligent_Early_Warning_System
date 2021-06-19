# -*- coding: utf-8 -*-
# @File  : rle.py
# @Author: ddmm
# @Date  : 2021/3/23
# @Desc  :


import numpy as np
import json
import time

def generate_json(masks,bbox):

    send_info = {
        "time": time.time(),
        "cnt": len(masks),
    }
    person_list = []
    for i, (mask) in enumerate(masks):
        # send_info["person{}".format(i)] = rle_encoder(mask)
        person_list.append(rle_encoder(mask))
    send_info["person"] = person_list
    send_info["bbox"] = bbox
    json_send =json.dumps(send_info)
    return json_send


def rle_encoder(mask):
    mask_copy = mask.astype(np.uint8)
    # l1 = len(mask[mask == 1])
    # l2 = len(mask[mask == 0])
    # l3 = len(mask[mask > 1])
    # print(l3)
    # assert  l1 + l2 == len(mask)

    start_time = time.time()  # 记录程序开始运行时间
    flatten_mask = np.ravel(mask_copy)
    rle_code = ""
    locate_zero = np.argwhere(flatten_mask == 0).flatten()
    locate_one = np.argwhere(flatten_mask == 1).flatten()

    max_len = len(flatten_mask)
    zero_loc = max_len if len(locate_zero) == 0 else locate_zero[0]
    one_loc = max_len if len(locate_one) == 0 else locate_one[0]

    # start = 1 if one_loc < zero_loc else 0
    while True:
        if zero_loc < one_loc:
            length = one_loc - zero_loc
            rle_code += "{}:{},".format(0, length)
            zero_next = locate_zero[np.argwhere(locate_zero > one_loc).flatten()]

            if len(zero_next) == 0:
                length = max_len - zero_loc -length
                rle_code += "{}:{},".format(1, length)
                break
            zero_loc = zero_next[0]
        else:
            # one_loc < zero_loc
            # 1[one_loc]1111110[zero_loc]000000011

            length = zero_loc - one_loc
            rle_code += "{}:{},".format(1, length)
            one_next = locate_one[np.argwhere(locate_one > zero_loc).flatten()]
            if len(one_next) == 0:
                length = max_len - one_loc -length
                rle_code += "{}:{},".format(0, length)
                break
            one_loc = one_next[0]
    end_time = time.time()  # 记录程序结束运行时间
    print('usr time %f second' % (end_time - start_time))
    return rle_code


if __name__ == '__main__':
    test = []
    mask1 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, ],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, ],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, ]]
    )
    mask2 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
    ])

    test.append(mask1)
    test.append(mask2)

    generate_json(test)

    for mask in test:
        rle = rle_encoder(mask)
        print(rle)