import json
import math
import random
import time
import cv2
import os
import sys
import numpy as np
import copy

from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist


CLASSES = ('unlabeled','ship','storage_tank','baseball_diamond',
           'tennis_court','basketball_court','Ground_Track_Field','Bridge',
           'Large_Vehicle','Small_Vehicle','Helicopter','Swimming_pool',
           'Roundabout','Soccer_ball_field','plane','Harbor')

PALETTE = [(0,  0,  0), (0,  0, 63), (0, 63, 63), (0, 63,  0),
           (0, 63,127), (0, 63,191), (0, 63,255), (0,127, 63),
           (0,127,127), (0,  0,127), (0,  0,191), (0,  0,255),
           (0,191,127), (0,127,191), (0,127,255),(0,100,155)]

dota_wordname_15 = [
    'plane',
    'baseball-diamond',
    'bridge',
    'ground-track-field',
    'small-vehicle',
    'large-vehicle',
    'ship',
    'tennis-court',
    'basketball-court',
    'storage-tank',
    'soccer-ball-field',
    'roundabout',
    'harbor',
    'swimming-pool',
    'helicopter']

isaid_wordname_16 = [
    'unlabeled',
    'plane',
    'baseball_diamond',
    'Bridge',
    'Ground_Track_Field',
    'Small_Vehicle',
    'Large_Vehicle',
    'ship',
    'tennis_court',
    'basketball_court',
    'storage_tank',
    'Soccer_ball_field',
    'Roundabout',
    'Harbor',
    'Swimming_pool',
    'Helicopter'
]


# category map
mapping = dict(zip(isaid_wordname_16[1:], dota_wordname_15))

# classify category via sem_color 
def get_class_from_color(sem_color):
    # change sem_color to BGR
    bgr_color = (sem_color[2], sem_color[1], sem_color[0])

    if bgr_color==(124,116,104):  # padding
        return 'padding'
    color_index = PALETTE.index(bgr_color)
    class_name = CLASSES[color_index]

    return class_name


def get_single_pt_via_mask(root, out_path, dis_threshold):
    
    ins_mask_root=root+'Instance_masks_images'
    semantic_root=root+'Semantic_masks_images'
    
    mask_image_list=os.listdir(ins_mask_root)
    # sem_image_list=os.listdir(semantic_root)

    for mask_image_name in mask_image_list:
        mask_path=os.path.join(ins_mask_root,mask_image_name)
        # mask_name P0000_instance_id_RGB__1024__0___3296.png
        # sem_name P0000_instance_color_RGB__1024__0___3296.png
        sem_image_name = mask_image_name.replace('id', 'color')
        sem_path=os.path.join(semantic_root,sem_image_name)
        out_image_name=mask_image_name.replace('_instance_id_RGB','')
        out_txt_name=out_image_name.split('.')[0]+'.txt'


        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        sem_image = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)
        out_image_vis = copy.deepcopy(mask_image)
        unique_colors = set(tuple(pixel) for row in mask_image for pixel in row)
        # unique_colors = np.unique(mask_image.reshape(-1, 3), axis=0)
        # unique_colors = [tuple(color) for color in unique_colors]

        f = open(os.path.join(out_path, out_txt_name), 'w')
        # each instance
        for color in unique_colors:
            # get all pixels in a mask
            ins_pxs = np.where(np.all(mask_image == color, axis=-1))
            # ins_pxs2 = np.where(np.all(mask_image[..., ::-1] == color[::-1], axis=-1))
            # num_pxs = len(ins_pxs[0])
            x0 = ins_pxs[0][0]
            y0 = ins_pxs[1][0]
            sem_color = sem_image[x0, y0, :]
            # get category
            category = get_class_from_color(sem_color)
            
            if category == 'unlabeled' or category=='padding':
                # print('pass background class')
                continue
            # 确保该实例所有像素对应的语义标签颜色是否一致
            if not np.all(sem_image[ins_pxs[0], ins_pxs[1], :] == sem_color):
                print("警告：实例像素的颜色不一致!!!")
                break

            ## 1)根据像素坐标获取实例重心坐标
            ct_x = np.mean(ins_pxs[0])
            ct_y = np.mean(ins_pxs[1])
            ## 2)计算实例所有点到重心的区域
            # 计算每个点到重心的平面距离
            ins_pxs = np.column_stack((ins_pxs[0], ins_pxs[1]))
            # 计算每个点到重心的欧氏距离
            distances = cdist(ins_pxs, [[ct_x, ct_y]])
            ## 3)选出随机点
            # 获取最小距离和最大距离
            min_dis = np.min(distances)
            max_dis = np.max(distances)
            dis_thr = min_dis + (max_dis - min_dis) * dis_threshold  # 范围比例

            # 使用阈值得到中心区域的待选点
            candidate_indices = np.where(distances <= dis_thr)[0]
            candidate_points = ins_pxs[candidate_indices]
            candidate_num = len(candidate_points)
            if candidate_num > 0:
                # 在待选点中随机选取
                # seed = int(time.time()) + random.randint(1, 1000)
                seed = int.from_bytes(os.urandom(8), byteorder="big") % 1000000
                random.seed(seed)
                rand_point = random.choice(candidate_points)
            else:
                # 此时初始范围内无匹配点,将最近点作为最终点
                print('选择最近点!')
                rand_point = ins_pxs[np.where(distances == min_dis)[0]]

            x = float(rand_point[1])  # 更换x,y的顺序
            y = float(rand_point[0])
            x1 = x - 1.0
            y1 = y - 1.0
            x2 = x + 1.0
            y2 = y - 1.0
            x3 = x + 1.0
            y3 = y + 1.0
            x4 = x - 1.0
            y4 = y + 1.0
            difficult = 0

            # 写新文件

            dota_class_name = mapping[category]
            # 转换为pseudo box写入
            f.write(str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' ' +
                    str(x3) + ' ' + str(y3) + ' ' + str(x4) + ' ' + str(y4) + ' ' +
                    dota_class_name + ' ' + str(difficult) + '\n')
        f.close()
    print('done!')
    return

def main():
    # data_root='/media/dell/data1/ljw/code/test1/mmrotate1_x/data/split_ss_isaid_mask_1024_200/'
    # out_path='/media/dell/data1/ljw/code/test1/mmrotate1_x/data/split_ss_isaid_mask_1024_200/labelTxt_sp_thr02'
   
    data_root='/media/dell/data1/ljw/data/DOTA/iSAID_anno/trainval/'
    out_path='/media/dell/data1/ljw/data/DOTA/iSAID_anno/trainval/labelTxt_sp_thr02'
    dis_threshold=0.2
    get_single_pt_via_mask(data_root, out_path, dis_threshold)

    print('done!')


if __name__ == '__main__':
    main()
