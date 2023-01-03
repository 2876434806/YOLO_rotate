import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile
import codecs

import numpy as np
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (LOGGER, check_dataset, check_requirements, check_yaml, clean_str, segments2boxes, xyn2xy,
                           xywh2xyxy, xywhn2xyxy, xyxy2xywhn)

def rotate_augment(angle, scale, image, labels):
    """
    旋转目标增强  随机旋转
    @param angle: 旋转增强角度 int 单位为度
    @param scale: 设为1,尺度由train.py中定义
    @param image:  img信息  shape(heght, width, 3)
    @param labels:  (num, [classid x_c y_c longside shortside Θ]) Θ ∈ int[0,180)
    @return:
           array rotated_img: augmented_img信息  shape(heght, width, 3)
           array rotated_labels: augmented_label:  (num, [classid x_c y_c longside shortside Θ])
    """
    Pi_angle = -angle * math.pi / 180.0  # 弧度制，后面旋转坐标需要用到，注意负号！！！
    rows, cols = image.shape[:2]
    a, b = cols / 2, rows / 2
    M = cv2.getRotationMatrix2D(center=(a, b), angle=angle, scale=scale)
    rotated_img = cv2.warpAffine(image, M, (cols, rows))  # 旋转后的图像保持大小不变
    rotated_labels = []
    for label in labels:
        # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
        rect = longsideformat2cvminAreaRect(label[1], label[2], label[3], label[4], (label[5] - 179.9))
        # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        poly = cv2.boxPoints(rect)  # 返回rect对应的四个点的值 normalized

        # 四点坐标反归一化
        poly[:, 0] = poly[:, 0] * cols
        poly[:, 1] = poly[:, 1] * rows

        # 下面是计算旋转后目标相对旋转过后的图像的位置
        X0 = (poly[0][0] - a) * math.cos(Pi_angle) - (poly[0][1] - b) * math.sin(Pi_angle) + a
        Y0 = (poly[0][0] - a) * math.sin(Pi_angle) + (poly[0][1] - b) * math.cos(Pi_angle) + b

        X1 = (poly[1][0] - a) * math.cos(Pi_angle) - (poly[1][1] - b) * math.sin(Pi_angle) + a
        Y1 = (poly[1][0] - a) * math.sin(Pi_angle) + (poly[1][1] - b) * math.cos(Pi_angle) + b

        X2 = (poly[2][0] - a) * math.cos(Pi_angle) - (poly[2][1] - b) * math.sin(Pi_angle) + a
        Y2 = (poly[2][0] - a) * math.sin(Pi_angle) + (poly[2][1] - b) * math.cos(Pi_angle) + b

        X3 = (poly[3][0] - a) * math.cos(Pi_angle) - (poly[3][1] - b) * math.sin(Pi_angle) + a
        Y3 = (poly[3][0] - a) * math.sin(Pi_angle) + (poly[3][1] - b) * math.cos(Pi_angle) + b

        poly_rotated = np.array([(X0, Y0), (X1, Y1), (X2, Y2), (X3, Y3)])
        # 四点坐标归一化
        poly_rotated[:, 0] = poly_rotated[:, 0] / cols
        poly_rotated[:, 1] = poly_rotated[:, 1] / rows

        rect_rotated = cv2.minAreaRect(np.float32(poly_rotated))  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）

        c_x = rect_rotated[0][0]
        c_y = rect_rotated[0][1]
        w = rect_rotated[1][0]
        h = rect_rotated[1][1]
        theta = rect_rotated[-1]  # Range for angle is [-90，0)
        # (num, [classid x_c y_c longside shortside Θ])
        label[1:] = cvminAreaRect2longsideformat(c_x, c_y, w, h, theta)

        if (sum(label[1:-1] <= 0) + sum(label[1:3] >= 1)) >= 1:  # 0<xy<1, 0<side<=1
            # print('bbox[:2]中有>= 1的元素,bbox中有<= 0的元素,已将某个box排除,')
            np.clip(label[1:-1], 0, 1, out=label[1:-1])

        label[-1] = int(label[-1] + 180.5)  # range int[0,180] 四舍五入
        if label[-1] == 180:  # range int[0,179]
            label[-1] = 179
        rotated_labels.append(label)
        labels = np.array(rotated_labels)
    return rotated_img, labels
def rotateAugment(angle, scale, image, labels):
    """
    旋转目标增强  随机旋转
    @param angle: 旋转增强角度 int 单位为度
    @param scale: 设为1,尺度由train.py中定义
    @param image:  img信息  shape(heght, width, 3)
    @param labels:  (num, [classid x_c y_c longside shortside Θ]) Θ ∈ int[0,180)
    @return:
            rotated_img: augmented_img信息  shape(heght, width, 3)
            rotated_labels: augmented_label:  (num, [classid x_c y_c longside shortside Θ])
    """
    Pi_angle = -angle * math.pi / 180.0  # 弧度制，后面旋转坐标需要用到，注意负号！！！
    rows, cols = image.shape[:2]
    a, b = cols / 2, rows / 2
    M = cv2.getRotationMatrix2D(center=(a, b), angle=angle, scale=scale)
    rotated_img = cv2.warpAffine(image, M, (cols, rows))  # 旋转后的图像保持大小不变
    rotated_labels = []
    for label in labels:
        # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
        rect = longsideformat2cvminAreaRect(label[1], label[2], label[3], label[4], (label[5] - 179.9))
        # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        poly = cv2.boxPoints(rect)  # 返回rect对应的四个点的值 normalized

        # 四点坐标反归一化
        poly[:, 0] = poly[:, 0] * cols
        poly[:, 1] = poly[:, 1] * rows

        # 下面是计算旋转后目标相对旋转过后的图像的位置
        X0 = (poly[0][0] - a) * math.cos(Pi_angle) - (poly[0][1] - b) * math.sin(Pi_angle) + a
        Y0 = (poly[0][0] - a) * math.sin(Pi_angle) + (poly[0][1] - b) * math.cos(Pi_angle) + b

        X1 = (poly[1][0] - a) * math.cos(Pi_angle) - (poly[1][1] - b) * math.sin(Pi_angle) + a
        Y1 = (poly[1][0] - a) * math.sin(Pi_angle) + (poly[1][1] - b) * math.cos(Pi_angle) + b

        X2 = (poly[2][0] - a) * math.cos(Pi_angle) - (poly[2][1] - b) * math.sin(Pi_angle) + a
        Y2 = (poly[2][0] - a) * math.sin(Pi_angle) + (poly[2][1] - b) * math.cos(Pi_angle) + b

        X3 = (poly[3][0] - a) * math.cos(Pi_angle) - (poly[3][1] - b) * math.sin(Pi_angle) + a
        Y3 = (poly[3][0] - a) * math.sin(Pi_angle) + (poly[3][1] - b) * math.cos(Pi_angle) + b

        poly_rotated = np.array([(X0, Y0), (X1, Y1), (X2, Y2), (X3, Y3)])
        # 四点坐标归一化
        poly_rotated[:, 0] = poly_rotated[:, 0] / cols
        poly_rotated[:, 1] = poly_rotated[:, 1] / rows

        rect_rotated = cv2.minAreaRect(np.float32(poly_rotated))  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）

        c_x = rect_rotated[0][0]
        c_y = rect_rotated[0][1]
        w = rect_rotated[1][0]
        h = rect_rotated[1][1]
        theta = rect_rotated[-1]  # Range for angle is [-90，0)

        label[1:] = cvminAreaRect2longsideformat(c_x, c_y, w, h, theta)

        if (sum(label[1:-1] <= 0) + sum(label[1:3] >= 1)) >= 1:  # 0<xy<1, 0<side<=1
            # print('bbox[:2]中有>= 1的元素,bbox中有<= 0的元素,已将某个box排除,')
            # print(
            #     '出问题的longside形式数据:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (label[1], label[2], label[3], label[4], label[5]))
            continue

        label[-1] = int(label[-1] + 180.5)  # range int[0,180] 四舍五入
        if label[-1] == 180:  # range int[0,179]
            label[-1] = 179
        rotated_labels.append(label)

    return rotated_img, np.array(rotated_labels)

def longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, theta_longside):
    '''
    trans longside format(x_c, y_c, longside, shortside, θ) to minAreaRect(x_c, y_c, width, height, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param longside: 最长边
    @param shortside: 最短边
    @param theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    @return: ((x_c, y_c),(width, height),Θ)
            x_c: center_x
            y_c: center_y
            width: x轴逆时针旋转碰到的第一条边最长边
            height: 与width不同的边
            theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    '''
    if ((theta_longside >= -180) and (theta_longside < -90)):  # width is not the longest side
        width = shortside
        height = longside
        theta = theta_longside + 90
    else:
        width = longside
        height = shortside
        theta = theta_longside

    if (theta < -90) or (theta >= 0):
        print('当前θ=%.1f，超出opencv的θ定义范围[-90, 0)' % theta)

    return ((x_c, y_c), (width, height), theta)

def cvminAreaRect2longsideformat(x_c, y_c, width, height, theta):
    '''
    trans minAreaRect(x_c, y_c, width, height, θ) to longside format(x_c, y_c, longside, shortside, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param width: x轴逆时针旋转碰到的第一条边
    @param height: 与width不同的边
    @param theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    @return:
            x_c: center_x
            y_c: center_y
            longside: 最长边
            shortside: 最短边
            theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    '''
    '''
    意外情况:(此时要将它们恢复符合规则的opencv形式：wh交换，Θ置为-90)
    竖直box：box_width < box_height  θ=0
    水平box：box_width > box_height  θ=0
    '''
    if theta == 0:
        theta = -90
        buffer_width = width
        width = height
        height = buffer_width

    if theta > 0:
        if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
            print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (
            x_c, y_c, width, height, theta))
        return False

    if theta < -90:
        print(
            'θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if width != max(width, height):  # 若width不是最长边
        longside = height
        shortside = width
        theta_longside = theta - 90
    else:  # 若width是最长边(包括正方形的情况)
        longside = width
        shortside = height
        theta_longside = theta

    if longside < shortside:
        print('旋转框转换表示形式后出现问题：最长边小于短边;[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
        x_c, y_c, longside, shortside, theta_longside))
        return False
    if (theta_longside < -180 or theta_longside >= 0):
        print('旋转框转换表示形式时出现问题:θ超出长边表示法的范围：[-180,0);[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
        x_c, y_c, longside, shortside, theta_longside))
        return False

    return x_c, y_c, longside, shortside, theta_longside

def get_list(f):
    line = f.readline()   # 以行的形式进行读取文件
    list1 = []
    while line:
        list2 = []
        a = line.split()
        for i in a:  # 这是选取需要读取的位数
            list2.append(i)  # 将其添加在列表之中
        line = f.readline()
        list1.append(list2)
    f.close()
    return list1

def get_labels(label_path):
    f = codecs.open(label_path, mode='r', encoding='utf-8')
    line = f.readline()  # 以行的形式进行读取文件
    list1 = []
    while line:
        num = 0
        list2 = []
        a = line.split()
        for i in a:  # 这是选取需要读取的位数
            num = num + 1
            if num == 1 or num == 6:
                i = int(i)
            else:
                i = float(i)
            list2.append(i)  # 将其添加在列表之中
        line = f.readline()
        list1.append(list2)
    f.close()
    return list1

def save_txt(data,file):
    list_file = open(file, 'w')
    for i in data:
        num = 0
        for j in i:
            num = num + 1
            if num == 1 or num == 6:
                j = int(j)
            else:
                j = float(j)
            print(j)
            list_file.write(str(j) + " ")
        list_file.write("\n")


if __name__ == '__main__':
    img_ori_path = "F:\dataset\HRSC2016\pad_augment_data\images_o\\"
    label_ori_path = "F:\dataset\HRSC2016\pad_augment_data\labels_o\\"
    save_img_path = "F:\dataset\HRSC2016\pad_augment_data\\after_augment\\images_4\\"
    save_label_path = "F:\dataset\HRSC2016\pad_augment_data\\after_augment\\labels_4\\"
    img_files = os.listdir(img_ori_path)
    print(img_files)
    for i in img_files:
        img_path = os.path.join(img_ori_path, i)
        l = i.replace("png","txt")
        i = i.replace(".png","")
        label_path = os.path.join(label_ori_path, l)
        im = cv2.imread(img_path)  # BGR
        label = get_labels(label_path)

        label = np.array(label)

        # # #上下翻转—2
        # img = np.flipud(im)
        # label[:, 2] = 1 - label[:, 2]
        # label[:, 5] = 180 - label[:, 5]  # θ changed when Flip up-down.
        # label[label[:, 5] == 180, 5] = 0

        # #左右翻转-1
        # img = np.fliplr(im)
        # label[:, 1] = 1 - label[:, 1]
        # label[:, 5] = 180 - label[:, 5]  # θ changed when Flip left-right.
        # label[label[:, 5] == 180, 5] = 0

        #旋转20度-3-20 -4-40
        degrees = 40.0
        rotate_angle = random.uniform(-degrees, degrees)
        img, label = rotateAugment(rotate_angle, 1, im, label)

        labels_out = torch.zeros((label.shape[0], 6))
        labels_out = torch.from_numpy(label)

        save_imgs_path = save_img_path + i +"_4.png"
        save_labels_path = save_label_path + i + "_4.txt"
        cv2.imwrite(save_imgs_path, img)
        save_txt(labels_out, save_labels_path)
