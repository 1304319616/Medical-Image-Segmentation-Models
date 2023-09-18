# coding=gbk
# coding:utf-8
import glob
import os

import cv2
import numpy as np


def npz():
    # 原图像路径
    path = r'G:\datasets\省医院数据\分割数据集\train\PNG\*\PNG\*.png'
    # path = r'G:\datasets\省医院数据\分割数据集\train\PNG\*\PNG\*.png'
    path1 = r'G:\datasets\data\Lung Segmentation\labels\*.png'
    # 项目中存放训练所用的npz文件路径
    path2 = r'G:\PycharmProjects\lesson-2\data\Synapse\train_npz\\'
    for i, img_path in enumerate(glob.glob(path)):
        print(img_path)
        # 读入图像
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 0)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 读入标签
        label_path = img_path.replace('PNG', 'LABEL')
        # label_path = img_path.replace('images', 'labels')
        label =cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), 0)
        print(label_path)
        # 将非目标像素设置为0
        label[label != 255] = 0
        # 将目标像素设置为1
        label[label == 255] = 1
        # 保存npz
        name = img_path.split('.')[0].split('\\')[-3]
        num = img_path.split('.')[0].split('\\')[-1]

        tar = os.path.join(path2, name + '_' + num + '_' + str(i))
        print(tar)
        np.savez(tar, image=image, label=label)
        print('------------', i)

    # 加载npz文件
    # data = np.load(r'G:\dataset\Unet\Swin-Unet-ori\data\Synapse\train_npz\0.npz', allow_pickle=True)
    # image, label = data['image'], data['label']

    print('ok')


def npz1():
    # 原图像路径
    path = r'G:\datasets\省医院数据\分割数据集\test\PNG\*\PNG\*.png'
    path1 = r'G:\datasets\data\Lung Segmentation\labels_test\*.png'
    # 项目中存放训练所用的npz文件路径
    path2 = r'G:\PycharmProjects\lesson-2\data\Synapse\test_vol_h5\\'
    for i, img_path in enumerate(glob.glob(path)):
        print(img_path)
        # 读入图像
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 0)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 读入标签
        label_path = img_path.replace('PNG', 'LABEL')
        # label_path = img_path.replace('images', 'labels')
        label = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), 0)
        print(label_path)
        # 将非目标像素设置为0
        label[label != 255] = 0
        # 将目标像素设置为1
        label[label == 255] = 1
        # 保存npz
        name = img_path.split('.')[0].split('\\')[-3]
        num = img_path.split('.')[0].split('\\')[-1]

        tar = os.path.join(path2, name + '_' + num + '_' + str(i))
        print(tar)
        np.savez(tar, image=image, label=label)
        print('------------', i)

    # 加载npz文件
    # data = np.load(r'G:\dataset\Unet\Swin-Unet-ori\data\Synapse\train_npz\0.npz', allow_pickle=True)
    # image, label = data['image'], data['label']

    print('ok')


if __name__ == "__main__":
    npz()
    npz1()

