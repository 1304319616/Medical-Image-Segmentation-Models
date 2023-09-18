# coding=gbk
# coding:utf-8
import glob
import os

import cv2
import numpy as np


def npz():
    # ԭͼ��·��
    path = r'G:\datasets\ʡҽԺ����\�ָ����ݼ�\train\PNG\*\PNG\*.png'
    # path = r'G:\datasets\ʡҽԺ����\�ָ����ݼ�\train\PNG\*\PNG\*.png'
    path1 = r'G:\datasets\data\Lung Segmentation\labels\*.png'
    # ��Ŀ�д��ѵ�����õ�npz�ļ�·��
    path2 = r'G:\PycharmProjects\lesson-2\data\Synapse\train_npz\\'
    for i, img_path in enumerate(glob.glob(path)):
        print(img_path)
        # ����ͼ��
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 0)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # �����ǩ
        label_path = img_path.replace('PNG', 'LABEL')
        # label_path = img_path.replace('images', 'labels')
        label =cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), 0)
        print(label_path)
        # ����Ŀ����������Ϊ0
        label[label != 255] = 0
        # ��Ŀ����������Ϊ1
        label[label == 255] = 1
        # ����npz
        name = img_path.split('.')[0].split('\\')[-3]
        num = img_path.split('.')[0].split('\\')[-1]

        tar = os.path.join(path2, name + '_' + num + '_' + str(i))
        print(tar)
        np.savez(tar, image=image, label=label)
        print('------------', i)

    # ����npz�ļ�
    # data = np.load(r'G:\dataset\Unet\Swin-Unet-ori\data\Synapse\train_npz\0.npz', allow_pickle=True)
    # image, label = data['image'], data['label']

    print('ok')


def npz1():
    # ԭͼ��·��
    path = r'G:\datasets\ʡҽԺ����\�ָ����ݼ�\test\PNG\*\PNG\*.png'
    path1 = r'G:\datasets\data\Lung Segmentation\labels_test\*.png'
    # ��Ŀ�д��ѵ�����õ�npz�ļ�·��
    path2 = r'G:\PycharmProjects\lesson-2\data\Synapse\test_vol_h5\\'
    for i, img_path in enumerate(glob.glob(path)):
        print(img_path)
        # ����ͼ��
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 0)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # �����ǩ
        label_path = img_path.replace('PNG', 'LABEL')
        # label_path = img_path.replace('images', 'labels')
        label = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), 0)
        print(label_path)
        # ����Ŀ����������Ϊ0
        label[label != 255] = 0
        # ��Ŀ����������Ϊ1
        label[label == 255] = 1
        # ����npz
        name = img_path.split('.')[0].split('\\')[-3]
        num = img_path.split('.')[0].split('\\')[-1]

        tar = os.path.join(path2, name + '_' + num + '_' + str(i))
        print(tar)
        np.savez(tar, image=image, label=label)
        print('------------', i)

    # ����npz�ļ�
    # data = np.load(r'G:\dataset\Unet\Swin-Unet-ori\data\Synapse\train_npz\0.npz', allow_pickle=True)
    # image, label = data['image'], data['label']

    print('ok')


if __name__ == "__main__":
    npz()
    npz1()

