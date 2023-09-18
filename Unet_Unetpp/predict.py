import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet
from model.unetplusplus import NestedUNet
import random
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume

from tqdm import tqdm

from model.unet_model import UNet

import torch
from torchvision import transforms


def inference(model, test_save_path=None):
    db_test = Synapse_dataset(base_dir='./data/Synapse/test_vol_h5', list_dir='./lists/lists_Synapse', split="test_vol")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        image = image.unsqueeze(dim=0)
        label = label.unsqueeze(dim=0).float()

        # 测试参数
        metric_i = test_single_volume(image, label, model, classes=2, patch_size=[512, 512],
                                      test_save_path=test_save_path, case=case_name)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, 2):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"



if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    # 使用UNet
    net = UNet(n_channels=1, n_classes=1)
    # 使用UNet++
    # net = NestedUNet()
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('output/best_network.pth', map_location=device))

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+'best_network.pth'+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info('best_model.pth')

    test_save_dir = os.path.join('output', "predictions")
    test_save_path = test_save_dir
    os.makedirs(test_save_path, exist_ok=True)

    inference(net, test_save_path)
