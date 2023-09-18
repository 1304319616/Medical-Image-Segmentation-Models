# -*- coding: UTF-8 -*
import random

import numpy as np
from tqdm import tqdm

from model.unet_model import UNet
from model.unetplusplus import NestedUNet
# from utils.dataset import ISBI_Loader
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from torch import optim
import torch.nn as nn
import torch
from torchvision import transforms
from early_stop import EarlyStopping
from utils import test_single_volume

Batch_size = 1


def train_net(net, epochs=100, batch_size=Batch_size, lr=1e-3):
    # 加载训练集
    # isbi_dataset = ISBI_Loader(data_path)
    db_train = Synapse_dataset(base_dir='./data/Synapse/train_npz', list_dir='./lists/lists_Synapse', split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[512, 512])]))
    from torch.utils.data import DataLoader

    def worker_init_fn(worker_id):
        random.seed(1234 + worker_id)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    db_test = Synapse_dataset(base_dir='./data/Synapse/test_vol_h5', list_dir='./lists/lists_Synapse', split="test_vol")
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)

    # 定义RMSprop算法
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    save_path = "output"  # 当前目录下
    early_stopping = EarlyStopping(save_path)

    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        print('-----------------------epoch: ', epoch, '-----------------------')
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch = image_batch.unsqueeze(dim=0)
            label_batch = label_batch.unsqueeze(dim=0).float()
            # print(image_batch.shape)
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = net(image_batch)
            loss = criterion(outputs, label_batch)
            print('Loss/train', i_batch, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        net.eval()
        metric_list = 0.0

        for i_batch, sampled_batch in enumerate(test_loader):
            # print("test_data len:",len(test_data))
            image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
            image = image.unsqueeze(dim=0)
            label = label.unsqueeze(dim=0).float()

            # 测试参数
            metric_i = test_single_volume(image, label, net, classes=2, patch_size=[512, 512],
                                          test_save_path=None, case=case_name)
            metric_list += np.array(metric_i)

        metric_list = metric_list / len(db_test)

        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]

        early_stopping(performance, net)
        if early_stopping.early_stop:
            print('Early stopping')
            break

        '''for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
                print('save best mdoel: ', )
            # 更新参数
            loss.backward()
            optimizer.step()'''


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    # 使用UNet
    net = UNet(n_channels=Batch_size, n_classes=1)
    # 使用UNet++
    # net = NestedUNet()
    # 将网络拷贝到deivce中
    net.to(device=device)

    train_net(net)
