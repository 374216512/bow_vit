import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Dataset
import os
from PIL import Image


logger = logging.getLogger(__name__)


class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.img_list = []
        self.w_list = []
        self.label_list = []
        self.label2idx = {}
        self.idx2label = {}
        self.weather2idx = {
            "暴风雪": 1,
            "暴雪": 2,
            "暴雨": 3,
            "冰雹": 4,
            "大风": 5,
            "大雾": 6,
            "大雪": 7,
            "大雨": 8,
            "地震": 9,
            "冻雨": 10,
            "多云": 11,
            "风": 12,
            "狂风": 13,
            "雷电": 14,
            "雷雨": 15,
            "雷阵雨": 16,
            "其他": 17,
            "晴": 18,
            "晴朗": 19,
            "晴天": 20,
            "沙尘": 21,
            "沙尘暴": 22,
            "霜冻": 23,
            "台风": 24,
            "雾": 25,
            "小雪": 26,
            "小雨": 27,
            "雪": 28,
            "阴": 29,
            "阴天": 30,
            "雨": 31,
            "雨夹雪": 32,
            "雨天": 33,
            "雨雪": 34,
            "阵雪": 35,
            "阵雨": 36,
            "中雪": 37,
            "山火": 100
        }

        for idx, label in enumerate(sorted(os.listdir(data_dir))):
            c_dir = os.path.join(data_dir, label)
            self.label2idx[label] = idx
            self.idx2label[idx] = label

            for img_path in sorted(os.listdir(c_dir)):
                weather = img_path.split('_')[3]
                self.img_list.append(os.path.join(c_dir, img_path))
                self.w_list.append(weather)
                self.label_list.append(idx)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        label = self.label_list[item]
        weather = self.weather2idx[self.w_list[item]]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, weather, label

    def __len__(self):
        return len(self.img_list)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(
            0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    trainset = MyDataset(f'{args.data_path}/training', transform_train)
    testset = MyDataset(f'{args.data_path}/testing', transform_test)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(
        trainset) if args.local_rank == -1 else DistributedSampler(trainset)  # 相当于是打乱样本顺序
    test_sampler = SequentialSampler(testset)  # 按照数据集的顺序依次采样，而不进行任何打乱

    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
