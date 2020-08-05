import sys
import six
import random
from PIL import Image

import lmdb
import numpy as np
import cv2  # chuan

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms


class lmdbDataset(Dataset):
    """chuan: 把一张图片和其标签从lmdb中读出来。"""

    def __init__(self, root=None, channel=3, transform=None, target_transform=None):
        self.env = lmdb.open(root,
                             max_readers=1,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)

        self.channel = channel

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)  # chuan: 意义？

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode('utf-8')))
            self.nSamples = nSamples  # zyk:所有样本数

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                # zyk
                if self.channel == 3:
                    img = Image.open(buf)
                else:
                    img = Image.open(buf).convert('L')  # chuan:'L'表示转换为灰度图

            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]  # chuan: 这是什么操作？

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode('utf-8'))

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


# class lmdbDataset0(Dataset):
#     """chuan: 把一张图片和其标签从lmdb中读出来。"""
#
#     def __init__(self, root=None, transform=None, target_transform=None):
#         self.images = np.load(root+'/'+'images.npy')
#         self.labels = np.load(root+'/'+'labels.npy')
#
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __len__(self):
#         assert len(self.images) == len(self.labels)
#         return len(self.images)
#
#     def __getitem__(self, index):
#         img = self.images[index]
#         label = self.labels[index]
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             label = self.target_transform(label)
#         return (img, label)


class ResizeNormalize:
    def __init__(self, size, interpolation=Image.BILINEAR, crop=True):
        self.size = size
        self.interpolation = interpolation
        self.crop = crop
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        if self.crop:
            img = self.toTensor(img)  # PIL -> 0:1
            img = img[:,1:-2, :] if random.randint(0, 1) else img[:,2:-1, :]
            img.sub_(0.5).div_(0.5)  # 0:1 -> -1:1
        else:
            img = img.resize(self.size, self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)  # chuan: why
        return img


class RandomSequentialSampler(Sampler):
    """随机顺序的采样索引。"""

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class AlignCollate(object):
    """chuan: 
    align：v. 排整齐；校准；使成一条直线；
    collate: v. 核对，校勘"""

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1,crop=True):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.crop = crop
    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []  # chuan: 宽高比
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))  # chuan: 得到最大的宽
            imgW = max(imgH * self.min_ratio, imgW)  # 宽不得小于高，最不济也得是个正方形

        transform = ResizeNormalize((imgW, imgH),self.crop)
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        # zyk:
        # keepratio:可能会出现每个batch的宽度不同，直接导致最后的T不一致，查看model
        return images, labels
