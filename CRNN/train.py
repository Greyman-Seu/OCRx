from __future__ import print_function, division

import argparse
import random
import os
import sys
import numpy as np
# from itertools import zip_longest

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import utils
import dataset
import params
import models.crnn as net

# single task
path_10w_text = "/home/chuan/dataset/captcha/other/lmdb_output/text"
path_10w_color = "/home/chuan/dataset/captcha/other/lmdb_output/color"

# multitasks
# path_10w_both = "/home/chuan/dataset/captcha/other/lmdb_output/both"

# improved 1
path_31w_red = "/home/chuan/dataset/captcha/red_data/lmdb_output/"  # 红字符
path_31w_rgb = "/home/chuan/dataset/captcha/red_data/lmdb_output/"  # 全字符

# improved 2


lmdb_path = path_31w_red

parser = argparse.ArgumentParser()
parser.add_argument('-train', '--trainroot', default=os.path.join(lmdb_path, "train"),  # required=True,
                    help='path to train dataset')
parser.add_argument('-val', '--valroot', default=os.path.join(lmdb_path, "val"),  # ,required=True,
                    help='path to val dataset')
args = parser.parse_args()

if not os.path.exists(params.expr_dir):
    os.makedirs(params.expr_dir)

# ensure everytime the random is the same
random.seed(params.manualSeed)
np.random.seed(params.manualSeed)
torch.manual_seed(params.manualSeed)

torch.backends.cudnn.benchmark = True

if torch.cuda.is_available() and not params.cuda:
    print("WARNING: You have a CUDA device, so you should probably set cuda in params.py to True")


def data_loader():
    """Get train and val data_loader."""
    # train
    train_dataset = dataset.lmdbDataset(root=args.trainroot, channel=params.nc)  # zyk: params.nc 为图通道
    assert train_dataset
    assert params.nc == np.asarray(train_dataset[0][0]).shape[2]  # zyk 确定下图的通道输出是否为nc

    if not params.random_sample:
        sampler = dataset.RandomSequentialSampler(
            train_dataset, params.batchSize)
    else:
        sampler = None
    train_loader = DataLoader(train_dataset, batch_size=params.batchSize,
                              shuffle=True, sampler=sampler, num_workers=int(params.workers),
                              collate_fn=dataset.AlignCollate(imgH=params.imgH, imgW=params.imgW,
                                                              keep_ratio=params.keep_ratio, crop=params.crop))

    # val
    val_dataset = dataset.lmdbDataset(
        root=args.valroot, transform=dataset.ResizeNormalize((params.imgW, params.imgH), crop=params.crop))
    assert val_dataset
    val_loader = DataLoader(val_dataset, shuffle=False,
                            batch_size=2 * params.batchSize, num_workers=int(params.workers))

    return train_loader, val_loader


train_loader, val_loader = data_loader()


# -----------------------------------------------
# In this block
#     Net init
#     Weight init
#     Load pretrained model


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def net_init():
    nclass = len(params.alphabet) + 1

    # zyk
    print("channel of img is {}".format(params.nc))

    crnn = net.CRNN(params.imgH, params.nc, nclass, params.nh)
    crnn.apply(weights_init)
    if params.pretrained != '':
        print('loading pretrained model from %s' % params.pretrained)
        if params.multi_gpu:
            crnn = nn.DataParallel(crnn)
        crnn.load_state_dict(torch.load(params.pretrained))

    return crnn


crnn = net_init()
print(crnn)

# -----------------------------------------------
# In this block
#     Init some utils defined in utils.py

# Compute average for `torch.Variable` and `torch.Tensor`.
loss_avg = utils.Averager()

# Convert between str and label.
converter = utils.StrLabelConverter(params.alphabet)

# -----------------------------------------------
# In this block
#     criterion define

criterion = nn.CTCLoss()

# -----------------------------------------------
# In this block
#     Init some tensor
#     Put tensor and net on cuda
#     NOTE:
#         image, text, length is used by both val and train
#         becaues train and val will never use it at the same time.

image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
text = torch.LongTensor(params.batchSize * 5)
length = torch.LongTensor(params.batchSize)

if params.cuda and torch.cuda.is_available():
    criterion = criterion.cuda()
    image = image.cuda()
    text = text.cuda()

    crnn = crnn.cuda()
    if params.multi_gpu:
        crnn = nn.DataParallel(crnn, device_ids=range(params.ngpu))

# chuan: these are placeholders
image = Variable(image)
text = Variable(text)
length = Variable(length)

# -----------------------------------------------
# In this block
#     Setup optimizer
if params.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=params.lr,
                           betas=(params.beta1, 0.999))
elif params.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

# -----------------------------------------------
# In this block
#     Dealwith lossnan
#     NOTE:
#         I use different way to dealwith loss nan according to the torch version.
if params.dealwith_lossnan:
    if torch.__version__ >= '1.1.0':
        """
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.
        Pytorch add this param after v1.1.0 
        """
        criterion = nn.CTCLoss(zero_infinity=True)
    else:
        """
        only when
            torch.__version__ < '1.1.0'
        we use this way to change the inf to zero
        """
        crnn.register_backward_hook(crnn.backward_hook)


# -----------------------------------------------


def val(net, criterion):
    print('\nStart val:')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    val_iter = iter(val_loader)

    i = 0
    n_correct, n_correct_char = 0, 0
    total_char = 0
    loss_avg = utils.Averager()  # The blobal loss_avg is used by train

    max_iter = len(val_loader)
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        # chuan: nn.CTCloss已经默认了是mean模式，没必要再除以batch_size
        # cost = criterion(preds, text, preds_size, length) / batch_size
        cost = criterion(preds, text, preds_size, length)
        loss_avg.add(cost)

        # preds:23,128,322  T,B,C
        _, preds = preds.max(2)
        # preds: 23,128 ->128,23 ->2944
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        cpu_texts_decode = []
        for i in cpu_texts:
            cpu_texts_decode.append(i.decode('utf-8', 'strict'))
        for pred, target in zip(sim_preds, cpu_texts_decode):
            if pred == target:
                n_correct += 1

            total_char += len(target)
            # smaller_len = min(len(pred), len(target))
            for p_char, t_char in zip(pred, target):
                if p_char == t_char:
                    n_correct_char += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[
                :params.n_val_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * val_loader.__dict__['batch_size'])
    accuracy_char = n_correct_char / total_char
    print('\nVal loss: %f, accuracy: %f, accuracy_char: %f\n' %
          (loss_avg.val(), accuracy, accuracy_char))


def train(net, criterion, optimizer, train_iter):
    for p in crnn.parameters():
        p.requires_grad = True
    crnn.train()

    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)  # target
    utils.loadData(length, l)  # target length

    optimizer.zero_grad()
    preds = crnn(image)
    preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
    # chuan: nn.CTCloss已经默认了是mean模式，没必要再除以batch_size
    # cost = criterion(preds, text, preds_size, length) / batch_size
    cost = criterion(preds, text, preds_size, length)
    """
    preds: T,batch,Channel_output (23,64,5) 5分类问题
    text:batch,label_length_max  (最大的字符长度) shape=(64,6)  
    preds_size:batch shape(64,1) [23]*64
    
    """
    # crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


if __name__ == "__main__":
    # params.cuda = False
    sys.stdout = utils.Logger(params.log)
    for epoch in range(params.nepoch):
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            cost = train(crnn, criterion, optimizer, train_iter)
            loss_avg.add(cost)
            i += 1

            if i % params.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, params.nepoch, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

            if i % params.valInterval == 0:
                val(crnn, criterion)

        # do checkpointing
        if (epoch) % 5 == 0:
            torch.save(
                crnn.state_dict(), '{0}/netCRNN_{1}.pth'.format(params.expr_dir, epoch))
