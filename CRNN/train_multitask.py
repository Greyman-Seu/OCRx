import argparse
import logging
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import crnn_multitask as crnn
import utils
import dataset
import configure as cfg


class Trainer:
    def __init__(self, trainroot, testroot, alpha=0.85):
        self.trainroot = trainroot
        self.testroot = testroot

        # ensure everytime the random is the same
        random.seed(cfg.manualSeed)
        np.random.seed(cfg.manualSeed)
        torch.manual_seed(cfg.manualSeed)

        self.logger = self.setLogger()
        self.alpha = alpha
        self.train_loss = []
        self.test_loss = []
        self.test_loss_t = []
        self.test_loss_c = []
        self.acc = {'acc': [], 'acc_t': [], 'acc_c': []}
        self.device = torch.device(
            "cuda" if cfg.use_cuda and torch.cuda.is_available() else "cpu")
        self.model = self.net_init().to(self.device)
        if cfg.label_type == "both":
            self.codec = dataset.Codec(cfg.alphabet[0])
            self.codec_color = dataset.Codec(cfg.alphabet[1])
        else:
            self.codec = dataset.Codec(cfg.alphabet)
        self.loss_fn = nn.CTCLoss() if not cfg.dealwith_lossnan else nn.CTCLoss(zero_infinity=True)
        self.loss_fn = self.loss_fn.to(self.device)
        if cfg.adam:
            self.optim = optim.Adam(self.model.parameters(), lr=cfg.lr,
                                    betas=(cfg.beta1, 0.999))
        elif cfg.adadelta:
            self.optim = optim.Adadelta(self.model.parameters())
        else:
            self.optim = optim.RMSprop(self.model.parameters(), lr=cfg.lr)

    def setLogger(self):
        logger = logging.getLogger("Trainer")
        logger.setLevel(logging.DEBUG)
        c_handler = logging.StreamHandler()
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        f_handler = logging.FileHandler(
            os.path.join(parent_dir, 'log/train.log'), 'w')
        c_handler.setLevel(level=logging.INFO)
        f_handler.setLevel(level=logging.DEBUG)
        format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(format)
        f_handler.setFormatter(format)
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        return logger

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def net_init(self):
        self.logger.info(f"Channel of image is {cfg.nc}.")
        nClass = [l+1 for l in map(len, cfg.alphabet)] if isinstance(
            cfg.alphabet, tuple) else len(cfg.alphabet) + 1
        model = crnn.CRNN(cfg.imgH, cfg.nc, nClass, cfg.nh)
        model.apply(self.weights_init)
        if cfg.pretrained != '':
            self.logger(f"Load pretrained model from {cfg.pretrained}.")
            if cfg.multi_gpu:
                model = nn.DataParallel(model)
            model.load_state_dict(torch.load(cfg.pretrained))
        return model

    def dataloader(self):
        # train_loader
        train_dataset = dataset.lmdbDataset(
            root=self.trainroot, label_type=cfg.label_type, channel=cfg.nc)  # zyk: cfg.nc 为图通道
        # zyk: 确定下图的通道输出是否为nc
        assert cfg.nc == np.asarray(train_dataset[0][0]).shape[2]
        collate = dataset.AlignCollate(
            imgH=cfg.imgH, imgW=cfg.imgW, keep_ratio=cfg.keep_ratio, crop=cfg.crop)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batchSize,
                                  shuffle=True, num_workers=int(cfg.workers),
                                  collate_fn=collate)

        # test_loader
        test_dataset = dataset.lmdbDataset(
            root=self.testroot, label_type=cfg.label_type, transform=dataset.ResizeNormalize((cfg.imgW, cfg.imgH), crop=cfg.crop))
        test_loader = DataLoader(test_dataset, shuffle=False,
                                 batch_size=2*cfg.batchSize, num_workers=int(cfg.workers))

        return train_loader, test_loader

    def train(self, batch):
        self.model.train()
        # read
        image, label = batch
        label, label_color = zip(*[ele.split('_') for ele in label])

        # encode
        label, length = self.codec.encode(label)
        label_color, length_color = self.codec_color.encode(label_color)

        image = image.to(self.device)
        label = label.to(self.device)
        label_color = label_color.to(self.device)

        self.optim.zero_grad()
        pred, pred_c = self.model(image)
        pred_size = torch.LongTensor([pred.size(0)] * pred.size(1))
        pred_size_c = torch.LongTensor(
            [pred_c.size(0)] * pred_c.size(1))
        # Note only pred and label are in cuda

        loss_text = self.loss_fn(pred, label, pred_size, length)
        loss_color = self.loss_fn(
            pred_c, label_color, pred_size_c, length_color)
        loss = self.alpha*loss_text + (1-self.alpha)*loss_color
        loss.backward()
        self.optim.step()
        self.train_loss.append(loss.item())
        return loss

    def test(self, test_loader):
        self.logger.info(
            "The model is running on the test set, please wait ...")
        self.model.eval()

        counter = torch.zeros(2, dtype=torch.float, requires_grad=False)
        counter_t = torch.zeros(4, dtype=torch.float, requires_grad=False)
        counter_c = torch.zeros(4, dtype=torch.float, requires_grad=False)
        loss_t = loss_c = 0
        with torch.no_grad():
            for batch in test_loader:
                image, label_str = batch
                image = image.to(self.device)
                pred_t, pred_c = self.model(image)
                label_str_t, label_str_c = zip(*[ele.split('_')
                                                 for ele in label_str])
                check_t = self.checkBatch(pred_t, label_str_t, self.codec)
                check_c = self.checkBatch(
                    pred_c, label_str_c, self.codec_color)

                counter_t += check_t[0]
                counter_c += check_c[0]
                loss_t += check_t[1]
                loss_c += check_c[1]

                # simple prediction
                sp = [sp_t + '_' + sp_c for sp_t,
                      sp_c in zip(check_t[2], check_c[2])]
                assert counter_t[0] == counter_c[0]
                for p, t in zip(sp, label_str):
                    if p == t:
                        counter[1] += 1
        batch_size = image.size(0)
        pred_size_t = torch.LongTensor([pred_t.size(0)] * batch_size)
        _, pred_t = pred_t.max(2)
        pred_t = pred_t.transpose(1, 0).contiguous().view(-1)
        pred_size_c = torch.LongTensor([pred_c.size(0)] * batch_size)
        _, pred_c = pred_c.max(2)
        pred_c = pred_c.transpose(1, 0).contiguous().view(-1)
        # raw prediction text/color
        rpt = self.codec.decode(pred_t, pred_size_t, raw=True)
        rpc = self.codec_color.decode(pred_c, pred_size_c, raw=True)
        sampler_ind = random.sample(range(batch_size), cfg.nTestDisplay)
        for i in sampler_ind:
            self.logger.info("%-26s >>> %-26s, GT: %s" %
                             (rpt[i], check_t[2][i], label_str_t[i]))
            self.logger.info("%-26s >>> %-26s, GT: %s\n" %
                             (rpc[i], check_c[2][i], label_str_c[i]))
        counter[0] = counter_t[0]
        loss_t /= (counter[0]).item()  # average loss
        loss_c /= (counter[0]).item()  # average loss
        loss = self.alpha*loss_t + (1-self.alpha)*loss_c

        acc = (counter[1]/counter[0]).item()
        acc_t = (counter_t[1]/counter_t[0]).item()
        acc_c = (counter_c[1]/counter_c[0]).item()
        for k, v in zip(self.acc.keys(), [acc, acc_t, acc_c]):
            self.acc[k].append(v)
        acc_t_unit = (counter_t[3]/counter_t[2]).item()
        acc_c_unit = (counter_c[3]/counter_c[2]).item()
        self.logger.info("Test loss: %f, Accuracy: %f" % (loss, acc))
        self.logger.info("Test loss of text: %f, Accuracy: %f, AccuracyUnit: %f" %
                         (loss_t, acc_t, acc_t_unit))
        self.logger.info("Test loss of color: %f, Accuracy: %f, AccuracyUnit: %f\n" %
                         (loss_c, acc_c, acc_c_unit))

        self.test_loss.append(loss)
        self.test_loss_t.append(loss_t)
        self.test_loss_c.append(loss_c)
        return acc

    def checkBatch(self, pred, label_str, codec):
        batch_size = pred.size(1)
        pred_size = torch.LongTensor([pred.size(0)] * batch_size)

        # check loss
        label, length = codec.encode(label_str)
        label = label.to(self.device)
        loss = batch_size * self.loss_fn(pred, label, pred_size, length).item()

        # check predictions
        _, pred = pred.max(2)
        pred = pred.transpose(1, 0).contiguous().view(-1)
        simple_pred = codec.decode(pred, pred_size, raw=False)
        nTotal = nCorrect = nTotalUnit = nCorrectUnit = 0
        for p, t in zip(simple_pred, label_str):  # p: prediction, t: target
            if p == t:
                nCorrect += 1
            else:
                nTotal += 1
            nTotalUnit += len(t)
            # NOTE `zip` always truncates the longer sequence
            for p_char, t_char in zip(p, t):
                if p_char == t_char:
                    nCorrectUnit += 1
        nTotal += nCorrect
        counter = torch.tensor(
            [nTotal, nCorrect, nTotalUnit, nCorrectUnit], dtype=torch.int)
        return counter, loss, simple_pred

    def plot(self):
        plt.style.use('seaborn')
        fig = plt.figure()
        x_train = [i+1 for i in range(len(self.train_loss))]
        x_test = [(i+1)*cfg.testInterval for i in range(len(self.test_loss))]
        plt.plot(x_train, self.train_loss, '--g', label="Train Loss")
        plt.plot(x_test, self.test_loss, '--r', label="Test Loss")
        plt.xlabel("Iter Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("log/loss.png")
        return fig

    def main(self):
        if not os.path.exists(cfg.weight_dir):
            os.makedirs(cfg.weight_dir)

        torch.backends.cudnn.benchmark = True

        if torch.cuda.is_available() and not cfg.use_cuda:
            self.logger.warning(
                "You have a CUDA device, so you should probably set cuda in cfg.py to True")

        train_loader, test_loader = self.dataloader()
        self.logger.critical(f"The model structer is:\n{self.model}")

        if cfg.use_cuda and cfg.multi_gpu and torch.cuda.is_available():
            self.model = nn.DataParallel(
                self.model, device_ids=range(cfg.ngpu))

        ACC = 0
        batchNo = 0  # batch Number
        for epoch in range(cfg.nepoch):
            for i, batch in enumerate(train_loader):
                loss = self.train(batch)
                batchNo += 1
                if (i + 1) % cfg.displayInterval == 0 or i + 1 == len(train_loader):
                    self.logger.info('[%d/%d][%d/%d] Loss: %f' %
                                     (epoch, cfg.nepoch, i+1, len(train_loader), loss.item()))
                if batchNo % cfg.testInterval == 0:
                    acc = self.test(test_loader)
                    if acc > ACC:
                        torch.save(self.model.state_dict(),
                                   os.path.join(cfg.weight_dir, "checkpoint.pt"))
                        ACC = acc
        self.plot()
        return


if __name__ == "__main__":
    # single task
    path_10w = "/home/chuan/dataset/captcha/raw_data/lmdb"

    # improved 1
    path_31w_red = "/home/chuan/dataset/captcha/red_data/lmdb/"

    lmdb_path = path_10w

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trainroot', default=os.path.join(lmdb_path, "train"),
                        help='path to train dataset')
    parser.add_argument('-e', '--testroot', default=os.path.join(lmdb_path, "test"),
                        help='path to test dataset')
    args = parser.parse_args()

    # cfg.use_cuda = False
    try:
        trainer = Trainer(args.trainroot, args.testroot)
        trainer.main()
    except KeyboardInterrupt:
        trainer.plot()
