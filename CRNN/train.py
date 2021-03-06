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

from model import crnn_github as crnn
import utils
import dataset
import configure as cfg


class Trainer:
    def __init__(self, trainroot, testroot):
        self.trainroot = trainroot
        self.testroot = testroot

        # ensure everytime the random is the same
        random.seed(cfg.manualSeed)
        np.random.seed(cfg.manualSeed)
        torch.manual_seed(cfg.manualSeed)

        self.logger = self.setLogger()
        self.loss = {"train": [], "test": []}
        self.acc = {"acc": [], "acc_unit": []}
        self.device = torch.device(
            "cuda" if cfg.use_cuda and torch.cuda.is_available() else "cpu")
        self.model = self.net_init().to(self.device)
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
        if cfg.use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optim, milestones=cfg.milestones, gamma=cfg.gamma)

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
        nClass = len(cfg.alphabet) + 1
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
        image, label = batch  # label is str
        label, length = self.codec.encode(label)  # label is mathmatical code
        image = image.to(self.device)
        label = label.to(self.device)
        self.optim.zero_grad()
        pred = self.model(image)
        pred_size = torch.LongTensor([pred.size(0)] * pred.size(1))
        # Note only pred and label are in cuda
        loss = self.loss_fn(pred, label, pred_size, length)
        loss.backward()
        self.optim.step()
        self.loss["train"].append(loss.item())
        return loss

    def test(self, test_loader):
        self.logger.info(
            "The model is running on the test set, please wait ...")
        self.model.eval()

        nCorrect = nWrong = 0  # 上下两种写法究竟是否同效
        nTotalUnit, nCorrectUnit = 0, 0
        loss = 0
        with torch.no_grad():
            for image, label_str in test_loader:
                batch_size = image.size(0)
                label, length = self.codec.encode(label_str)
                image = image.to(self.device)
                label = label.to(self.device)

                pred = self.model(image)
                pred_size = torch.LongTensor([pred.size(0)] * batch_size)
                loss += batch_size * \
                    self.loss_fn(pred, label, pred_size, length).item()

                _, pred = pred.max(2)
                pred = pred.transpose(1, 0).contiguous().view(-1)
                simple_pred = self.codec.decode(pred, pred_size, raw=False)
                for p, t in zip(simple_pred, label_str):  # p: prediction, t: target
                    if p == t:
                        nCorrect += 1
                    else:
                        nWrong += 1
                    nTotalUnit += len(t)
                    # NOTE `zip` always truncates the longer sequence
                    for p_char, t_char in zip(p, t):
                        if p_char == t_char:
                            nCorrectUnit += 1
        raw_pred = self.codec.decode(pred, pred_size, raw=True)
        sample_ind = random.sample(range(batch_size), cfg.nTestDisplay)
        # rp, p, t: raw prediction, prediction, target
        for i in sample_ind:
            rp = raw_pred[i]
            p = simple_pred[i]
            t = label_str[i]
            self.logger.info("%-26s >>> %-26s, GT: %s" % (rp, p, t))

        loss /= (nCorrect + nWrong)  # average loss
        self.loss["test"].append(loss)
        acc = nCorrect / (nCorrect + nWrong)
        acc_unit = nCorrectUnit / nTotalUnit
        self.acc["acc"].append(acc)
        self.acc["acc_unit"].append(acc_unit)
        self.logger.info("\nTest loss: %f, Accuracy: %f, AccuracyUnit: %f\n" %
                         (loss, acc, acc_unit))
        return acc, acc_unit

    def plot(self):
        plt.style.use('seaborn')
        fig1, ax1 = plt.subplots()
        ax1.set(xlabel="Iter Step", ylabel="Loss")
        x_train = [i+1 for i in range(len(self.loss["train"]))]
        x_test = [(i+1)*cfg.testInterval for i in range(len(self.loss["test"]))]
        ax1.plot(x_train, self.loss["train"], '--g', label="Train Loss")
        ax1.plot(x_test, self.loss["test"], '--r', label="Test Loss")

        fig2, ax2 = plt.subplots()
        ax2.set(xlabel="Test Step", ylabel="Accuracy")
        x_acc = range(1, len(self.acc["acc"])+1)
        ax2.plot(x_acc, self.acc["acc"], label="acc")
        ax2.plot(x_acc, self.acc["acc_unit"], label="acc_unit")
        ax2.legend()
        plt.legend()
        fig1.savefig("log/loss.png")
        fig2.savefig("log/acc.png")
        return fig1, fig2

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
                    acc, acc_unit = self.test(test_loader)
                    if acc > ACC:
                        torch.save(self.model.state_dict(),
                                   os.path.join(cfg.weight_dir, "checkpoint.pt"))
                        ACC = acc
            if cfg.use_lr_scheduler:
                self.scheduler.step()
        self.plot()
        return


if __name__ == "__main__":
    path_10w = "/home/chuan/dataset/captcha/raw_data/lmdb"
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
