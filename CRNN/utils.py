import os
import shutil
import logging
import sys
import time

import numpy as np
import pandas as pd
import cv2 as cv

import torch


def labelDistribution(srcDir):
    """Count the distribution of labels of images in a source director.

    Returns:
    --------
    chars(pd.DateFrame)
    colors(pd.DateFrame)
    """
    files = os.listdir(srcDir)
    chars, colors = [], []
    for file in files:
        if file.endswith('.png'):
            _, label, color = file.split('_')
            chars.extend(label)
            colors.extend(color[:-4])

    # process chars
    record = {}
    for char in chars:
        if char in record:
            record[char] += 1
        else:
            record[char] = 1
    keys = np.array(list(record.keys()))
    values = np.array(list(record.values()))
    sorted_index = np.argsort(values)
    keys = keys[sorted_index]
    values = values[sorted_index]
    chars = pd.DataFrame({'character': keys, 'frequency': values})

    # process colors
    record = {}
    for color in colors:
        if color in record:
            record[color] += 1
        else:
            record[color] = 1
    keys = np.array(list(record.keys()))
    values = np.array(list(record.values()))
    sorted_index = np.argsort(values)
    keys, values = keys[sorted_index], values[sorted_index]
    colors = pd.DataFrame({'color': keys, 'frequency': values})

    return chars, colors


class ImageChannelTransformer:
    def __init__(self, sourceDir, dstDir, digit=8):
        self.src = os.path.abspath(sourceDir)
        self.dst = os.path.abspath(dstDir)
        self.digit = digit
        self.COUNT = 0
        self.logger = self.setLogger()

        if os.path.exists(self.dst):
            shutil.rmtree(self.dst)
            os.makedirs(self.dst)
            self.logger.warning(
                f"Path {self.dst} already exists, the process deletes it and reconstructs a new one.")
        else:
            os.makedirs(self.dst)

    def __call__(self):
        self.main()

    def setLogger(self):
        logger = logging.getLogger("ImageChannelTransformer")
        logger.setLevel(logging.DEBUG)
        c_handler = logging.StreamHandler()
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        f_handler = logging.FileHandler(os.path.join(
            parent_dir, 'log/ImageChannelTransformer.log', 'w'))
        c_handler.setLevel(level=logging.INFO)
        f_handler.setLevel(level=logging.INFO)
        c_format = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        return logger

    def transImage(self, file):
        """Transform a single image."""

        # 读入原图
        # cv 中默认读入图像是(B,G,R)
        image = cv.imread(os.path.join(self.src, file))
        red2red_name = self.transName(file, '红', '红')  # 转换原图名称

        # 蓝色转红色
        # 蓝色R、G值较小，B值独大，B、G交换，接近红色分布
        blue2red = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        blue2red_name = self.transName(file, '蓝', '红')

        # 黑色转红色
        # 黑色三个通道都比较小， 用255-R通道，则R通道独大，其余通道小，接近红色的分布
        black2red = image.copy()
        black2red[:, :, 2] = 255 - black2red[:, :, 2]
        black2red_name = self.transName(file, '黑', '红')

        # 黄色转红色
        # 黄色R、G值较大，B值较小，先取反，则R、G值较小, B值独大，接近蓝色分布
        yellow2red = image.copy()
        yellow2red = cv.bitwise_not(image)
        yellow2red = cv.cvtColor(yellow2red, cv.COLOR_RGB2BGR)  # 与蓝色转红色相同
        yellow2red_name = self.transName(file, '黄', '红')

        # 写入结果
        names = [red2red_name, blue2red_name, black2red_name, yellow2red_name]
        imgs = [image, blue2red, black2red, yellow2red]
        for name, img in zip(names, imgs):
            if name is not None:
                self.COUNT += 1
                newName = f"{self.COUNT:0>{self.digit}d}-{name}"
                path = os.path.join(self.dst, newName)
                cv.imwrite(path, img)
                self.logger.info(f"Write {self.COUNT} transformed images.")
        return

    def transName(self, filename: str, origin: str, target: str) -> str:
        """Get the transformed name of an image.

        Parameters:
        -----------
        filename: name of the input image
        origin: original color flag
        target: target color flag
        """
        filename = filename.split('-')[-1]  # remove indexNo
        labels, colors = filename.split('_')
        indices = [i for i, color in enumerate(colors) if color == origin]
        if len(indices) == 0:
            return None

        res = []
        for i in indices:
            res.append(labels[i])
        res.append('_')
        for _ in indices:
            res.append(target)
        res.extend(colors[-4:])  # add ".png"
        res = ''.join(res)
        return res

    def main(self):
        filenames = os.listdir(self.src)
        filenames = [
            filename for filename in filenames if filename.endswith('.png')]
        filenames.sort()
        for filename in filenames:
            self.transImage(filename)

        actual_count = len(os.listdir(self.dst))
        warning = f"Expect {self.COUNT} pictures, write only {actual_count}!"
        assert self.COUNT == actual_count, warning
        return


def infer(model, test_loader, device, codec):
    model.eval()
    # maker logger
    logger = logging.getLogger("infer")
    logger.setLevel(logging.DEBUG)
    # add file handler
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    file_hdlr = logging.FileHandler(os.path.join(
        parent_dir, 'log/inference.log'), 'w')
    file_hdlr.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    file_hdlr.setLevel(logging.INFO)
    logger.addHandler(file_hdlr)
    # add stream handler
    stream_hdlr = logging.StreamHandler()
    stream_hdlr.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    stream_hdlr.setLevel(logging.INFO)
    logger.addHandler(stream_hdlr)
    with torch.no_grad():
        for image, label_text in test_loader:
            label, _ = codec.encode(label_text)
            image = image.to(device)
            label = label.to(device)

            pred = model(image)
            batch_size = image.size(0)
            pred_size = torch.LongTensor([pred.size(0)] * batch_size)

            _, pred = pred.max(2)
            pred = pred.transpose(1, 0).contiguous().view(-1)
            simple_pred = codec.decode(pred, pred_size, raw=False)
            raw_pred = codec.decode(pred, pred_size, raw=True)
            for rp, p, t in zip(raw_pred, simple_pred, label_text):
                logger.info("%-26s >>> %-26s, GT: %s" % (rp, p, t))

        sample = image[0].unsqueeze(0)
        begin = time.time()
        n_test = 100
        for _ in range(n_test):
            model(sample)
        time_cost = (time.time() - begin) / n_test
        time_cost = round(time_cost, 4)
        fps = 1 // time_cost
        logger.info(
            f"Inference Time Cost: {time_cost} sec/image; FPS: {fps:.0f}")


def readInferLog(filename, flag1=">>>", flag2="GT:"):
    """Resolve the inference log file.

    Returns:
    --------
    log (pd.DataFrame): inference log
    """
    preds, gts = [], []
    with open(filename, 'r') as f:
        template = f.readline()
        size1, size2 = len(flag1), len(flag2)
        ind1 = template.find(flag1) + size1 + 1
        ind2 = template.find(flag2) + size2 + 1
        assert ind1 != size1 or ind2 != size2, f"Log file error. Unable to find flag {flag1} and {flag2} both."
        f.seek(0)
        for line in f:
            pred = line[ind1:ind2-size2-3].rstrip()  # [ind1, ',')
            gt = line[ind2:-1].rstrip()  # [ind2, '\n')
            preds.append(pred)
            gts.append(gt)
    # pop the last line
    gts.pop()
    preds.pop()
    log = pd.DataFrame({'GroundTruth': gts, 'Predictions': preds})
    return log


class StatisticalInfo:
    """Statistics features of CRNN model inference."""

    def __init__(self, log: pd.DataFrame, alphabet: str):
        # chuan: consider combining with dataset.Codec
        self.alphabet = alphabet
        self.char_hash = {char: i for i, char in enumerate(alphabet)}
        self.log = log
        self.confusion, self.diff, self._mappings = self.base()

    def base(self):
        """Basic calculations for general statistics features.
        """
        gts = self.log["GroundTruth"]
        preds = self.log["Predictions"]

        diff_len = {i for i in self.log.index if len(gts[i]) != len(preds[i])}

        # 与字符表中字符的顺序保持一致
        mappings = tuple({} for _ in self.alphabet)
        for i, (gt, pred) in enumerate(zip(gts, preds)):
            if i in diff_len:
                continue
            for y, x in zip(gt, pred):
                ind = self.char_hash[y]
                mapping = mappings[ind]
                if x not in mapping:
                    mapping[x] = 1
                else:
                    mapping[x] += 1

        # get the confusion matrix
        header = list(self.alphabet)
        confusion = pd.DataFrame(mappings, index=header, columns=header)
        confusion.index.names = ["GroundTruth"]
        confusion.columns.names = ["Prediction"]

        # get items with different lengths of prediction and groundtruth
        diff = pd.DataFrame([(gts[i], preds[i]) for i in diff_len],
                            columns=['gts', 'preds'])

        return confusion, diff, mappings

    @property
    def acc_total(self):
        return (self.log.iloc[:, 0] == self.log.iloc[:, 1]).sum() / len(self.log)

    @property
    def acc_unit(self):
        nCorrect = nWrong = 0
        for (gt, pred) in self.log.values:
            for t, p in zip(gt, pred):
                if t == p:
                    nCorrect += 1
                else:
                    nWrong += 1
        return nCorrect / (nCorrect + nWrong)

    @property
    def acc_table(self):
        res = pd.Series(np.diag(self.confusion) / self.confusion.sum(axis=1),
                        index=self.confusion.index, name='Accuracy')
        res.sort_values(inplace=True)
        return res

    def check_char(self, char: str):
        assert char in self.char_hash, f"Character {char} is not in the alphabet!"
        ind = self.char_hash[char]
        res = self._mappings[ind]
        res['accuracy'] = res[char] / sum(res.values())
        res = pd.Series(res, name="Frequence/Accuracy")
        res.index.names = ["Predictions"]
        res.sort_values(ascending=False, inplace=True)
        return res


class PrintLogger:
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()  # 每次写入后刷新到文件中，防止程序意外结束

    def flush(self):
        self.log.flush()
