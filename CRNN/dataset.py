import os
import shutil
import sys
import six
import random
import logging
from PIL import Image

import lmdb
import numpy as np
import cv2 as cv

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms


class LMDBMaker:
    """Compress pictures from a folder into a Lightning Memory-Mapped Database.

        NOTE:
        --------
        1. The pictures in folder should be named in the format of
            `indexNo-textLabel_colorLabel.png`, e.g., `00000001-大小王_红黑蓝.png`.
        2. The indexNo should never be duplicate or missing.
        3. The indexNo should start with 1 rather than 0.
    """

    def __init__(self, sourceFolder: str, outputFolder: str, digit=8):
        self.src = os.path.abspath(sourceFolder)
        self.dst = os.path.abspath(outputFolder)
        self.digit = digit
        self.logger = self.setLogger()

    def __call__(self):
        indices, labels = self.readFromFolder()
        self.createDataset(indices, labels)
        self.showDemo()

    def setLogger(self):
        logger = logging.getLogger("LMDBMaker")
        logger.setLevel(logging.DEBUG)
        c_handler = logging.StreamHandler()
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        f_handler = logging.FileHandler(
            os.path.join(parent_dir, 'log/LMDBMaker.log'), 'w')
        c_handler.setLevel(level=logging.INFO)
        f_handler.setLevel(level=logging.WARNING)
        c_format = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        return logger

    def readFromFolder(self):
        files = os.listdir(self.src)
        # select the picture files
        files = [file for file in files if file.endswith(".png")]
        files.sort()

        # chuan: bad implementation
        indices = [file.split('-')[0] for file in files]
        labels = [file.split('-')[1][:-4] for file in files]

        return indices, labels

    def createDataset(self, indices, labels, check=True):
        """Create LMDB from src into dst for CRNN training.

        Parameters:
        -----------
        indices (list): list of int indices for images in imagePaths
        labels (list): list of corresponding groundtruth `textLabel_colorLabel`
        check (bool): if true, check the validity of every image
        """

        assert len(indices) == len(
            labels), "Number of indices and labels is different!"
        nSamples = len(indices)

        # If lmdb file already exists, remove it. Or the new data will add to it.
        if os.path.exists(self.dst):
            shutil.rmtree(self.dst)
            self.logger.warning(
                f"Path {self.dst} already exists, the process deletes it and reconstructs a new one.")
            os.makedirs(self.dst)
        else:
            os.makedirs(self.dst)

        self.logger = logging.getLogger("LMDBMaker")
        env = lmdb.open(self.dst, map_size=1099511627776)  # 1024**4, 1TB
        cache = {"src": self.src,
                 "nSamples": str(nSamples)}
        # digit = len(str(nSamples))
        for i in range(nSamples):
            index = indices[i]
            label = labels[i]
            imagePath = os.path.join(
                self.src, self.nameNormalization(index, label))
            with open(imagePath, 'rb') as f:
                imageBin = f.read()
            if check and not self.checkImageIsValid(imageBin):
                self.logger.warning(f"{imagePath} is not a valid image.")
                continue

            index = int(index)
            imageKey = f"image-{index:0>{self.digit}d}"
            labelKey = f"label-{index:0>{self.digit}d}"
            cache[imageKey] = imageBin
            cache[labelKey] = label
            if (i+1) % 1000 == 0 or i + 1 == nSamples:
                self.writeCache(env, cache)
                cache = {}

                self.logger.info("Written %d / %d" % (i+1, nSamples))
        self.writeCache(env, cache)
        env.close()
        self.logger.info(
            f"Created lmdb dataset with {nSamples} samples, done.")
        return

    def nameNormalization(self, index: str, label: str):
        return f"{index}-{label}.png"

    def checkImageIsValid(self, imageBin: bytes):
        if imageBin is None:
            return False
        try:
            imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
            img = cv.imdecode(imageBuf, cv.IMREAD_COLOR)  # cv.IMREAD_GRAYSCALE
            imgH, imgW = img.shape[0], img.shape[1]
        except:
            return False
        else:
            if imgH * imgW == 0:
                return False
        return True

    def writeCache(self, env, cache):
        """Write the data in cache into a lmdb environment.

        Parameters:
        -----------
        env: Environment of lmdb.
        cache (dict): a dict to be written into env.
        """
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                # chuan: encode str to bytes
                if type(k) == str:
                    k = k.encode()
                if type(v) == str:
                    v = v.encode()
                txn.put(k, v)

    def showDemo(self, num=3):
        self.logger.info("Jump a line.\n")
        self.logger.info("Here are some demos created in LMDB.")
        self.logger.info(
            "First line shows a path, second line shows corresponding label.")
        with lmdb.open(self.dst) as env:
            with env.begin(write=False) as txn:
                nSamples = txn.get("nSamples".encode())
                # digit = len(nSamples)
                src = txn.get("src".encode()).decode()
                indices = np.random.randint(1, int(nSamples)+1, size=num)
                for index in indices:
                    index = f"{index:0>{self.digit}d}"
                    labelKey = f"label-{index}".encode()
                    label = txn.get(labelKey).decode()
                    name = self.nameNormalization(index, label)
                    self.logger.info(os.path.join(src, name))
                    self.logger.info(label)
        return


class lmdbDataset(Dataset):
    """Read an image and its label from created lmdb dataset.

    Returns:
    --------
    img (PIL.Image or Tensor): if self.transform is None, img is PIL.Image
    label (bytes): label of img
    """

    def __init__(self, root=None, label_type="text", channel=3, digit=8, transform=None):
        self.env = lmdb.open(root,
                             max_readers=1,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        assert label_type in {"text", "color",
                              "both"}, "Check your label type!"
        self.label_type = label_type
        self.channel = channel
        self.digit = digit
        self.transform = transform

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)  # chuan: 意义？

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('nSamples'.encode('utf-8')))
            self.nSamples = nSamples

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        index += 1
        with self.env.begin(write=False) as txn:
            imgKey = f"image-{index:0>{self.digit}d}"
            imgbuf = txn.get(imgKey.encode('utf-8'))

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:  # zyk
                if self.channel == 3:  # RGB image
                    img = Image.open(buf)
                else:  # grayscale image
                    img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]  # chuan: 这是什么操作？

            if self.transform is not None:
                img = self.transform(img)

            labelKey = f"label-{index:0>{self.digit}d}"
            label = txn.get(labelKey.encode('utf-8'))
            label = label.decode('utf-8', 'strict')
        if self.label_type == "text":
            label = label.split('_')[0]
        elif self.label_type == "color":
            label = label.split('_')[1]
        else:
            label = label
        return (img, label)


class ResizeNormalize:
    def __init__(self, size, crop=True, interpolation=Image.BILINEAR):
        from configure import manualSeed
        random.seed(manualSeed)
        self.size = size
        self.crop = crop
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        if self.crop:
            img = self.toTensor(img)  # PIL -> 0:1
            img = img[:, 1:-2, :] if random.randint(0, 1) else img[:, 2:-1, :]
            img.sub_(0.5).div_(0.5)  # 0:1 -> -1:1
        else:  # resize
            img = img.resize(self.size, self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)  # chuan: why
        return img


class AlignCollate(object):
    """chuan:
    align：v. 排整齐；校准；使成一条直线；
    collate: v. 核对，校勘"""

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1, crop=True):
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

        transform = ResizeNormalize((imgW, imgH), self.crop)
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        # zyk:
        # keepratio:可能会出现每个batch的宽度不同，直接导致最后的T不一致，查看model
        return images, labels


class Codec:
    """Encoder and Decoder between text label and mathmatical label.

    NOTE:
    --------
    Insert `blank` to the alphabet for CTC.

    Parameters:
    ----------
    alphabet (str): set of the possible characters.
    ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet: str, ignore_case=False):
        """
        NOTE:
        ------
        1) for self.dict, 0 is reserved for 'blank' required by wrap_ctc
        2) 注意编码使用的是self.dict(), blank位于0位
        3) self.alphabet 解码中 - 放在了最后一位
        """
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.dict = {char: i+1 for i, char in enumerate(alphabet)}
        self.alphabet = alphabet + '-'  # for `-1` index

    def encode(self, text_seq: list):
        """Encode a sequence (batch) of texts to a sequence of codes.

        Parameters:
        -----------
        text_seq (list of str): texts to be encoded into mathmatical tensor.

        Returns:
        --------
        codes (2d-torch.LongTensor, batch_size*max_len): encoded texts.
        lens (1d-torch.LongTensor): lengths of each text.
        """

        codes = []
        lens = []
        for text in text_seq:
            # text = text.decode('utf-8', 'strict')
            lens.append(len(text))
            code = [self.dict[char] for char in text]
            codes.append(torch.LongTensor(code))
        # pad the label codes to make equal length
        codes = pad_sequence(codes, batch_first=True, padding_value=0)
        lens = torch.LongTensor(lens)
        return (codes, lens)

    def decode(self, codes, lens, raw=False):
        """Decode encoded tensor of texts back into strs.

        Args:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text_seq.

        Returns:
            text_seq (str or list of str): texts to convert.

        Raises:
            AssertionError: when the texts and its length does not match.
        """
        texts = []  # return must be a sequence
        if lens.numel() == 1:  # chuan: total number of elements in a tensor
            lens = lens.item()
            assert codes.numel() == lens, "text_seq with lens: {} does not match declared lens: {}".format(
                codes.numel(), lens)
            if raw:
                texts.append(''.join([self.alphabet[i - 1] for i in codes]))
            else:
                char_list = []
                for i in range(lens):  # 全序列转换
                    # chuan: ???
                    if codes[i] != 0 and (not (i > 0 and codes[i - 1] == codes[i])):
                        # 预测不能为0，且该字符不能和上一字符相等
                        char_list.append(self.alphabet[codes[i] - 1])
                        # codes[i] - 1 这边为了弥补 把-放置在最后一位的
                texts.append(''.join(char_list))
            return texts
        else:  # batch mode (codes: (batch_size*26,))
            assert codes.numel() == lens.sum(
            ), "texts codes with length: {} does not match declared length: {}".format(codes.numel(), lens.sum())
            index = 0
            for i in range(lens.numel()):  # range(batch)
                length = lens[i]
                # chuan: 递归666
                texts.extend(self.decode(
                    codes[index:index + length], torch.LongTensor([length]), raw=raw))
                index += length
            return texts
