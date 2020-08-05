import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
import collections


class StrLabelConverter:
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Parameters:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet: str, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index
        # NOTE: 0 is reserved for 'blank' required by wrap_ctc
        # 这里用的是alphabet而不是self.alphabet，没有最后的'-'
        self.dict = {char: i+1 for i, char in enumerate(alphabet)}
        # NOTE:注意编码使用的是self.dict() blank位于0位
        # self.alphabet 解码中 - 放在了最后一位

    def encode(self, text_seq: list):
        """Encode a sequence (batch) of texts to a sequence of codes.

        Parameters:
            text_seq (list of bytes str): texts to convert.

        Returns:
            codes (2d-torch.LongTensor, batch_size*max_len): encoded texts.
            lens (1d-torch.LongTensor): lengths of each text.
        """

        codes = []
        lens = []
        for text in text_seq:
            text = text.decode('utf-8', 'strict')
            lens.append(len(text))
            code = [self.dict[char] for char in text]
            codes.append(torch.LongTensor(code))
        codes = pad_sequence(codes, batch_first=True, padding_value=0)  # label进行padding，变为等长
        lens = torch.LongTensor(lens)
        return (codes, lens)

    def decode(self, codes, lens, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text_seq.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text_seq (str or list of str): texts to convert.
        """
        if lens.numel() == 1:  # chuan: total number of elements in a tensor
            lens = lens.item()
            assert codes.numel() == lens, "text_seq with lens: {} does not match declared lens: {}".format(
                codes.numel(), lens)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in codes])
            else:
                char_list = []
                for i in range(lens):  # 全序列转换
                    # chuan: ???
                    if codes[i] != 0 and (not (i > 0 and codes[i - 1] == codes[i])):
                        # 预测不能为0，且该字符不能和上一字符相等
                        char_list.append(self.alphabet[codes[i] - 1])
                        # codes[i] - 1 这边为了弥补 把-放置在最后一位的
                return ''.join(char_list)
        else:  # batch mode (codes: (batch_size*26,))
            assert codes.numel() == lens.sum(
            ), "texts codes with length: {} does not match declared length: {}".format(codes.numel(), lens.sum())
            texts = []
            index = 0
            for i in range(lens.numel()):  # range(batch)
                length = lens[i]
                # chuan: 递归666
                texts.append(self.decode(
                    codes[index:index + length], torch.LongTensor([length]), raw=raw))
                index += length
            return texts


class Averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img


class Logger:
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()  # 每次写入后刷新到文件中，防止程序意外结束

    def flush(self):
        self.log.flush()
