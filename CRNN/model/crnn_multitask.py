import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, bidirectional=True):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=bidirectional)

    def forward(self, input):
        output, _ = self.rnn(input)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [32, 32, 64, 64, 64, 64, 256]  # /2

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0, True)
        convRelu(1, True)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3, True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(4, True)
        convRelu(5, True)

        self.cnn = cnn
        self.spp1 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.spp2 = nn.MaxPool2d((3, 2), (2, 1), (0, 1))
        self.spp3 = nn.MaxPool2d((3, 2), (2, 1), (0, 1))

        self.rnn = nn.Sequential(
            BidirectionalLSTM(640, nh, bidirectional=True),
            BidirectionalLSTM(nh * 2, nh, bidirectional=True))

        self.text_cls = nn.Linear(nh * 2, nclass[0])
        self.color_cls = nn.Linear(nh * 2, nclass[1])

    def forward(self, input):
        # conv features
        conv = self.cnn(input)

        spp1 = self.spp1(conv)
        b, c, h, w = spp1.size()
        spp1 = spp1.reshape(b, -1, w)  # 128,4,23
        spp2 = self.spp2(conv).reshape(b, -1, w)  # 128,3,23
        spp3 = self.spp3(conv).reshape(b, -1, w)  # 128,2,23

        out = torch.cat([spp1, spp2, spp3], 1).permute(2, 0, 1)  # 1280

        # rnn features
        out = self.rnn(out)
        out_text = self.text_cls(out)
        out_color = self.color_cls(out)

        # add log_softmax to converge output
        out_text = F.log_softmax(out_text, dim=2)
        out_color = F.log_softmax(out_color, dim=2)

        return out_text, out_color

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0  # replace all nan/inf in gradients to zero


if __name__ == "__main__":
    import torch
    from torchsummary import summary

    model = CRNN(imgH=32, nc=3, nclass=355, nh=128, leakyRelu=False)
    model(torch.rand(1, 3, 32, 90))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #summary(model.to(device), (3, 32, 90))
    print(model)
    torch.save(model, "test.pt")
