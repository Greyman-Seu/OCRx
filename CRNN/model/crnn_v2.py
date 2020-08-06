import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # chuan: 尝试用PyTorch自带的nn.Embedding?
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=2, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        if downsample == "justH":
            self.downsample = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        else:
            self.downsample = nn.MaxPool2d(2, 2)

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.conv = conv3x3(3, 32)
        self.stage1 = BasicBlock(32, 32)
        self.stage2 = BasicBlock(64, 64)
        #self.stage3 = BasicBlock(128, 128)



    def forward(self, input):
        # conv features
        conv = self.conv(input)
        conv = self.stage1(conv)
        conv = self.stage2(conv)
        conv = self.stage3(conv)
        # b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        # conv = conv.squeeze(2)
        # conv = conv.permute(2, 0, 1)  # [w, b, c] chuan: width, batch, chaneel
        return conv

        # # rnn features
        # output = self.rnn(conv)
        #
        # # add log_softmax to converge output
        # output = F.log_softmax(output, dim=2)
        #
        # return output

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero


if __name__ == "__main__":
    import torch

    model = CRNN(imgH=32, nc=3, nclass=355, nh=256)

    from torchsummary import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    summary(model.to(device), (3, 32, 90), device=device)

    # img = torch.rand((3, 3, 32, 90))
    # model(img)
    # print(model(img)[0].size(), model(img)[1].size())
