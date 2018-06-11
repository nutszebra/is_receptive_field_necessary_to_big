import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn import Conv2d
try:
    from .prototype import NN
except:
    from prototype import NN


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True)


def conv(in_planes, out_planes, is_conv1x1=True, stride=1):
    if bool(is_conv1x1) is True:
        print('----conv1x1 {}_{}'.format(in_planes, out_planes))
        return conv1x1(in_planes, out_planes, stride=stride)
    else:
        print('----conv3x3 {}_{}'.format(in_planes, out_planes))
        return conv3x3(in_planes, out_planes, stride=stride)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class wide_basic(NN):

    def __init__(self, in_planes, planes, dropout_rate, is_conv1x1=(True, True), stride=1):
        super(wide_basic, self).__init__()
        print('--bottleneck start')
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv(in_planes, planes, is_conv1x1=is_conv1x1[0], stride=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv(planes, planes, is_conv1x1=is_conv1x1[1], stride=stride)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, planes, stride),
            )
        print('--bottleneck end')

    def weight_initialization(self):
        self.conv1.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv1))
        self.conv1.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv1, constant=0))
        self.conv2.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv2))
        self.conv2.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv2, constant=0))

    def forward(self, x):
        out = F.maxpool2d(self.dropout(self.conv1(F.relu(self.bn1(x)))), 3, 1, 1)
        out = F.maxpool2d(self.conv2(F.relu(self.bn2(out))), 3, 1, 1)
        out += self.shortcut(x)
        return out


class Wide_ResNet(NN):

    def __init__(self, depth, widen_factor, dropout_rate, num_classes, conv1x1=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), base=16):
        super(Wide_ResNet, self).__init__()
        self.in_planes = base

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor
        assert(int(6 * n + 1) == len(conv1x1))

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [base, base * k, base * k * 2, base * k * 4]

        counter = 0
        # first conv
        self.conv1 = conv(3, nStages[0], is_conv1x1=conv1x1[counter], stride=1)
        counter += 1
        # first bottlenecks
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1, is_conv1x1=conv1x1[counter:counter + 2 * n])
        counter += 2 * n
        # second bottlenecks
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2, is_conv1x1=conv1x1[counter:counter + 2 * n])
        counter += 2 * n
        # third bottlenecks
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2, is_conv1x1=conv1x1[counter:counter + 2 * n])
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
        self.name = 'Wide_ResNet_{}_{}_{}_{}_{}'.format(depth, widen_factor, dropout_rate, num_classes, '-'.join([str(i) for i in conv1x1]))

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, is_conv1x1):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for i in range(int(len(is_conv1x1) / 2)):
            layers.append(block(self.in_planes, planes, dropout_rate, is_conv1x1=(is_conv1x1[2 * i], is_conv1x1[2 * i + 1]), stride=strides[i]))
            self.in_planes = planes
        [layer.weight_initialization() for layer in layers]
        return nn.Sequential(*layers)

    def weight_initialization(self):
        self.conv1.weight.data = torch.FloatTensor(NN.weight_relu_initialization(self.conv1))
        self.conv1.bias.data = torch.FloatTensor(NN.bias_initialization(self.conv1, constant=0))
        self.linear.weight.data = torch.FloatTensor(np.random.uniform(-0.1, 0.1, self.linear.weight.data.shape))
        self.linear.bias.data = torch.FloatTensor(NN.bias_initialization(self.linear, constant=0.0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    net = Wide_ResNet(16, 1, 0.3, 10, conv1x1=(1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0))
    y = net(Variable(torch.randn(1, 3, 32, 32)))
    print(y.size())
