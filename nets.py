import chainer
from chainer import configuration
from chainer import functions as F
from chainer import links as L
from chainer import reporter

import variational_dropout as VD


class LeNet300100VD(VD.VariationalDropoutChain):

    def __init__(self, warm_up=0.0001):
        super(LeNet300100VD, self).__init__(warm_up=warm_up)
        self.add_link('l1', VD.VariationalDropoutLinear(784, 300))
        self.add_link('l2', VD.VariationalDropoutLinear(300, 100))
        self.add_link('l3', VD.VariationalDropoutLinear(100, 10))

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h


class LeNet5VD(VD.VariationalDropoutChain):

    def __init__(self, warm_up=0.0001):
        super(LeNet5VD, self).__init__(warm_up=warm_up)
        self.add_link('conv1', VD.VariationalDropoutConvolution2D(1, 20, 5))
        self.add_link('conv2', VD.VariationalDropoutConvolution2D(20, 50, 5))
        self.add_link('fc3', VD.VariationalDropoutLinear(800, 500))
        self.add_link('fc4', VD.VariationalDropoutLinear(500, 10))

    def __call__(self, x):
        if x.ndim == 2:
            width = int(x.shape[1] ** 0.5)
            x = x.reshape(x.shape[0], 1, width, width)
        h = F.max_pooling_2d(self.conv1(x), 2, stride=2)
        h = F.max_pooling_2d(self.conv2(h), 2, stride=2)
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        return h
