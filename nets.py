import chainer
from chainer import configuration
from chainer import functions as F
from chainer import links as L

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


class Block(chainer.Chain):

    """A convolution, batch norm, ReLU block.

    A block in a feedforward network that performs a
    convolution followed by batch normalization followed
    by a ReLU activation.

    For the convolution operation, a square filter size is used.

    Args:
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.

    """

    def __init__(self, out_channels, ksize, pad=1):
        initializer = chainer.initializers.HeNormal()
        super(Block, self).__init__(
            conv=L.Convolution2D(None, out_channels, ksize, pad=pad,
                                 nobias=True, initialW=initializer),
            bn=L.BatchNormalization(out_channels)
        )

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)


class VGG16(chainer.Chain):

    """A VGG-style network for very small images.

    This model is based on the VGG-style model from
    http://torch.ch/blog/2015/07/30/cifar.html
    which is based on the network architecture from the paper:
    https://arxiv.org/pdf/1409.1556v6.pdf

    This model is intended to be used with either RGB or greyscale input
    images that are of size 32x32 pixels, such as those in the CIFAR10
    and CIFAR100 datasets.

    On CIFAR10, it achieves approximately 89% accuracy on the test set with
    no data augmentation.

    On CIFAR100, it achieves approximately 63% accuracy on the test set with
    no data augmentation.

    Args:
        class_labels (int): The number of class labels.

    """

    def __init__(self, class_labels=10):
        initializer = chainer.initializers.HeNormal()
        super(VGG16, self).__init__(
            block1_1=Block(64, 3),
            block1_2=Block(64, 3),
            block2_1=Block(128, 3),
            block2_2=Block(128, 3),
            block3_1=Block(256, 3),
            block3_2=Block(256, 3),
            block3_3=Block(256, 3),
            block4_1=Block(512, 3),
            block4_2=Block(512, 3),
            block4_3=Block(512, 3),
            block5_1=Block(512, 3),
            block5_2=Block(512, 3),
            block5_3=Block(512, 3),
            fc1=L.Linear(None, 512, nobias=True, initialW=initializer),
            bn_fc1=L.BatchNormalization(512),
            fc2=L.Linear(None, class_labels, nobias=True,
                         initialW=initializer),
        )
        self.use_raw_dropout = False

    def __call__(self, x):
        train = configuration.config.train
        if train:
            # horizontal flips
            flipped = x[:x.shape[0] // 2, :, :, ::-1]
            x = self.xp.concatenate([flipped, x[x.shape[0] // 2:]], axis=0)

        # 64 channel blocks:
        h = self.block1_1(x)
        if self.use_raw_dropout:
            h = F.dropout(h, ratio=0.3)
        h = self.block1_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 128 channel blocks:
        h = self.block2_1(h)
        if self.use_raw_dropout:
            h = F.dropout(h, ratio=0.4)
        h = self.block2_2(h)
        h = F.max_pooling_2d(
            h, ksize=2, stride=2)
        # 256 channel blocks:
        h = self.block3_1(h)
        if self.use_raw_dropout:
            h = F.dropout(h, ratio=0.4)
        h = self.block3_2(h)
        if self.use_raw_dropout:
            h = F.dropout(h, ratio=0.4)
        h = self.block3_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512
        # channel
        # blocks:
        h = self.block4_1(h)
        if self.use_raw_dropout:
            h = F.dropout(h, ratio=0.4)
        h = self.block4_2(h)
        if self.use_raw_dropout:
            h = F.dropout(h, ratio=0.4)
        h = self.block4_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block5_1(h)
        if self.use_raw_dropout:
            h = F.dropout(h, ratio=0.4)
        h = self.block5_2(h)
        if self.use_raw_dropout:
            h = F.dropout(h, ratio=0.4)
        h = self.block5_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        if self.use_raw_dropout:
            h = F.dropout(h, ratio=0.5)
        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = F.relu(h)
        if self.use_raw_dropout:
            h = F.dropout(h, ratio=0.5)
        return self.fc2(h)


class VGG16VD(VD.VariationalDropoutChain, VGG16):

    def __init__(self, class_labels=10, warm_up=0.0001):
        super(VGG16VD, self).__init__(
            warm_up=warm_up, class_labels=class_labels)
