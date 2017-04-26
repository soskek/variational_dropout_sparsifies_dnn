import numpy

import chainer
from chainer import reporter
from chainer import configuration
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import warnings


def calculate_stats(chain, threshold=0.95):
    xp = chain.xp
    stats = {}
    all_p = xp.concatenate(
        [link.calculate_p().data.flatten()
         for link in chain.links()
         if getattr(link, 'is_variational_dropout', False)],
        axis=0)
    stats['mean_p'] = xp.mean(all_p)

    all_threshold = [link.p_threshold
                     for link in chain.links()
                     if getattr(link, 'is_variational_dropout', False)]
    if any(th != threshold for th in all_threshold):
        warnings.warn('The threshold for sparsity calculation is different from'
                      ' thresholds used for prediction with'
                      ' threshold-based pruning.')

    is_zero = (all_p > threshold)
    stats['sparsity'] = xp.mean(is_zero)

    n_non_zero = (1 - is_zero).sum()
    if n_non_zero == 0:
        stats['W/Wnz'] = float('inf')
    else:
        stats['W/Wnz'] = all_p.size * 1. / n_non_zero
    return stats


class VariationalDropoutLinear(chainer.links.Linear):

    def __init__(self, in_size, out_size, nobias=False,
                 initialW=None, initial_bias=None,
                 p_threshold=0.95, loga_threshold=3.,
                 initial_log_sigma2=chainer.initializers.Constant(-8.)):
        super(VariationalDropoutLinear, self).__init__(
            in_size, out_size, nobias,
            initialW, initial_bias)
        self.add_param('log_sigma2', initializer=initial_log_sigma2)
        if in_size is not None:
            self._initialize_params(in_size, log_sigma2=True)
        self.p_threshold = p_threshold
        self.loga_threshold = loga_threshold
        self.is_variational_dropout = True

    def _initialize_params(self, in_size, log_sigma2=False):
        if not log_sigma2:
            self.W.initialize((self.out_size, in_size))
        else:
            self.log_sigma2.initialize((self.out_size, in_size))

    def dropout_linear(self, x):
        train = configuration.config.train
        W, b = self.W, self.b
        log_alpha = F.clip(self.log_sigma2 - F.log(W ** 2), -8., 8.)
        clip_mask = (log_alpha.data > self.loga_threshold)
        if train:
            W = (1. - clip_mask) * W
            mu = F.linear(x, W)
            si = F.sqrt(F.linear(x * x, F.exp(log_alpha) * W * W) + 1e-8)
            normal_noise = self.xp.random.normal(
                0., 1., mu.shape).astype('f')
            activation = mu + si * normal_noise
            return F.bias(activation, b)
        else:
            return F.linear(x, (1. - clip_mask) * W, b)

    def calculate_kl(self):
        W = self.W
        log_alpha = F.clip(self.log_sigma2 - F.log(W ** 2), -8., 8.)
        clip_mask = (log_alpha.data > self.loga_threshold)
        normalizer = 1. / W.size
        reg = (0.5 * F.log1p(F.exp(- log_alpha)) -
               (0.03 + 1.0 / (1.0 + F.exp(- (1.5 * (log_alpha + 1.3)))) * 0.64))
        min_val = self.xp.full(reg.shape, -0.67).astype('f')
        reg = F.where(clip_mask, min_val, reg)
        return F.sum(reg) * normalizer
        # def regf(self, a):
        # return F.sum(0.5 * F.log(a) + 1.16145124 * a - 1.50204118 * a * a +
        # 0.58629921 * a * a * a)

    def calculate_p(self):
        W = self.W
        alpha = F.exp(F.clip(self.log_sigma2 - F.log(W ** 2), -8., 8.))
        p = alpha / (1 + alpha)
        return p

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.size // x.shape[0])

        return self.dropout_linear(x)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class VariationalDropoutConvolution2D(chainer.links.Convolution2D):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 deterministic=False,
                 p_threshold=0.95, loga_threshold=3.,
                 initial_log_sigma2=chainer.initializers.Constant(-8.)):
        super(VariationalDropoutConvolution2D, self).__init__(
            in_channels, out_channels, ksize, stride, pad,
            nobias, initialW, initial_bias, deterministic)

        self.add_param('log_sigma2', initializer=initial_log_sigma2)
        if in_channels is not None:
            self._initialize_params(in_channels, log_sigma2=True)
        self.p_threshold = p_threshold
        self.loga_threshold = loga_threshold
        self.is_variational_dropout = True

    def _initialize_params(self, in_channels, log_sigma2=False):
        kh, kw = _pair(self.ksize)
        W_shape = (self.out_channels, in_channels, kh, kw)
        if not log_sigma2:
            self.W.initialize(W_shape)
        else:
            self.log_sigma2.initialize(W_shape)

    def dropout_convolution_2d(self, x):
        train = configuration.config.train
        W, b = self.W, self.b
        log_alpha = F.clip(self.log_sigma2 - F.log(W ** 2), -8., 8.)
        clip_mask = (log_alpha.data > self.loga_threshold)
        if train:
            W = (1. - clip_mask) * W
            mu = F.convolution_2d(x, (1. - clip_mask) * W, b=None,
                                  stride=self.stride, pad=self.pad,
                                  deterministic=self.deterministic)
            si = F.sqrt(
                F.convolution_2d(x * x, F.exp(log_alpha) * W * W, b=None,
                                 stride=self.stride, pad=self.pad,
                                 deterministic=self.deterministic) + 1e-8)
            normal_noise = self.xp.random.normal(
                0., 1., mu.shape).astype('f')
            activation = mu + si * normal_noise
            return F.bias(activation, b)
        else:
            return F.convolution_2d(x, (1. - clip_mask) * W, b,
                                    stride=self.stride, pad=self.pad,
                                    deterministic=self.deterministic)

    def calculate_kl(self):
        W = self.W
        log_alpha = F.clip(self.log_sigma2 - F.log(W ** 2), -8., 8.)
        clip_mask = (log_alpha.data > self.loga_threshold)
        normalizer = 1. / W.size
        reg = (0.5 * F.log1p(F.exp(- log_alpha)) -
               (0.03 + 1.0 / (1.0 + F.exp(- (1.5 * (log_alpha + 1.3)))) * 0.64))
        min_val = self.xp.full(reg.shape, -0.67).astype('f')
        reg = F.where(clip_mask, min_val, reg)
        return F.sum(reg) * normalizer

    def calculate_p(self):
        W = self.W
        alpha = F.exp(F.clip(self.log_sigma2 - F.log(W ** 2), -8., 8.))
        p = alpha / (1 + alpha)
        return p

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return self.dropout_convolution_2d(x)


class VariationalDropoutChain(chainer.link.Chain):

    def __init__(self, warm_up=0.0001):
        super(VariationalDropoutChain, self).__init__()
        self.warm_up = warm_up
        if self.warm_up:
            self.kl_coef = 0.
        else:
            self.kl_coef = 1.

    def __call__(self, x):
        NotImplementedError

    def calc_loss(self, x, t):
        self.y = self(x)
        self.class_loss = F.softmax_cross_entropy(self.y, t)
        a_regf = sum(
            link.calculate_kl()
            for link in self.links()
            if getattr(link, 'is_variational_dropout', False))
        self.kl_loss = a_regf * self.kl_coef

        train = configuration.config.train
        if train:
            reporter.report({'kl_coef': self.kl_coef}, self)
            self.kl_coef = min(self.kl_coef + self.warm_up, 1.)

        self.loss = self.class_loss + self.kl_loss
        reporter.report({'loss': self.loss}, self)
        reporter.report({'class': self.class_loss}, self)
        reporter.report({'kl': self.kl_loss}, self)

        self.accuracy = F.accuracy(self.y, t)
        reporter.report({'accuracy': self.accuracy}, self)

        stats = calculate_stats(self)
        reporter.report({'mean_p': stats['mean_p']}, self)
        reporter.report({'sparsity': stats['sparsity']}, self)
        reporter.report({'W/Wnz': stats['W/Wnz']}, self)

        return self.loss
