import numpy
import scipy
from scipy import sparse

import chainer
from chainer import cuda
from chainer import reporter
from chainer import configuration
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import warnings


class SparseLinearForwardCPU(chainer.links.Linear):

    def __init__(self, old_linear, W_mask=None, with_dense=False):
        W, b = old_linear.W.data, old_linear.b.data
        super(SparseLinearForwardCPU, self).__init__(
            W.shape[1], W.shape[0],
            initialW=W, initial_bias=b)
        if not with_dense:
            delattr(self, 'W')
            delattr(self, 'b')

        xp = cuda.get_array_module(W)
        if W_mask is None:
            W_mask = xp.ones(W.shape).astype('f')

        if xp is numpy:
            self.sparse_W = sparse.csc_matrix(W * W_mask)
            self.sparse_b = b
        else:
            self.sparse_W = sparse.csr_matrix(
                xp.asnumpy(W) * xp.asnumpy(W_mask))
            self.sparse_b = xp.asnumpy(b)[None, ]

    def __call__(self, x):
        train = configuration.config.train
        if self.xp is numpy and not train:
            if isinstance(x, chainer.Variable):
                x = x.data
            return self.sparse_W.dot(x.T).T + self.sparse_b
        else:
            warnings.warn('SparseLinearForwardCPU link is made for'
                          ' inference usage. Sparse computation'
                          ' (scipy.sparse) computation is used'
                          ' only in inference mode'
                          ' rather than training mode.')
            if hasattr(self, 'W'):
                return super(SparseLinearForwardCPU, self).__call__(x)
            else:
                NotImplementedError
