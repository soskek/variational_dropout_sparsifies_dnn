import numpy

import chainer
from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check
from chainer import configuration


def compositional_calculate_kl(W, log_sigma2, loga_threshold=3.,
                               eps=1e-8, thresholds=(-8., 8.)):

    def _calculate_kl(W, log_sigma2):
        log_alpha = F.clip(log_sigma2 - F.log(W ** 2 + 1e-8), -8., 8.)
        clip_mask = (log_alpha.data > loga_threshold)
        normalizer = 1. / W.size
        reg = (0.63576 * F.sigmoid(1.87320 + 1.48695 * log_alpha)) + \
              (- 0.5 * F.log1p(F.exp(- log_alpha))) - 0.63576
        xp = cuda.get_array_module(reg)
        reg = F.where(clip_mask, xp.zeros(reg.shape).astype('f'), reg)
        return - F.sum(reg) * normalizer

    return _calculate_kl(W, log_sigma2)


def _sigmoid(x):
    half = x.dtype.type(0.5)
    return numpy.tanh(x * half) * half + half


def _grad_sigmoid(x):
    return x * (1 - x)


class KL(function.Function):

    def __init__(self, clip_mask):
        self.clip_mask = clip_mask

    def check_type_forward(self, in_types):
        pass

    def forward_cpu(self, inputs):
        log_alpha = inputs[0]
        reg = (0.63576 * _sigmoid(1.87320 + 1.48695 * log_alpha)) + \
              (- 0.5 * numpy.log1p(numpy.exp(- log_alpha))) - 0.63576
        reg = reg * (1. - self.clip_mask)
        reg = - reg.sum() / log_alpha.size
        reg = utils.force_array(reg, log_alpha.dtype)
        return reg,

    def backward_cpu(self, inputs, gy):
        log_alpha = inputs[0]
        gy = gy[0]

        sig = _sigmoid(1.87320 + 1.48695 * log_alpha)
        exp_m_log_alpha = numpy.exp(- log_alpha)

        reg = (0.63576 * sig) + \
              (- 0.5 * numpy.log1p(exp_m_log_alpha)) - 0.63576
        reg = reg * (1. - self.clip_mask)

        greg = - gy / log_alpha.size * reg

        gla_from_1 = greg * 0.63576 * _grad_sigmoid(sig) * 1.48695
        gla_from_2 = greg * \
            (- 0.5) / (1. + exp_m_log_alpha) * exp_m_log_alpha

        gla = gla_from_1 + gla_from_2
        gla = utils.force_array(gla, log_alpha.dtype)
        return gla,

    def forward_gpu(self, inputs):
        log_alpha = inputs[0]
        reg = cuda.elementwise(
            'T la, T clip',
            'T reg',
            '''
            const T half = 0.5;
            const T c063576 = 0.63576;
            reg = (c063576 * 
                   (tanh(((T)1.87320 + (T)1.48695 * la) * half) * half + half) 
                   - half * log1p(exp(-la)) - c063576) * ((T)1.0 - clip);
            ''',
            'kl_fwd')(
                log_alpha, self.clip_mask)
        reg = utils.force_array(- reg.sum() / log_alpha.size, log_alpha.dtype)
        return reg,

    def backward_gpu(self, inputs, gy):
        log_alpha = inputs[0]
        gy = gy[0]

        gla = cuda.elementwise(
            'T gy, T la, T clip',
            'T gla',
            '''
            const T half = 0.5;
            const T c1 = 1.0;
            const T c063576 = 0.63576;
            const T c148695 = 1.48695;
            T sig = (tanh((1.87320 + c148695 * la) * half) * half + half);
            T exp_m_la = exp(- la);
            T reg = (c063576 * sig - 
                     half * log1p(exp_m_la) - c063576) * (c1 - clip);
            T greg = - gy * reg;
            gla = greg * (c063576 * (sig * (c1 - sig)) * c148695
                          - half / (c1 + exp_m_la) * exp_m_la)
            ''',
            'kl_bwd')(
                (gy / log_alpha.size).astype(log_alpha.dtype),
                log_alpha, self.clip_mask)
        return gla,


def calculate_kl(W=None, loga_threshold=3.,
                 log_sigma2=None, log_alpha=None,
                 eps=1e-8, thresholds=(-8., 8.)):
    if log_alpha is None:
        if log_sigma2 is None or W is None:
            AttributeError()
        log_alpha = calculate_log_alpha(
            W, log_sigma2, eps=eps, thresholds=thresholds)
    clip_mask = (log_alpha.data > loga_threshold).astype(
        log_alpha.data.dtype, copy=False)
    return KL(clip_mask)(log_alpha)


class LogAlpha(function.Function):
    """Function calculate log alpha from W and log sigma^2.

    This function is memory-efficient by recomputing in backward.
    """

    def __init__(self, eps=1e-8, lower_threshold=-8., upper_threshold=8.):
        self.eps = eps
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def check_type_forward(self, in_types):
        pass

    def forward_cpu(self, inputs):
        W, log_sigma2 = inputs
        log_alpha = log_sigma2 - numpy.log(numpy.square(W) + self.eps)
        log_alpha = utils.force_array(
            numpy.minimum(numpy.maximum(
                self.lower_threshold, log_alpha), self.upper_threshold),
            W.dtype)
        return log_alpha,

    def backward_cpu(self, inputs, gy):
        W, log_sigma2 = inputs
        gy = gy[0]
        square_W = numpy.square(W) + self.eps
        log_alpha = log_sigma2 - numpy.log(square_W)
        clip = (self.lower_threshold < log_alpha) * \
               (log_alpha < self.upper_threshold)
        clip_gy = gy * clip
        gs = clip_gy
        gW = - clip_gy / square_W * 2. * W
        gs = utils.force_array(gs, log_sigma2.dtype)
        gW = utils.force_array(gW, W.dtype)
        return gW, gs

    def forward_gpu(self, inputs):
        W, log_sigma2 = inputs
        return cuda.elementwise(
            'T W, T ls, T eps, T lo_th, T up_th',
            'T y',
            'y = min(max(ls - log(W * W + eps), lo_th), up_th)',
            'log_alpha_fwd')(
                W, log_sigma2,
                self.eps, self.lower_threshold, self.upper_threshold),

    def backward_gpu(self, inputs, gy):
        W, log_sigma2 = inputs
        gy = gy[0]
        gW, gs = cuda.elementwise(
            'T W, T ls, T gy, T eps, T lo_th, T up_th',
            'T gW, T gs',
            '''
            T square_W = W * W + eps;
            T y = ls - log(square_W);
            gs = ((y > lo_th) & (y < up_th))? gy : (T)0;
            gW = - gs / square_W * 2 * W;
            ''',
            'log_alpha_bwd')(
                W, log_sigma2, gy,
                self.eps, self.lower_threshold, self.upper_threshold)
        return gW, gs


def calculate_log_alpha(W, log_sigma2, eps=1e-8, thresholds=(-8., 8.)):
    lower_threshold, upper_threshold = thresholds
    return LogAlpha(eps, lower_threshold, upper_threshold)(W, log_sigma2)


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


class VDLinear(function.Function):
    """Linear function using variational dropout.

    This function is memory-efficient by recomputing in backward.
    """

    def __init__(self, clip_mask,
                 eps=1e-8, lower_threshold=-8., upper_threshold=8.):
        self.clip_mask = clip_mask
        self.eps = eps
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def check_type_forward(self, in_types):
        pass

    def forward_cpu(self, inputs):
        # TODO: merge calculate_log_alpha and reuse W ** 2
        x, W, log_alpha = inputs[:3]
        x = _as_mat(x)
        W = ((1. - self.clip_mask) * W).astype(W.dtype, copy=False)
        mu = x.dot(W.T)
        si = numpy.sqrt(
            (x * x).dot((numpy.exp(log_alpha) * W * W).T) + self.eps)
        self.normal_noise = numpy.random.standard_normal(mu.shape).astype(
            x.dtype, copy=False)
        y = mu + si * self.normal_noise
        if len(inputs) == 4:
            b = inputs[3]
            y += b
        return y,

    def forward_gpu(self, inputs):
        # TODO: cuda kernel
        # TODO: merge calculate_log_alpha and reuse W ** 2
        x, W, log_alpha = inputs[:3]
        x = _as_mat(x)
        W, alpha_W2 = cuda.elementwise(
            'T W, T clip_mask, T la',
            'T clip_W, T alpha_W2',
            '''
            clip_W = ((T)1.0 - clip_mask) * W;
            alpha_W2 = exp(la) * clip_W * clip_W;
            ''',
            'vdl1_fwd')(
                W, self.clip_mask, log_alpha)
        x2 = x * x
        mu = x.dot(W.T)
        si2 = x2.dot(alpha_W2.T)
        self.normal_noise = xp.random.standard_normal(mu.shape).astype(
            x.dtype, copy=False)
        y = cuda.elementwise(
            'T mu, T si2, T eps, T noise',
            'T y',
            '''
            y = mu + sqrt(si2 + eps) * noise;
            ''',
            'vdl2_fwd')(
                mu, si2, self.eps, self.normal_noise)
        if len(inputs) == 4:
            b = inputs[3]
            y += b
        return y,

    def backward_cpu(self, inputs, gy):
        # TODO: merge calculate_log_alpha and reuse W ** 2
        x, W, log_alpha = inputs[:3]
        x = _as_mat(x)
        gy = gy[0]

        clip = (1. - self.clip_mask).astype(W.dtype, copy=False)
        clip_W = clip * W
        x2 = x * x
        W2 = clip_W * clip_W
        alpha = numpy.exp(log_alpha)
        alpha_W2 = W2 * alpha
        si_before_sqrt = x2.dot(alpha_W2.T) + self.eps

        gmu = gy
        gx_from_gmu = gmu.dot(clip_W)
        gW_from_gmu = gmu.T.dot(x) * clip

        gsi = gy * self.normal_noise
        gsi_before_sqrt = gsi * (0.5 / numpy.sqrt(si_before_sqrt))
        gx2_from_gsi = gsi_before_sqrt.dot(alpha_W2)
        gx_from_gsi = gx2_from_gsi * (2. * x)
        galpha_W2_from_gsi = gsi_before_sqrt.T.dot(x2)
        gW2_from_gsi = galpha_W2_from_gsi * alpha
        gW_from_gsi = gW2_from_gsi * (2. * clip_W)
        galpha_from_gsi = galpha_W2_from_gsi * W2
        glog_alpha = galpha_from_gsi * numpy.exp(log_alpha)

        gx = (gx_from_gmu + gx_from_gsi).astype(x.dtype, copy=False).reshape(
            inputs[0].shape)
        gW = (gW_from_gmu + gW_from_gsi).astype(W.dtype, copy=False)
        if len(inputs) == 4:
            gb = gy.sum(0)
            return gx, gW, glog_alpha, gb
        else:
            return gx, gW, glog_alpha

    def backward_gpu(self, inputs, gy):
        # TODO: merge calculate_log_alpha and reuse W ** 2
        x, W, log_alpha = inputs[:3]
        x = _as_mat(x)
        xp = cuda.get_array_module(x)
        gy = gy[0]

        clip, clip_W, W2, alpha, alpha_W2 = cuda.elementwise(
            'T W, T clip_mask, T la',
            'T clip, T clip_W, T W2, T alpha, T alpha_W2',
            '''
            clip = ((T)1.0 - clip_mask);
            clip_W = clip * W;
            W2 = clip_W * clip_W;
            alpha = exp(la);
            alpha_W2 = W2 * alpha;
            ''',
            'vdl1_bwd')(
                W, self.clip_mask, log_alpha)

        x2 = x * x
        si_before_sqrt = x2.dot(alpha_W2.T)
        gx_from_gmu = gy.dot(clip_W)
        gW_from_gmu = gy.T.dot(x) * clip

        gsi_before_sqrt = cuda.elementwise(
            'T gy, T noise, T si_bf_sqrt, T eps',
            'T gsi_bf_sqrt',
            '''
            gsi_bf_sqrt = gy * noise * ((T)0.5 / sqrt(si_bf_sqrt + eps));
            ''',
            'gsi_bwd')(
                gy, self.normal_noise, si_before_sqrt, self.eps)

        galpha_W2_from_gsi = gsi_before_sqrt.T.dot(x2)
        gW_from_gsi = galpha_W2_from_gsi * alpha * (2. * clip_W)
        glog_alpha = galpha_W2_from_gsi * W2 * alpha

        gW, glog_alpha = cuda.elementwise(
            'T galpha_W2_from_gsi, T alpha, T clip_W, T W2, T gW_from_gmu',
            'T gW, T glog_alpha',
            '''
            gW = galpha_W2_from_gsi * alpha * (T)2.0 * clip_W + gW_from_gmu;
            glog_alpha = galpha_W2_from_gsi * W2 * alpha;
            ''',
            'gW_glog_bwd')(
                galpha_W2_from_gsi, alpha, clip_W, W2, gW_from_gmu)

        gx_from_gsi = gsi_before_sqrt.dot(alpha_W2) * 2. * x
        gx = (gx_from_gmu + gx_from_gsi).astype(x.dtype, copy=False).reshape(
            inputs[0].shape)
        if len(inputs) == 4:
            gb = gy.sum(0)
            return gx, gW, glog_alpha, gb
        else:
            return gx, gW, glog_alpha


def vd_linear(x, W, b, loga_threshold=3., log_sigma2=None,
              log_alpha=None, eps=1e-8, thresholds=(-8., 8.)):
    if log_alpha is None:
        if log_sigma2 is None:
            AttributeError()
        log_alpha = calculate_log_alpha(
            W, log_sigma2, eps=eps, thresholds=thresholds)
    clip_mask = (log_alpha.data > loga_threshold).astype(
        log_alpha.data.dtype, copy=False)

    train = configuration.config.train
    if train:
        if b is None:
            return VDLinear(clip_mask, eps)(
                x, W, log_alpha)
        else:
            return VDLinear(clip_mask, eps)(
                x, W, log_alpha, b)
    else:
        return F.linear(x, (1. - clip_mask) * W, b)

if __name__ == '__main__':
    F = chainer.functions
    import time
    import cupy
    from chainer import testing

    #xp = numpy
    xp = cupy

    batchsize = 128
    m = 256
    n = 512

    check_log_alpha = True
    if check_log_alpha:
        print('### LOG ALPHA ###')
        xp.random.seed(777)
        W = chainer.Variable(
            xp.random.rand(n, m).astype('f') * 20)
        log_sigma2 = chainer.Variable(
            xp.random.rand(n, m).astype('f') * 20)

        times = []
        for i in range(10):
            start = time.time()
            y = calculate_log_alpha(W, log_sigma2)
            F.sum(y).backward()
            vs1 = [y.data,
                   W.grad,
                   W.data,
                   log_sigma2.grad,
                   log_sigma2.data]
            times.append(time.time() - start)
        print('direct cuda', numpy.mean(times[5:]))

        W.cleargrad()
        log_sigma2.cleargrad()

        times = []
        for i in range(10):
            start = time.time()
            y = F.clip(log_sigma2 - F.log(W * W + 1e-8), -8., 8.)
            F.sum(y).backward()
            vs2 = [y.data,
                   W.grad,
                   W.data,
                   log_sigma2.grad,
                   log_sigma2.data]
            times.append(time.time() - start)
        print('composition', numpy.mean(times[5:]))

        for v1, v2 in zip(vs1, vs2):
            testing.assert_allclose(v1, v2)

    print('### VD LINEAR ###')
    xp.random.seed(777)
    W = chainer.Variable(
        xp.random.rand(n, m).astype('f') * 10)
    b = chainer.Variable(
        xp.random.rand(n).astype('f'))
    x = chainer.Variable(
        xp.random.rand(batchsize, m).astype('f'))
    log_sigma2 = chainer.Variable(
        xp.random.rand(n, m).astype('f') * 20)
    loga_threshold = 3.

    times = []
    for i in range(10):
        W.cleargrad()
        b.cleargrad()
        x.cleargrad()
        log_sigma2.cleargrad()
        xp.random.seed(777)
        start = time.time()
        y = vd_linear(x, W, b, loga_threshold, log_sigma2=log_sigma2,
                      log_alpha=None, eps=1e-8, thresholds=(-8., 8.))
        F.sum(y).backward()
        vs1 = [y.data,
               W.grad,
               b.grad,
               x.grad,
               log_sigma2.grad]
        times.append(time.time() - start)
    print('direct cuda', numpy.mean(times[5:]))

    times = []
    for i in range(10):
        W.cleargrad()
        b.cleargrad()
        x.cleargrad()
        log_sigma2.cleargrad()
        xp.random.seed(777)

        start = time.time()

        log_alpha = F.clip(log_sigma2 - F.log(W * W + 1e-8), -8., 8.)
        clip_mask = (log_alpha.data > loga_threshold)
        _W = (1. - clip_mask) * W
        mu = F.linear(x, _W)
        si = F.sqrt(F.linear(x * x, F.exp(log_alpha) * _W * _W) + 1e-8)
        normal_noise = xp.random.standard_normal(mu.shape).astype('f')
        y = mu + si * normal_noise
        if b is not None:
            y = F.bias(y, b)

        F.sum(y).backward()
        vs2 = [y.data,
               W.grad,
               b.grad,
               x.grad,
               log_sigma2.grad, ]
        times.append(time.time() - start)

    print('composition', numpy.mean(times[5:]))
    for v1, v2 in zip(vs1, vs2):
        testing.assert_allclose(v1, v2, rtol=0.001)

    print('### KL ###')
    times = []
    for i in range(10):
        W.cleargrad()
        log_sigma2.cleargrad()
        xp.random.seed(777)
        start = time.time()
        y = calculate_kl(W, loga_threshold=3.,
                         log_sigma2=log_sigma2, log_alpha=None,
                         eps=1e-8, thresholds=(-8., 8.))
        F.sum(y).backward()
        vs1 = [y.data,
               W.grad,
               log_sigma2.grad]
        times.append(time.time() - start)
    print('direct cuda', numpy.mean(times[5:]))

    times = []
    for i in range(10):
        W.cleargrad()
        log_sigma2.cleargrad()
        xp.random.seed(777)

        start = time.time()
        y = compositional_calculate_kl(W, log_sigma2, loga_threshold=3.,
                                       eps=1e-8, thresholds=(-8., 8.))
        F.sum(y).backward()
        vs2 = [y.data,
               W.grad,
               log_sigma2.grad, ]
        times.append(time.time() - start)

    print('composition', numpy.mean(times[5:]))
    for v1, v2 in zip(vs1, vs2):
        testing.assert_allclose(v1, v2, rtol=0.001)
