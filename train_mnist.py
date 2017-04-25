
#!/usr/bin/env python

from __future__ import print_function

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse

import numpy

import chainer
from chainer import reporter
from chainer import configuration
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


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
            normal_noise = self.xp.random.normal(0., 1., mu.shape).astype('f')
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

    def calculate_p(self):
        W = self.W
        alpha = F.exp(F.clip(self.log_sigma2 - F.log(W ** 2), -8., 8.))
        p = alpha / (1 + alpha)
        return p

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.size // x.shape[0])

        return self.dropout_linear(x)

# Network definition


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out, n_layers=3):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=chainer.ChainList(*[
                VariationalDropoutLinear(n_units, n_units)
                for i in range(n_layers)]),
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )
        self.units = n_units
        self.layers = n_layers

        self.coef = 0.

    def __call__(self, x):
        train = configuration.config.train
        h = F.relu(self.l1(x))
        for layer in self.l2:
            h = F.relu(layer(h))
        return self.l3(h)

    # def regf(self, a):
    # return F.sum(0.5 * F.log(a) + 1.16145124 * a - 1.50204118 * a * a +
    # 0.58629921 * a * a * a)

    def calc_loss(self, x, t):
        self.y = self(x)
        self.class_loss = F.softmax_cross_entropy(self.y, t)
        a_regf = sum(
            link.calculate_kl()
            for link in self.links()
            if getattr(link, 'is_variational_dropout', False))
        self.kl_loss = a_regf
        self.kl_loss *= self.coef

        self.coef += 0.001
        if self.coef >= 1.:
            self.coef = 1.

        self.loss = self.class_loss + self.kl_loss
        reporter.report({'loss': self.loss}, self)
        reporter.report({'class': self.class_loss}, self)
        reporter.report({'kl': self.kl_loss}, self)
        self.accuracy = F.accuracy(self.y, t)
        reporter.report({'accuracy': self.accuracy}, self)

        p = F.stack(
            [link.calculate_p()
             for link in self.links()
             if getattr(link, 'is_variational_dropout', False)]).data
        self.m = self.xp.mean(p)
        reporter.report({'mean_p': self.m}, self)
        self.r = self.xp.mean(p < 0.95)
        reporter.report({'rate_p<95': self.r}, self)

        return self.loss


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=500,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    # model = L.Classifier(MLP(args.unit, 10))
    model = MLP(args.unit, 10)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    #updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                                       loss_func=model.calc_loss)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    #trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.Evaluator(test_iter, L.Classifier(model),
                                        device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    # trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    # trainer.extend(extensions.LogReport())
    per = min(len(train) // args.batchsize, 1000)
    trainer.extend(extensions.LogReport(trigger=(per, 'iteration')))

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy',
         'main/class', 'main/kl', 'main/mean_p', 'main/rate_p<95',
         'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
