
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

import variational_dropout


# Network definition

class MLP(chainer.Chain):

    def __init__(self, n_units, n_out, n_layers=1, warm_up=0.001):
        super(MLP, self).__init__(
            l1=variational_dropout.VariationalDropoutLinear(784, n_units),
            l2=chainer.ChainList(*[
                variational_dropout.VariationalDropoutLinear(n_units, n_units)
                for i in range(n_layers)]),
            l3=variational_dropout.VariationalDropoutLinear(n_units, n_out),
        )
        self.units = n_units
        self.layers = n_layers

        self.warm_up = warm_up
        if self.warm_up:
            self.kl_coef = 0.
        else:
            self.kl_coef = 1.

    def __call__(self, x):
        train = configuration.config.train
        h = F.relu(self.l1(x))
        for layer in self.l2:
            h = F.relu(layer(h))
        return self.l3(h)

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

        stats = variational_dropout.calculate_stats(self)
        reporter.report({'mean_p': stats['mean_p']}, self)
        reporter.report({'sparsity': stats['sparsity']}, self)
        reporter.report({'W/Wnz': stats['W/Wnz']}, self)

        return self.loss


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
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
    per = min(len(train) // args.batchsize // 2, 1000)
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
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy',
         'main/class', 'main/kl', 'main/mean_p', 'main/sparsity',
         'main/W/Wnz', 'main/kl_coef',
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
