#!/usr/bin/env python

from __future__ import print_function

import argparse
import copy
import time

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

import nets


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--model', default='fc',
                        help='Model type from [fc, conv, lenet300100, lenet5]')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    if args.model in ['fc', 'lenet300100']:
        model = nets.LeNet300100VD(warm_up=0.001)
    elif args.model in ['conv', 'lenet5']:
        model = nets.LeNet5VD(warm_up=0.001)
    else:
        exit()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=1e-3)
    # Note:
    # Original paper sets the learning rate alpha=1e-4,
    # and linearly decays it to zero during 200 epochs.
    # And, original paper first trains a model without variational dropout,
    # and finetunes it for 10-30 epochs
    # with variational dropout and learning rate=1e-5.
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu,
                                       loss_func=model.calc_loss)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, L.Classifier(model),
                                        device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    # trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
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

    print('Measure inference speeds for 1 sample inference...')
    test_iter = chainer.iterators.SerialIterator(
        test, 1, repeat=False, shuffle=False)

    if args.gpu >= 0:
        classifier = L.Classifier(model.copy())
        start = time.time()
        accuracy = extensions.Evaluator(
            test_iter, classifier, device=args.gpu)()['main/accuracy']
        print('dense Gpu:', time.time() - start, 's/{} imgs'.format(len(test)))

    model.to_cpu()
    classifier = L.Classifier(model.copy())
    start = time.time()
    accuracy = extensions.Evaluator(
        test_iter, classifier, device=-1)()['main/accuracy']
    print('dense Cpu:', time.time() - start, 's/{} imgs'.format(len(test)))

    model.to_cpu_sparse()
    model.name = None
    classifier = L.Classifier(copy.deepcopy(model))
    start = time.time()
    accuracy = extensions.Evaluator(
        test_iter, classifier, device=-1)()['main/accuracy']
    print('sparse Cpu:', time.time() - start, 's/{} imgs'.format(len(test)))

if __name__ == '__main__':
    main()
