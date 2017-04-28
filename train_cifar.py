from __future__ import print_function
import argparse
import copy
import time

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

import nets
# VGG16VD


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    if args.dataset == 'cifar10':
        print('Using CIFAR10 dataset.')
        class_labels = 10
        train, test = get_cifar10()
    elif args.dataset == 'cifar100':
        print('Using CIFAR100 dataset.')
        class_labels = 100
        train, test = get_cifar100()
    else:
        raise RuntimeError('Invalid dataset choice.')
    #model = L.Classifier(nets.VGG16(class_labels))
    #model.calc_loss = model.__call__
    model = nets.VGG16VD(class_labels, warm_up=0.0001)
    model(train[0][0][None, ])  # for setting in_channels automatically
    model.to_variational_dropout()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    #optimizer = chainer.optimizers.MomentumSGD(0.1)
    optimizer = chainer.optimizers.Adam(1e-4)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(10.))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

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

    # Reduce the learning rate by half every 25 epochs.
    # trainer.extend(extensions.ExponentialShift('lr', 0.5),
    #               trigger=(25, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    # trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    # trainer.extend(extensions.LogReport())
    per = min(len(train) // args.batchsize // 2, 1000)
    trainer.extend(extensions.LogReport(trigger=(per, 'iteration')))

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
