#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import division
from __future__ import print_function
import argparse

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from chainer import training
from chainer.training import extensions

import nets


class UnShift(extensions.LinearShift):
    """Trainer extension to retain an optimizer attribute within a range.
    """
    invoke_before_training = True

    def __init__(self, attr, value_range, time_range, optimizer=None):
        super(UnShift, self).__init__(attr, value_range, time_range, optimizer)

    def __call__(self, trainer):
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        t1, t2 = self._time_range
        v1, v2 = self._value_range
        if t1 <= self._t < t2:
            if self._attr < v1:
                setattr(optimizer, self._attr, v1)
            elif v2 < self._attr:
                setattr(optimizer, self._attr, v2)
        self._t += 1


# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.
class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        # Offsets maintain the position of each sequence in the mini-batch.
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a different position in the original sequence. Each item is
        # represented by a pair of two word IDs. The first word is at the
        # "current" position, while the second word at the next position.
        # At each iteration, the iteration count is incremented, which pushes
        # forward the "current" position.
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration
        cur_words = self.get_words()
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / len(self.dataset)

    def get_words(self):
        # It returns a list of current words.
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len, device,
                 loss_func=None, decay_iter=(0, 0)):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device, loss_func=loss_func)
        self.bprop_len = bprop_len
        self.decay_iter_start, self.decay_iter_span = decay_iter
        self.decay_iter_start *= bprop_len
        self.decay_iter_span *= bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()

            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = self.converter(batch, self.device)

            # Compute the loss at this time step and accumulate it
            if self.loss_func is None:
                loss += optimizer.target(chainer.Variable(x),
                                         chainer.Variable(t))
            else:
                loss += self.loss_func(chainer.Variable(x),
                                       chainer.Variable(t))

            if self.decay_iter_span != 0 and hasattr(optimizer, 'lr'):
                if train_iter.iteration >= self.decay_iter_start and \
                   train_iter.iteration % self.decay_iter_span == 0:
                    setattr(optimizer, 'lr', optimizer.lr / 2.)
                    print('lr: {} -> {}'.format(
                        optimizer.lr * 2., optimizer.lr))

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters

        reporter.report(
            {'lr': getattr(optimizer, 'lr', getattr(
                optimizer, 'alpha', None))},
            optimizer.target)


# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
def compute_perplexity(result):
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--pretrain', default=0,
                        help='Pretrain (w/o VD) or not (w/ VD).' +
                        ' default is not (0).')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    args = parser.parse_args()

    # Load the Penn Tree Bank long word sequence dataset
    train, val, test = chainer.datasets.get_ptb_words()
    n_vocab = max(train) + 1  # train is just an array of integers
    print('#vocab =', n_vocab)

    if args.test:
        train = train[:1000]
        val = val[:1000]
        test = test[:1000]

    train_iter = ParallelSequentialIterator(train, args.batchsize)
    val_iter = ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = ParallelSequentialIterator(test, 1, repeat=False)
    print('# of train:', len(train))
    n_iters = len(train) // args.batchsize // args.bproplen
    print('# of train batch/epoch:', n_iters)

    # Prepare an RNNLM model
    if args.pretrain:
        model = nets.RNNForLM(n_vocab, args.unit)

        def calc_loss(x, t):
            model.y = model(x)
            model.loss = F.softmax_cross_entropy(model.y, t)
            reporter.report({'loss': model.loss}, model)
            model.accuracy = F.accuracy(model.y, t)
            reporter.report({'accuracy': model.accuracy}, model)
            return model.loss

        model.calc_loss = calc_loss
        model.use_raw_dropout = True
    elif args.resume:
        model = nets.RNNForLMVD(n_vocab, args.unit, warm_up=1.)
        model.to_variational_dropout()
        chainer.serializers.load_npz(args.resume, model)
    else:
        model = nets.RNNForLMVD(n_vocab, args.unit, warm_up=2e-6)
        model.to_variational_dropout()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    # Set up an optimizer
    if args.pretrain:
        optimizer = chainer.optimizers.SGD(lr=1.0)
    else:
        optimizer = chainer.optimizers.Adam(alpha=1e-4)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.))

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu,
                          loss_func=model.calc_loss,
                          decay_iter=((n_iters * 6, n_iters) if args.pretrain
                                      else (0, 0)))
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Model with shared params and distinct states
    eval_model = L.Classifier(model.copy())
    eval_rnn = eval_model.predictor
    trainer.extend(extensions.Evaluator(
        val_iter, eval_model, device=args.gpu,
        # Reset the RNN state at the beginning of each evaluation
        eval_hook=lambda _: eval_rnn.reset_state()))

    interval = min(10 if args.test else 100,
                   max(n_iters, 1))
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(interval, 'iteration')))

    if args.pretrain:
        trainer.extend(extensions.PrintReport(
            ['epoch', 'iteration',
             'perplexity', 'val_perplexity',
             'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy',
             'main/lr',
             'elapsed_time']), trigger=(interval, 'iteration'))
    else:
        trainer.extend(extensions.PrintReport(
            ['epoch', 'iteration',
             'perplexity', 'val_perplexity',
             'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy',
             'main/class', 'main/kl', 'main/mean_p', 'main/sparsity',
             'main/W/Wnz', 'main/kl_coef', 'main/lr',
             'elapsed_time']), trigger=(interval, 'iteration'))

    trainer.extend(extensions.ProgressBar(
        update_interval=1 if args.test else 10))
    # trainer.extend(extensions.snapshot())
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    # Evaluate the final model
    print('test')
    eval_rnn.reset_state()
    evaluator = extensions.Evaluator(
        test_iter, eval_model, device=args.gpu)
    result = evaluator()
    print('test perplexity:', np.exp(float(result['main/loss'])))


if __name__ == '__main__':
    main()
