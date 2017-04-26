# Variational Dropout Sparsifies Deep Neural Networks
The code includes variational dropout linear layer to sparsify deep neural networks.
It will replicate experiments in the paper below  
```
Variational Dropout Sparsifies Deep Neural Networks.  
Dmitry Molchanov, Arsenii Ashukha, Dmitry Vetrov.  
Under review by ICML 2017.
```

See https://arxiv.org/pdf/1701.05369.pdf.

This implementation uses new Chainer version 2 (see https://github.com/pfnet/chainer/tree/_v2 or https://github.com/pfnet/chainer/releases/tag/v2.0.0b1).
This does not work on the old version of Chainer (in its master branch).

This is based on the paper and the authors' [repository](https://github.com/ars-ashuha/variational-dropout-sparsifies-dnn), which uses theano.

# Requirements

A few requirements obey those of Chainer v2.

Note Chainer are planning the v2.0.0 release on May 16.
Until the day, you can install Chainer v2 by following  
```
pip install chainer --pre
```
If you use GPU (CUDA/cuDNN), also  
```
pip install cupy
```

This reposity itself does not need any setup.

# Examples

- MNIST: Fully-connected DNN for MNIST. The example is derived from the official MNIST example of Chainer v2.  
  ```
  python -u train_mnist.py --gpu=0
  ```
  Some settings are different from those of experiments in the paper;
  this learning rate is higher and not decayed and this uses warmup (annealing) training rather than
  two seperate stages of pretraining (w/o VD) and finetuning (w/ VD).

# How to use variational dropout (VD) in Chainer

This implements a general model class `VariationalDropoutChain`, which inherits `chainer.Chain`.
The class has a function to calculate joint objective about loss (sum of cross entroy and KL divergence).
So, if you use Chainer's official Updater in your code, you can use VD training by writing as follows
```
updater = training.StandardUpdater(
    train_iter, optimizer, device=args.gpu,
    loss_func=model.calc_loss)
```
You can also observe some statistics about variational dropout (e.g., sparsity) in the model
during training using `chainer.extensions.PrintReport` (see the MNIST example).

A model based on `VariationalDropoutChain` can use special layers (Chainer's `link`) in its structure.
This repository provides both
- `VariationalDropoutLinear`, which inherits inherits `chainer.links.Linear`
- `VariationalDropoutConvolution2D`, which inherits `chainer.links.Convolution2D`

You can use them just by replacing usual `chainer.links.Linear` or `chainer.links.Convolution2D` respectively.
All of the available arguments of usual variants are supported.
And, additional arguments for hyperparameters
(`p_threshold`, `loga_threshold` and `initial_log_sigma2`) are also available.
They are already set good parameters shown in the paper by default.
