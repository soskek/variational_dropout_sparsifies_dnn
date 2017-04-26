# Variational Dropout Sparsifies Deep Neural Networks
The code includes variational dropout for linear and convolutional layers to sparsify deep neural networks.
It will replicate experiments in the paper below  
```
Variational Dropout Sparsifies Deep Neural Networks.  
Dmitry Molchanov, Arsenii Ashukha, Dmitry Vetrov.  
Under review by ICML 2017.
```

See https://arxiv.org/pdf/1701.05369.pdf.

This implementation uses new Chainer version 2 (see https://github.com/pfnet/chainer/tree/_v2 or https://github.com/pfnet/chainer/releases/tag/v2.0.0b1).
This does not work on the old version of Chainer (in its master branch).

This repository contains  
- MNIST example using variational dropout
    - LeNet-300-100 and LeNet5
- General Chain for models using variational dropout
- Linear link using variational dropout
- Convolution2D link using variational dropout
- Sparse forward computation of Linear link

The code of variational dropout is partly based on the paper and the authors' [repository](https://github.com/ars-ashuha/variational-dropout-sparsifies-dnn), which uses theano instead of Chainer.

# Requirements

Requirements of this code obey those of Chainer v2. Additionally, this requires [scipy](https://www.scipy.org/) for sparse matrix computation on CPU.

Note Chainer are planning the v2.0.0 release on May 16.
Until the day, you can install Chainer v2 beta by following  
```
pip install chainer --pre
```
If you use GPU (CUDA/cuDNN), also  
```
pip install cupy
```

This reposity itself does not need any setup.

# Examples

- MNIST: Convolutional network (LeNet5) or fully-connected feedforward network (LeNet-300-100) for MNIST. The example is derived from the official MNIST example of Chainer v2.  
  ```
  python -u train_mnist.py --gpu=0
  ```
  Some settings are different from those of experiments in the paper;
  this learning rate is higher and not decayed and this uses warmup (annealing) training rather than
  two seperate stages of pretraining (w/o VD) and finetuning (w/ VD).

# How to use variational dropout (VD) in Chainer

## VariationalDropoutChain
This implements a general model class `VariationalDropoutChain`, which inherits `chainer.Chain`.
The class has a function to calculate joint objective about loss (sum of cross entroy and KL divergence).
So, if you use Chainer's official Updater in your code, you can use VD training by writing as follows
```
updater = training.StandardUpdater(
    train_iter, optimizer, device=args.gpu,
    loss_func=model.calc_loss)
```
You can also observe some statistics about VD (e.g., sparsity) in the model
during training using `chainer.extensions.PrintReport` (see the MNIST example).

## VariationalDropoutLinear, VariationalDropoutConvolution2D
A model based on `VariationalDropoutChain` can use special layers (Chainer's `link`) in its structure.
This repository provides both
- `VariationalDropoutLinear`, which inherits `chainer.links.Linear`
- `VariationalDropoutConvolution2D`, which inherits `chainer.links.Convolution2D`

You can use them just by replacing exsisting `chainer.links.Linear` or `chainer.links.Convolution2D` respectively.
All available arguments of the old variants are supported.
And, additional arguments for hyperparameters
(`p_threshold`, `loga_threshold` and `initial_log_sigma2`) are also available.
They are already set good parameters shown in the paper by default.

## Forward Propagation using Sparse Computation of scipy.sparse
A model based on `VariationalDropoutChain` can use the method `model.to_cpu_sparse()`.
The method transforms all linear layers in the model into new layers with pruned weights
using sparse matrix on scipy.sparse.
It accelerates the forward propagation and reduces memory after VD training.
Please see this usage in MNIST example.

Note: The transformed model works only on CPUs, for the forward propagation, and in inference.
