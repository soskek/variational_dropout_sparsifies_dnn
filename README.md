# Variational Dropout Sparsifies Deep Neural Networks
The code includes variational dropout for linear and convolutional layers to sparsify deep neural networks by Chainer.
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
- CIFAR-10 or -100 example using variational dropout
    - VGGNet16
- PennTreeBank RNN language model example using variational dropout
    - 2-layer LSTM LM
    - This experiment is original and does not exist in the paper
- General Chain for models using variational dropout
- Linear link using variational dropout
- Convolution2D link using variational dropout
- Sparse forward computation of Linear link

The code of variational dropout is partly based on the paper and the authors' [repository](https://github.com/ars-ashuha/variational-dropout-sparsifies-dnn), which uses theano instead of Chainer.
Example scripts are derived from official examples of Chainer.

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
  
- CIFAR-10 or 100: Convolutional network (VGGNet) for CIFAR. The example is derived from the official CIFAR example of Chainer v2.  
  ```
  python -u train_cifar.py --gpu=0
  ```
  This currently fails to completely reproduce results shown in the paper w.r.t both improving sparsity and retaining accuracy.  
  Additional arguments for running are as follows  
  - `--resume FILE`: Load a pretrain model (if needed). Default is none, and start training from random initialization.
  - `--pretrain 1/0`: 1 -> Pretrain w/o VD. 0 -> finetune (from `resume`) or warmup training w/ VD. Default is 0.
  - `--dataset cifar10/cifar100`: Target dataset. Default is cifar10.

- PTB: RNNLM using recurrent network for language modeling on PennTreeBank. This experiment is original from this repository rather than from the paper. The example is derived from the official PTB example of Chainer v2.  
  ```
  python -u train_ptb.py --gpu=0
  ```
  VD makes RNN require large memory and much time.

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
during training using `chainer.extensions.PrintReport` (see the MNIST or CIFAR example).

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

These links are used as a primitive part of more complex neural networks.
For example,
tanh RNN (i.e., vanilla RNN) can be written with `chainer.links.Linear` layer.
Thus, the VD variant of tanh RNN `VariationalDropoutTanhRNN` can also be written with `VariationalDropoutLinear`.
This is used in PTB example, and see `VariationalDropoutTanhRNN` in `net.py` for detailed structure.


## Convert common Chain to new Chain using VD
You can also use variational dropout on an existing `chainer.Chain` model class
by wrapping the target class with `VariationalDropoutChain` and
calling `.to_variational_dropout()` as follows
```
class VGG16VD(VD.VariationalDropoutChain, VGG16):
    def __init__(self, class_labels=10, warm_up=0.0001):
        super(VGG16VD, self).__init__(warm_up=warm_up, class_labels=class_labels)

model = VGG16VD()
model.to_variational_dropout()
```

You can see this usage in CIFAR example.

## Forward Propagation using Sparse Computation of scipy.sparse
After training, especially VD training,
it is desirable to use a model for inference lightly on CPU.
A model based on `VariationalDropoutChain` can use the method `.to_cpu_sparse()`.
The method transforms all linear layers in the model into new layers with pruned weights
using sparse matrix on `scipy.sparse`.
This accelerates the forward propagation and reduces memory after VD training.
However, the current implementation does not accelerates convolutional layers
due to a lack of good methods of convoluions with sparse filters.
Thus, a model almost consisting of convolutional layers (e.g. VGGNet) can not be accelerated.
Please see this usage in MNIST example.

Note: The transformed model works only on CPUs, for the forward propagation, and in inference.
