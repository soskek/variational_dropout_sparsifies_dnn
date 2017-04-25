# (WIP) Variational Dropout Sparsifies Deep Neural Networks
The code includes variational dropout linear layer to sparicy deep neural networks.
It will replicate experiments in the paper below  
```
Variational Dropout Sparsifies Deep Neural Networks.  
Dmitry Molchanov, Arsenii Ashukha, Dmitry Vetrov.  
Under review by ICML 2017.  
https://arxiv.org/pdf/1701.05369.pdf
```

This is based on the paper and the authors' [repository](https://github.com/ars-ashuha/variational-dropout-sparsifies-dnn), which uses theano.  
This implementation uses new Chainer version 2 (https://github.com/pfnet/chainer/tree/_v2 or https://github.com/pfnet/chainer/releases/tag/v2.0.0b1).
(Master branch is https://github.com/pfnet/chainer, however this does not work on the old version.)

# Requirements

A few requirements obeys those of Chainer v2.

Note Chainer are planning the v2.0.0 release on May 16.
Until the day, you can install Chainer v2 by following  
```
pip install chainer --pre
```
If you use GPU (CUDA/cuDNN), also  
```
pip install cupy
```

# Examples

- MNIST: Fully-connected DNN for MNIST. The example is derived from the official MNIST example of Chainer v2.
