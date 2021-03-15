# chainer-transformer

This repository contains an example implementation of the transformer, 
as published in the [Attention is all you need](https://arxiv.org/abs/1706.03762) paper.
This implementation is based on the PyTorch implementation by 
[OpenNMT](http://nlp.seas.harvard.edu/2018/04/03/attention.html).

The code for the transformer can be found in the directory `transformer`.
Apart from the transformer code, this repository contains an example that
shows how to use the transformer in your code.
The example is a simple copy task that can be learned by the transformer.

## Using the Example

If you want to try the example, you'll need to do the following:

0. create a new virtual environment, using Python 3 (preferably >3.6)
. Python 2 might also work, but this is not guaranteed.
1. install all requirements via `pip install -r requirements.txt`
1. start the training with `python train_copy.py`. If you want to use
a GPU for training, you can specify the GPU with `-g`. For further options
type `python train_copy.py -h`.
1. wait some minutes and you should get a transformer that can copy digits.  


