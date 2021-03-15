import chainer
import chainer.functions as F
import chainer.links as L
import math
import numpy as np

from chainer import Chain, initializers


class PositionalEncoding(Chain):
    """
        Positional encoding, based on sin and cos functions as proposed in section 3.5
        of the paper "Attention is all you need"
    """

    def __init__(self, size, dropout_ratio=0.1, max_len=5000):
        super().__init__()
        self.size = size
        self.dropout_ratio = dropout_ratio
        self.max_len = max_len

        self.positional_embedding = np.zeros((max_len, size), dtype=chainer.get_dtype())
        position = np.arange(0, max_len)[..., np.newaxis]
        div_term = np.exp(np.arange(0, size, 2) / size * math.log(10000))

        self.positional_embedding[:, 0::2] = np.sin(position / div_term)
        self.positional_embedding[:, 1::2] = np.cos(position / div_term)
        self.positional_embedding = self.positional_embedding[np.newaxis, ...]

    def __call__(self, x):
        positional_embedding = chainer.Variable(self.positional_embedding[:, :x.shape[1]], requires_grad=False)
        positional_embedding.to_device(self.device)
        positional_embedding = F.broadcast_to(positional_embedding, (len(x),) + positional_embedding.shape[1:])
        x = x + positional_embedding
        return F.dropout(x, ratio=self.dropout_ratio)



class PositionalEncodingLearnable(Chain):
    """
        Positional encoding, based on sin and cos functions as proposed in section 3.5
        of the paper "Attention is all you need"
    """

    def __init__(self, size, dropout_ratio=0.1, max_len=5000):
        super().__init__()
        self.size = size
        self.dropout_ratio = dropout_ratio
        self.max_len = max_len

        with self.init_scope():
            self.position_enc = L.EmbedID(self.max_len, self.size, initialW=initializers.GlorotUniform())

       

    def __call__(self, x):
        #print(x.shape)
        #positional_embedding = chainer.Variable(self.positional_embedding[:, :x.shape[1]], requires_grad=False)
        seq = chainer.Variable(np.arange(x.shape[1]), requires_grad=False)
        seq.to_device(self.device)
        positional_embedding = self.position_enc(seq)
        
        #print(positional_embedding.shape)
        
        positional_embedding = F.broadcast_to(positional_embedding, x.shape)
       
        #print(positional_embedding.shape)
        #exit()
        
        x = x + positional_embedding
        
        return F.dropout(x, ratio=self.dropout_ratio)
