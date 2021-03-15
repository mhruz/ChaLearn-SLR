import math

import chainer
import chainer.links as L
import chainer.functions as F

from chainer import Chain, initializers

import numpy

class Embedding(Chain):
    """
        The Embedding used for the transformer
    """

    def __init__(self, size, vocab_size):
        super().__init__()
        self.size = size

        with self.init_scope():
            self.embed = L.EmbedID(vocab_size, size, initialW=initializers.GlorotUniform())

    def __call__(self, x):
        return self.embed(x) * math.sqrt(self.size)




class SlrEmbedding(Chain):
    """
        The Embedding used for the transformer
    """

    def __init__(self, inputsize, outputsize):
        super().__init__()
        self.inputsize = inputsize
        self.outputsize = outputsize
        w = chainer.initializers.Normal(scale=0.001)
        W = numpy.eye(outputsize,inputsize, dtype=numpy.float32)
        with self.init_scope():
            
            self.fc1 = L.Linear(inputsize, outputsize, initialW=initializers.GlorotUniform()) #puvodni
            #self.fc1 = L.Linear(inputsize, outputsize, initialW=initializers.GlorotUniform(), nobias=True) #nobias by mel mit duvod, s nim to fungovalo, bez nej zatim ne
            #self.fc1 = L.Linear(inputsize, outputsize, initialW=W) # inicializace jednotkovou matici
            
            #self.bn_fc1 = L.BatchNormalization(outputsize, axis=(0,1)) #normalizuj pres osu 0 a 1, pokusne, ale jde spocitat
            #self.norm = L.LayerNormalization(self.inputsize)
            
          

    def __call__(self, x):
        
        ##normalizace je zabijak, po 5 epose prestane konvegovat
        #batch_size, num_steps, size = x.shape
        #normed_x = self.norm(F.reshape(x, (-1, size)))
        #x = F.reshape(normed_x, (batch_size, num_steps, size))
        
        
        dropout_ppt = 0.3 #bylo a fungovalo
        #dropout_ppt = 0.1 #asi moc nejde
        #x = F.dropout(F.relu(self.bn_fc1(self.fc1(x, n_batch_axes=2))), dropout_ppt)
         
        #x = F.dropout(F.relu(self.fc1(x, n_batch_axes=2)), dropout_ppt) #relu, to co je na vstupu pouzti na vystup kladne nebo nulu
        #x = F.relu(self.fc1(x, n_batch_axes=2)) #jen relu
        #x = F.dropout(self.fc1(x, n_batch_axes=2), dropout_ppt) #linear projection
        x = self.fc1(x, n_batch_axes=2) #prima linear projection nepotrebuje bias ... logicky ... pozor, takto je to je to pro train03 a 04
        
        #return x * math.sqrt(self.outputsize) #dle originalu scaluju, ale proc?? to tam je
        return x

            
