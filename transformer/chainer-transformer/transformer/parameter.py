
import chainer
from chainer import initializers


class ParameterLayer(chainer.Link):

    def __init__(self, shape):
        super(ParameterLayer, self).__init__()
        
        with self.init_scope():
            #self.P = chainer.Parameter( initializers.Normal(), shape) 
            self.P = chainer.Parameter( initializers.GlorotUniform(), shape) 
            
            #self.W = chainer.Parameter(
                #initializers.Normal(), (n_out, n_in))
            #self.b = chainer.Parameter(
                #initializers.Zero(), (n_out,))

    def forward(self):
        #return F.linear(x, self.W) #, b=None, n_batch_axes=1)
        return self.P
