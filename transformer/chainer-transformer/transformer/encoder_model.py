from chainer import Chain
import chainer.functions as F

from transformer.parameter import ParameterLayer

class EncoderModel(Chain):
    """
        Chain that combines encoder and decoder
    """

    def __init__(self, encoder, src_embeddings, model_size):
        super().__init__()
        self.model_size = model_size
        with self.init_scope():
            self.encoder = encoder
            self.src_embeddings = src_embeddings
            self.patch = ParameterLayer((1, 1, model_size))
            

    def __call__(self, src, src_mask):
        """
            Perform forward pass thorugh the transformer
        :param src: input for the encoder
        :param src_mask: mask for guiding the encoder
        :return: the output of the transformer
        """
        return self.encode(src, src_mask)

    def encode(self, src, src_mask):
        #################
        #priprava dat
        
        #src_embeddings() dela jak embeding, tak i positional encoding
        x = self.src_embeddings(src)
        
        batch_size, n_seq, n_channel = x.shape
        
        ptch = self.patch()
        ptch = F.broadcast_to(ptch,(batch_size, 1, self.model_size))
        
        #pridam na zacatek kazde sekvence v batch ten samy parametr "patch"
        x = F.concat((ptch, x), axis=1)
        
        #################
        # a encoding
        return self.encoder(x, src_mask)

        #return self.encoder(self.src_embeddings(src), src_mask)

    
