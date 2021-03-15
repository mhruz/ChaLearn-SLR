
import chainer.functions as F
import chainer.links as L
from chainer import Chain

from transformer import slr_get_encoder
from transformer.utils import subsequent_mask

#from chainercv2.model_provider import get_model as chcv2_get_model


class HpoesTransformer(Chain):
    """
        This class shows how the transformer could be used.
        The copy transformer used in our example consists of an encoder/decoder
        stack with a stack size of two. We use greedy decoding during test time,
        to get the predictions of the transformer.

        :param vocab_size: vocab_size determines the number of classes we want to distinguish.
        Since we only want to copy numbers, the vocab_size is the same for encoder and decoder.
        :param max_len: determines the maximum sequence length, since we have no end of sequence token.
        :param start_symbol: determines the begin of sequence token.
        :param transformer_size: determines the number of hidden units to be used in the transformer
    """

    def __init__(self, max_len, input_size, vocab_size, N=6, transformer_size=512, ff_size=2048, num_heads=8, dropout_ratio=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.max_len = max_len
        self.transformer_size = transformer_size
        
        with self.init_scope():
            
            ##kdyz je tady, tak ji optimizer hodi na GPU a asi ji bude chtit optimizovat
            #backbone = chcv2_get_model("resnet50", in_channels=3, pretrained=True)
            ##with net.backbone.features.init_scope():
            #delattr(backbone.features, "final_pool")
            #delattr(backbone.output, "flatten")
            #delattr(backbone.output, "fc")
            #self.backbone = backbone
            
            #slr_get_encoder(input_size, N=6, model_size=512, ff_size=2048, num_heads=8, dropout_ratio=0.1):
            model = slr_get_encoder(
                input_size,
                N=N,
                model_size=self.transformer_size,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout_ratio=dropout_ratio,
                max_len=max_len
            )
            
            self.model = model
            #self.mask = subsequent_mask(self.transformer_size) #vytvorena zde jako numpy ale kdyz chainer prejde na gpu prevede asi vse v ramci initscope
            self.classifier = L.Linear(self.transformer_size, vocab_size)

    def __call__(self, x):
        
        result = self.model(x, None)
        #z modelu vypadne feature velikosti transformer_size
        #a ten se linearne transformuje na velikost slovniku, n_batch_axes=2 jsou dve dimenze na batch
        #print('transformer output', result.shape)
        
        #vezmu jen prvni vektor z prosle sekvence, viz dle ViT, MHR rada
        
        #return self.classifier(result, n_batch_axes=2)
        return self.classifier(result[:,0])

    
    
    #######################
    # vola evaluator
    
    def decode_prediction(self, x):
        """
            helper function for greedy decoding
        :param x: the output of the classifier
        :return: the most probable class index
        udela ze 100 odezev slovniku to slovo, ktere je nejpravdepodobnejsi
        """
        #x = F.reshape(x, (x.shape[0], -1))
        #print(x.shape)
        ##exit()
        
        #return F.argmax(F.softmax(x, axis=2), axis=2) #naraz pro batch i pro sekvenci
        x = F.softmax(x, axis=1)
        return F.argmax(x, axis=1), x
        
    def predict(self, x):
        """
            This method performs greedy decoding on the input vector x
        :param x: the input data that shall be copied.
        :return: the (hopefully) copied data.
        """
        
        #x = self.backbone.features(x)
        #x = x.data
        
        
        # first, we use the encoder on the input data
        prediction = self.model.encode(x, None)
        
        #prediction = self.classifier(prediction, n_batch_axes=2) #klasifikece pro vsechny, stary zpusob
        prediction = self.classifier(prediction[:,0]) #klasifikece pro vsechny, stary zpusob
        
        #print('prediction.shape', prediction.shape)
        #exit()
        
        decoded, x = self.decode_prediction(prediction)
        #print(decoded.dtype)
        #exit()
        
        return decoded.data, x.data
