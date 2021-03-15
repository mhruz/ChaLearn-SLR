import chainer
import chainer.functions as F
import numpy as np

from chainer.training import StandardUpdater


class HpoesTransformerUpdater(StandardUpdater):

    def update_core(self):
        with self.device.device:
            self.update_net()

    def update_net(self):
        
        batch = next(self.get_iterator('main'))
        
        ######################
        #zde je misto na augmentaci na CPU
        #####################
        
        
        #print(batch[0]['data'])
        #exit()
        
        
        #zde je batch jeste list distionary na CPU jako numpy 
        batch = self.converter(batch, self.device, padding=0.0) #padding=0.0 doplni nulama na stejnou velikost)
        #zde uy je to dictionary ndarray na GPU jako cupy
        
        ##########################
        #nebo zde je misto na augmentaci na GPU
        #########################
        
        #print(batch['data'].shape)
        #exit()
     
        
        optimizer = self.get_optimizer('main')
        net = optimizer.target

        ####################
        ## backbone zatim nepotrebujeme
        ##x = x.data.squeeze() #odstrani dimenze velikosti 1
        #with chainer.using_config('train', False):
            #x = net.backbone.features(batch['data'])
        #x = x.data
        
        #print('x', x.shape)
        #exit()
        
        x = batch['data']
        
        #print('chainer.config.train', chainer.config.train)
        
        predictions = net(x)
        
        #batch_size, num_steps, vocab_size = predictions.shape #uz ne, klasifikuji je 1. vektor z prosle sekvence 
        #print(predictions.shape)
        #exit()
        
        #overeno, sklada to dobre
        #predictions = F.reshape(predictions, (-1, vocab_size))
        
        #print('uz jsem zde', predictions )
        #exit()

        #ravel() #nedela kopii #flateten jo
        
        labels = batch['label']
        #print('uz jsem zde', labels.shape)
        #exit()
        
        ##pokud chci naklonovat classID pro vsechny vektory sekvence
        #labels = labels.repeat(num_steps, axis=1)
        labels = labels.ravel()
        
        #print(labels.dtype) #ano je tu int32
        #print(labels.shape)
        #exit()
        
        

        loss = F.softmax_cross_entropy(predictions, labels)
        accuracy = F.accuracy(F.softmax(predictions), labels)

        net.cleargrads()
        loss.backward()
        optimizer.update()
        
        
        #print(err)

        chainer.reporter.report({
            "loss": loss,
            "train/accuracy": accuracy
        })
