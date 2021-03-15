import chainer
import chainer.functions as F
from chainer import reporter
from chainer.backends import cuda
from chainer.training.extensions import Evaluator
import numpy as np
try:
  from cupy import bincount
except ImportError:
  from numpy import bincount


import pandas as pd
from chainer.backends.cuda import to_cpu

class HpoesTransformerEvaluationFunction:

    def __init__(self, net, device):
        self.net = net
        self.device = device

    def __call__(self, **kwargs):
        data = kwargs.pop('data')
        labels = kwargs.pop('label')
        
        
        ##with chainer.using_config('train', False):
        #x = self.net.backbone.features(data)
        #x = x.data
        #print('chainer.config.train', chainer.config.train)
        
        predictions, _ = self.net.predict(data)
        
        #print(predictions.shape)
        #exit()
        
        #zde uz jsou id classes
        
        #batch_size, num_steps = predictions.shape
        
        ##pokud chxi pro vsechny
        #predictions = F.reshape(predictions, (-1, vocab_size))
        #labels = labels.repeat(num_steps, axis=1)
        
        labels = labels.ravel()
        
        #print(labels.shape, labels.dtype)
        
        ##print(prediction[0])
        #exit()
        
        ## part accuracy is the accuracy for each number and accuracy is the accuracy
        ## for the complete vector of numbers
        #part_accuracy, accuracy = self.calc_accuracy(predictions, labels) #stara varianta
        accuracy = self.calc_accuracy_one(predictions, labels)

        reporter.report({
            "part/accuracy": 0,
            "validation/accuracy": accuracy
        })

    #################
    def calc_accuracy_one(self, predictions, labels):
            
        accuracy_result = (predictions == labels).sum()
        
        return accuracy_result / len(predictions)
    
    ####################
    def calc_accuracy(self, predictions, labels):
        correct_lines = 0
        correct_parts = 0
        for predicted_item, item in zip(predictions, labels):
            #item je jedno ID pro tuto polozku batche
            #predicted_item je 200 voleb
            # count the number of correct numbers
            accuracy_result = (predicted_item == item).sum()
            correct_parts += accuracy_result

            # if all numbers are correct, we can also increase the number of correct lines/vectors
            if accuracy_result == predictions.shape[1]: #vsechny volby spravne
                correct_lines += 1

        return correct_parts / predictions.size, correct_lines / len(predictions)

    


class HpoesTransformerEvaluationFunctionCodalab:

    def __init__(self, net, device):
        self.net = net
        self.device = device
        
        

    def __call__(self, **kwargs):
        data = kwargs.pop('data')
        #labels = kwargs.pop('label') #nemam k dispozici
        #print(data.dtype)
        
        ############### ma to tu vyznam, dava jine vysledky
        chainer.config.train = False
        ###############
        
        #predictions = self.net.predict(data)
        predictions, x = self.net.predict(data)
        
        #print(predictions.shape)
        #exit()
        #batch_size, num_steps = predictions.shape
        
        for predicted_item in predictions:
            
            id = int(predicted_item)
            #print(id)
            
            reporter.report({
                "classID": id
            })
        return x
    
        #funkcni, pokud mi predikuji vsechny vektory v sekvenci, to byl stary zpusob
        #for predicted_item in predictions:
            ##item je jedno ID pro tuto polozku batche
            ##predicted_item je 200 voleb
            #print(predicted_item)
            #exit()
            #bc = bincount(predicted_item)
            
            #id = int(bc.argmax())
            #print(id)
            
            #reporter.report({
                #"classID": id
            #})

    
    
    
###################################
class HpoesTransformerEvaluator(Evaluator):

    def evaluate(self):
        summary = reporter.DictSummary()
        eval_func = self.eval_func or self._targets['main']
        iterator = self.get_iterator('main')
        iterator.reset()
        observation = {}
        X = None
        with reporter.report_scope(observation):
           # we always use the same array for testing, since this is only an example ;)
           count = 0 
           while True:
              try:
                  batch = next(iterator)
              except StopIteration:
                  break
              
              count += 1
        
              batch = self.converter(batch, self.device, padding=0.0) #padding=0.0 doplni nulama na stejnou velikost)
                            
              eval_func(data=batch['data'], label=batch['label'])
              #x = eval_func(data=batch['data'], label=batch['label'])
              #if X is None:
                  #X = x
              #else:
                  #X = F.concat((X, x), axis=0)
              
     
              summary.add(observation)
              
        #X = pd.DataFrame(to_cpu(X.data))
        #X.to_csv('vle_4_test.csv', index=False)
        
        return summary.compute_mean()

###################################
class HpoesTransformerEvaluatorCodalab(Evaluator):

    def evaluate(self):
        summary = reporter.DictSummary()
        eval_func = self.eval_func or self._targets['main']
        iterator = self.get_iterator('main')
        iterator.reset()
        observation = {}
        X = None
        with reporter.report_scope(observation):
           # we always use the same array for testing, since this is only an example ;)
           count = 0 
           while True:
              try:
                  batch = next(iterator)
              except StopIteration:
                  break
              
              count += 1
        
              batch = self.converter(batch, self.device, padding=0.0) #padding=0.0 doplni nulama na stejnou velikost)
                            
              #eval_func(data=batch['data'], label=batch['label'])
              x = eval_func(data=batch['data'], label=batch['label'])
              if X is None:
                  X = x
              else:
                  X = F.concat((X, x), axis=0)
              
     
              summary.add(observation)
              
        X = pd.DataFrame(to_cpu(X.data))
        X.to_csv('vle.csv', index=False)
        
        return summary.compute_mean()
