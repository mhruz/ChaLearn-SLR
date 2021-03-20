import argparse
import os
import chainer

import matplotlib

from chainer import serializers
from chainer import Reporter

from chains.hpoes_transformer import HpoesTransformer
from datasets.hpoes_dataset import HpoesDataset
from evaluation.hpoes_transformer_eval_function import HpoesTransformerEvaluationFunction, HpoesTransformerEvaluationFunctionCodalab, HpoesTransformerEvaluator, HpoesTransformerEvaluatorCodalab
from hooks.noam_hook import NoamOptimizer
from updater.hpoes_transformer_updater import HpoesTransformerUpdater


######################################
#hlavni funkce pro trenovani s validaci
def main_train(args):
  
    #nevim zda je toto potreba
    device = chainer.get_device(args.gpu)
    device.use()
    
    ###########################
    # prepare the datasets and iterators
    train_dataset = HpoesDataset(data_dir=args.data_dir, input_size=args.input_size, max_len=args.max_len, isval=False)
    if args.debug_data == 1:
        train_dataset.N = 500
    val_dataset = HpoesDataset(data_dir=args.data_dir, input_size=args.input_size, max_len=args.max_len, isval=True)
    
    
    train_iter = chainer.iterators.MultithreadIterator(train_dataset, args.batch_size, shuffle=True, n_threads=12)
    #train_iter = chainer.iterators.SerialIterator(train_dataset, args.batch_size, shuffle=False)
    val_iter = chainer.iterators.MultithreadIterator(val_dataset, args.batch_size, repeat=False, shuffle=False, n_threads=12)
    #val_iter = chainer.iterators.SerialIterator(val_dataset, args.batch_size, repeat=False, shuffle=False)

    ################################
    # build the network we want to train
    ##                      max_len=200,   input_size=100,  vocab_size=226  N=6,             transformer_size=512,                   ff_size=2048,         num_heads=8,  dropout_ratio=0.1
    net = HpoesTransformer(args.max_len, args.input_size, args.vocab_size, N=args.N_stages, transformer_size=args.transformer_size,
                           ff_size=args.ff_size, num_heads=args.num_heads, dropout_ratio=0.1, enc_learn=args.enc_learn)
   
         
    #######################
    # build the optimizer
    print('using:', args.optimizer)
    if args.optimizer == 'ADAM':
        optimizer = chainer.optimizers.Adam(alpha=0.0, beta1=0.9, beta2=0.98, eps=1e-9)
        #optimizer = chainer.optimizers.Adam(alpha=0.0001, beta1=0.9, beta2=0.98, eps=1e-9)
    
    if args.optimizer == 'SGD':
        #optimizer = chainer.optimizers.SGD(lr=0.005) #s 0.005 to jede dobre
        #optimizer = chainer.optimizers.CorrectedMomentumSGD(lr=args.learning_rate, momentum=0.9) #zkurveny momentum
        optimizer = chainer.optimizers.SGD(lr=args.learning_rate)
   
       
    optimizer.setup(net)
    
    #optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(5.0)) #.... to prevent gradient explosion ....asi orezava gradienty .. args.gradclip treba prozkoumat co to dela 
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4)) #lehle kazeni vah malym cisel, prevence overfitingu
   
    
    ########################
    # create our custom updater that computes the loss and updates the params of the network
    updater = HpoesTransformerUpdater(train_iter, optimizer, device=args.gpu)

    # and init the trainer
    trainer = chainer.training.Trainer(updater, (args.epochs, 'epoch'), out=args.model_dir)
    
    
    #################
    # Learning rate decay ExponentialShift nasobi cislem promenou
    if args.optimizer == 'SGD':
        #trainer.extend(chainer.training.extensions.ExponentialShift('lr', 0.1), trigger=chainer.training.triggers.ManualScheduleTrigger([50,75],'epoch'))
        trainer.extend(chainer.training.extensions.ExponentialShift('lr', 0.9), trigger=(1,'epoch'))
    
    
    #################################
    # this hook is very important! Without it, the training won't converge! 
    # napr NoamOptimizer(2000, 10.0, net.transformer_size) heatsteps 1000 je prudke zahrati a lr vyleti vys, heatspeps 2000 je pozvolnejsi zahrati a lr tolik nevyleti
    # mene kroku zahrati znamena vetsi peek lr na zacatku, peak je skoro linearni
    # scale 10.0 urcuje absolutni velikost lr, je to nezavisle na poctu kroku zahrati, but to klesa a vesti vysky nebo z mensi, ale pak to do konce epochy jede stejne
    # vetsi transformer size znamena mensi lr ... jde proti scale
    #tot vse, zadna veda, 4000,50,512 vystoupa lr na nejakych 0.03 a pak jde dolu, 
    if args.optimizer == 'ADAM':
        optimizer.add_hook(
            NoamOptimizer(500, args.learning_rate * 500.0, net.transformer_size)
        )
        
    ########################
    # Log reporty
    trainer.extend(chainer.training.extensions.LogReport())
    trainer.extend(chainer.training.extensions.observe_lr(), trigger=(1, 'epoch'))
    trainer.extend(chainer.training.extensions.PrintReport(['epoch', 'loss', 'train/accuracy', 'validation/accuracy','lr']), trigger=(1, 'epoch')) #chainer.training.triggers.IntervalTrigger(5,'iteration')
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))
    
    #import matplotlib #nejak vymyslet pro metacentrum
    matplotlib.use('Agg')
    #trainer.extend(chainer.training.extensions.PlotReport(['loss', 'train/accuracy', 'validation/accuracy'], x_key='iteration', file_name='loss.png', trigger=(500,'iteration')), trigger=(500,'iteration'))
    trainer.extend(chainer.training.extensions.PlotReport(['train/accuracy', 'validation/accuracy'], x_key='epoch', file_name='loss.png'))
    #trainer.extend(chainer.training.extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    #trainer.extend(chainer.training.extensions.dump_graph('main/loss'))
    
    
    #####################
    # Take a snapshot of Trainer every epoch
    #trainer.extend(chainer.training.extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=(1, 'epoch'))
    
    
    #####################
    # Take the best snapshot of the model
    #record_trigger = chainer.training.triggers.MaxValueTrigger('validation/accuracy', (1, 'epoch'))
    #trainer.extend(chainer.training.extensions.snapshot_object(net, filename='model_epoch-{.updater.epoch}'), trigger=record_trigger)
    #trainer.extend(chainer.training.extensions.snapshot_object(net, filename='model_epoch-{.updater.epoch}'))
    best_trigger = chainer.training.triggers.MaxValueTrigger('validation/accuracy', (1, 'epoch'))
    trainer.extend(chainer.training.extensions.snapshot_object(net, 'best_model.npz'), trigger=best_trigger) #vzdy prepise
    
    
    #####################
    # create the evaluator
    eval_function = HpoesTransformerEvaluationFunction(net, args.gpu)
    trainer.extend(HpoesTransformerEvaluator(val_iter, net, device=args.gpu, eval_func=eval_function))

    
    #########################
    ## Pokracuj v trenovani
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
        
    if args.resume != "":
               
        chainer.serializers.load_npz(args.resume, trainer) #learning rate (tedy u SGD) to vubec neresi, to je divne
        print('resume', args.resume)
        
    
    
    ##########################
    # Zacni trenovani
    trainer.run()


#############################
#hlavni funkce pro predikci
def main_pred(args):
    
    val_dataset = HpoesDataset(data_dir=args.data_dir, input_size=args.input_size, max_len=args.max_len, isval=True, isforcodalab=True)
    #val_dataset = HpoesDataset(data_dir=args.data_dir, input_size=args.input_size, max_len=args.max_len, isval=True, isforcodalab=False)
    
    #val_iter = chainer.iterators.SerialIterator(val_dataset, args.batch_size, repeat=False, shuffle=False)
    val_iter = chainer.iterators.MultithreadIterator(val_dataset, args.batch_size, repeat=False, shuffle=False, n_threads=12)

    # build the network we want to train
    #                      max_len,      input_size,      vocab_size,      N=6,             transformer_size=512,                   ff_size=2048,         num_heads=8, dropout_ratio=0.1
    net = HpoesTransformer(args.max_len, args.input_size, args.vocab_size, N=args.N_stages, transformer_size=args.transformer_size, ff_size=args.ff_size, num_heads=args.num_heads)

    #serializers.load_npz('/home/dnn-user/mnt/pole/data-ntis/projects/cv/ChaLearnLAP/experiments/transformer/chainer-transformer/train01/model_epoch-35', net)
    serializers.load_npz(args.model_name, net)
    
    net.to_gpu(device=args.gpu)
    
    
    eval_function = HpoesTransformerEvaluationFunctionCodalab(net, args.gpu)
    evaluator = HpoesTransformerEvaluatorCodalab(val_iter, net, device=args.gpu, eval_func=eval_function)
    
    reporter = Reporter()
    a = 0
    reporter.add_observer('classID', a)
    
    observation = {}
    with reporter.scope(observation):
    
        print(evaluator.evaluate())
              

#################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the transformer under a copy task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-g", "--gpu", default=-1, type=int, help="gpu device to use (negative value indicates cpu)")
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size for training")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs to train")
    parser.add_argument("-v", "--vocab-size", type=int, default=226, help="number of different classes") #to je dobre, 226 trid indexovano od 0
    parser.add_argument("-is", "--input-size", type=int, default=100, help="size of data vector")
    parser.add_argument("-ts", "--transformer-size", type=int, default=512, help="size of transformer < input-size ??")
    parser.add_argument("-N",  "--N-stages", type=int, default=6, help="N stages")
    parser.add_argument("-fs", "--ff-size", type=int, default=2048, help="full conected size")
    parser.add_argument("-nh", "--num-heads", type=int, default=8, help="num heads")  #model size musi byt delitelny poctem hlav 512/8=64
    parser.add_argument("-ml", "--max-len", type=int, default=120, help="Max len sequence")
    parser.add_argument("-md", "--model-dir", type=str, default='train', help="where save model")
    parser.add_argument("-dd", "--data-dir", type=str, default='./', help="where are data")
    parser.add_argument("-ip", "--train-pred", type=str, default='none', help="if compute prediction")
    parser.add_argument("-mn", "--model-name", type=str, default='', help="name of the model")
    parser.add_argument("-re", "--resume", type=str, default='', help="Resume the training from snapshot")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("-bu", "--backbone-update", type=int, default=0, help="Learning rate")
    parser.add_argument("-op", "--optimizer", type=str, default='SGD', help="SGD/ADAM")
    parser.add_argument("-de", "--debug-data", type=int, default=0, help="just 500 samples")
    parser.add_argument("-el", "--enc-learn", type=int, default=1, help="Learnebale positional encoding")
    
    
    
    args = parser.parse_args()
    
    if args.train_pred == 'train':
        main_train(args)
    
    if args.train_pred == 'pred':
        main_pred(args)
