
DATADIR="/home/data/"

OUTPUTDIR="/home/data/models/transformer"
mkdir -p $OUTPUTDIR

WORKINGDIR="/home/experiment/transformer"
cd $WORKINGDIR

# finetunign a pred pro train41
python3 ./chainer-transformer/train_hpoes.py --train-pred pred \
                                             --gpu 0 \
                                             --data-dir $DATADIR \
                                             --batch-size 64 \
                                             --epochs 100 \
                                             --max-len 120 \
                                             --input-size 84 \
                                             --N-stages 2 \
                                             --transformer-size 1024 \
                                             --ff-size 2048 \
                                             --num-heads 2 \
                                             --model-dir $OUTPUTDIR \
                                             --learning-rate 0.1 \
                                             --optimize-alg SGD \
                                             --resume $OUTPUTDIR/snapshot_epoch_87 \
                                             --model-name $OUTPUTDIR/best_model.npz
                                             
#                                              --model-name "/model_epoch-88" 
                                             
