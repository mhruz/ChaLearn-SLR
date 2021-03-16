
DATADIR="/home/data/"

OUTPUTDIR="/home/data/models/VLE-transformer/"
mkdir -p $OUTPUTDIR

WORKINGDIR="/home/experiment/transformer-vle"
cd $WORKINGDIR

python3 ./chainer-transformer/train_hpoes.py --train-pred train \
                                             --gpu 0 \
                                             --data-dir $DATADIR \
                                             --batch-size 64 \
                                             --epochs 100 \
                                             --max-len 120 \
                                             --input-size 2590 \
                                             --N-stages 2 \
                                             --transformer-size 512 \
                                             --ff-size 2048 \
                                             --num-heads 2 \
                                             --model-dir $OUTPUTDIR \
                                             --learning-rate 0.1 \
                                             --optimizer SGD \
                                             --debug-data 0 \
                                             --resume ""

mv $OUTPUTDIR/best_model.npz $OUTPUTDIR/vle_12                                         
