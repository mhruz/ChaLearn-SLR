
DATADIR="/home/data/"

OUTPUTDIR="/home/data/models/Pose-transformer/"
mkdir -p $OUTPUTDIR

WORKINGDIR="/home/experiment/transformer/"
cd $WORKINGDIR

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
                                             --model-name $OUTPUTDIR/model_best
                                             
mv $WORKINGDIR/openpose.csv  /home/data/test_csv/openpose.csv 
