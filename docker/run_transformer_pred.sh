

################
# ViT Transformers training

mkdir -p /home/data/test_csv/

######
#OPENPOSE

##train
#./transformer/run_train.sh
##pred
./transformer/run_pred.sh
cp ./transformer/openpose_41b.csv /home/data/test_csv/

####
#VLE

##train
#./transformer-vle/run_train_3.sh
##pred
#./transformer-vle/run_pred_3.sh
#cp ./transformer-vle/vle.csv /home/data/test_csv/vle_3.csv

##train
#./transformer-vle/run_train_4.sh
##pred
#./transformer-vle/run_pred_4.sh
#cp ./transformer-vle/vle.csv /home/data/test_csv/vle_4.csv

##train
#./transformer-vle/run_train_12.sh
##pred
#./transformer-vle/run_pred_12.sh
#cp ./transformer-vle/vle.csv /home/data/test_csv/vle_12.csv

##########