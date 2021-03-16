

##########################
# ViT Transformers predict

mkdir -p /home/data/test_csv/

######
#OPENPOSE

##train
#./transformer/run_train.sh
##pred
./transformer/run_pred.sh

####
#VLE

##train
#./transformer-vle/run_train_3.sh
##pred
./transformer-vle/run_pred_3.sh

##train
#./transformer-vle/run_train_4.sh
##pred
./transformer-vle/run_pred_4.sh

##train
#./transformer-vle/run_train_12.sh
##pred
./transformer-vle/run_pred_12.sh
