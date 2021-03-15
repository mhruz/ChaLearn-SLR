
cd /openpose
python3 /home/experiment/data_prep/openpose.py "/home/data/train/" "/home/data/train_json/"
python3 /home/experiment/data_prep/openpose.py "/home/data/val/" "/home/data/val_json/"
python3 /home/experiment/data_prep/openpose.py "/home/data/test/" "/home/data/test_json/"

