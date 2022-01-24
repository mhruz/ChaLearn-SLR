import numpy as np
import pandas as pd
import h5py
import os

df_train = pd.read_csv('./data/final/test_labels.csv', sep=',', header=None)
video_names = df_train[0]
labels = df_train[1]
f_joints = h5py.File('./data/final/test_json_keypoints-raw.h5', 'r')

# /auto/plzen4-ntis/projects/korpusy_cv/AUTSL/train_crop
with open('data/final/test.txt', "w") as txt_file:
    for (i,video_name) in enumerate(video_names):
        video_len = f_joints[video_name + '_color'].shape[0]-1
        label = labels[i]
        a, b = video_name.split('_')
        video_dir = '/storage/plzen4-ntis/projects/korpusy_cv/AUTSL/test_crop/'+a+'/'+b
        txt_file.write(video_dir+';0;'+str(video_len)+';'+str(label)+'\n')
        print(video_dir+';0;'+str(video_len)+';'+str(label)+'\n')


