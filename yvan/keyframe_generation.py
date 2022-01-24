import h5py
import numpy as np
import pandas as pd
import os

BASE_DIR = "../data/"

f_joints = h5py.File('./data/train_json_keypoints-raw.h5', 'r')
video_fn_ref = "signer10_sample175_color.mp4"

df_train = pd.read_csv('./data/train_labels.csv', sep=',')
video_names = df_train['id']
labels = df_train['label']
df_train['keyframes'] = ""
df_train['keyframes'] = df_train['keyframes'].astype('object')

num_videos = 100
right_wrist = 4
left_wrist = 7
movement_threshold = 2
distance_from_start = 20
movement_all_threshold = 20
minimal_frame_step = 4

for (i,video_name) in enumerate(video_names):
    keyframes_indexes = []
    label = labels[i]
    joints_all = f_joints[video_name+'_color']
    first_frame = np.reshape(joints_all[0], (-1, 3))
    right_previous = first_frame[right_wrist,:2]
    left_previous = first_frame[left_wrist,:2]
    first_right = right_previous
    first_left = left_previous
    previous_all = first_frame[:,:2]
    last_saved_index = 0

    for (j, joint_frame) in enumerate(joints_all):
        joint_frame = np.reshape(joint_frame, (-1, 3))
        all = joint_frame[:,:2]
        right = joint_frame[right_wrist,:2]
        left = joint_frame[left_wrist,:2]

        dist_r = dist = np.linalg.norm(right - right_previous)
        dist_l = dist = np.linalg.norm(left - left_previous)

        dist_r_start = np.linalg.norm(right-first_right)
        dist_l_start = np.linalg.norm(left - first_left)

        dist_all = np.linalg.norm(all - previous_all)
        right_previous = right
        left_previous = left

        if dist_r_start<distance_from_start and dist_l_start<distance_from_start:
            continue

        if dist_r<movement_threshold and dist_l<movement_threshold:
            if dist_all> movement_all_threshold:
                if (j-last_saved_index) >= minimal_frame_step:
                    keyframes_indexes.append(j)
                    previous_all = all
                    last_saved_index = j

    if len(keyframes_indexes) == 0:
        keyframes_indexes = [int(np.round(j/5.0)), int(np.round(2*j/5.0)), int(np.round(3*j/5.0)), int(np.round(4*j/5.0))]

    # imgs_path = video_name.replace('_', '/')
    df_train.at[i, 'keyframes']= keyframes_indexes
    print(video_name + ': '+str(keyframes_indexes))
    # print(video_name +': '+str(len(keyframes_indexes)))
    # if i >= num_videos:
    #     break

df_train.to_csv("./data/train_list_keyframes.csv",index=False)