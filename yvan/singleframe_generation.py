import h5py
import numpy as np
import pandas as pd

df_train = pd.read_csv('./data/train_list_keyframes.csv', sep=',')

all_res = []
res = {'id':"", 'label':"", 'frame':""}
poc = 0

for i in range(0, len(df_train)):
    id = df_train['id'].iloc[i]
    label = df_train['label'].iloc[i]
    keyframes = df_train['keyframes'].iloc[i]
    keyframes = keyframes[1:-1].split(',')
    for frame in keyframes:
        res = {'id': id, 'label': label, 'frame': int(frame)}
        all_res.append(res)
    poc +=1


new_df = pd.DataFrame(all_res)

new_df.to_csv("./data/train_list_singleframes.csv",index=False)
print('Done')