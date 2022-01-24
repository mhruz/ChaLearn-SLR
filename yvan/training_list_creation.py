import pandas as pd
import os

BASE_DIR = "../data/"

df_train = pd.read_csv(os.path.join(BASE_DIR, "labels_jpg.csv"), header=None, sep=' ', names=['img', 'org_video', 'frame_num', 'label', 'signer'])

video_ids = []
video_name = df_train.iloc[0]['org_video']
id = 0

# df_train['video_id'] = df_train.groupby(['org_video']).ngroup()
df_train['img_list'] = df_train.groupby(['org_video'])['img'].transform(lambda x: ','.join(x))
df_train = df_train.drop(columns=['org_video','frame_num','signer', 'img'])
df_train = df_train[['img_list','label']].drop_duplicates().reset_index().drop(columns=['index'])
df_train.to_csv(os.path.join(BASE_DIR, "train_list.csv"),index=False)
print('Done')