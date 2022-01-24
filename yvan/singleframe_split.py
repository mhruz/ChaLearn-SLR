import pandas as pd
import numpy as np

df_train = pd.read_csv('./data/train_list_singleframes.csv', sep=',')

# train_length = int(np.round((85*len(df_train)/100)))
train_length = 98006
train = df_train[:train_length]
val = df_train[train_length:]

train.to_csv("./data/train_list_singleframes_train.csv",index=False)
val.to_csv("./data/train_list_singleframes_val.csv",index=False)