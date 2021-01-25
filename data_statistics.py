# Quick script for data statistics

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-core_directory', type=str, default='./Data/', help='Data directory')
    parser.add_argument('-csv_file', type=str, default='train_labels.csv', help='train csv file')
    args = parser.parse_args()
    BASE_DIR = args.core_directory

    df_train = pd.read_csv(os.path.join(BASE_DIR, "train_labels.csv"))
    label_num_min = df_train["label"].min()
    label_num_max = df_train["label"].max()
    labels = df_train["label"].value_counts()
    number_of_labels = len(labels)
    average_count = labels.mean()
    min = labels.min()
    max = labels.max()

    print('Number of labels: %i' %(number_of_labels))
    print('Average count of labels: %.2f' %(average_count))
    print('Average count of labels: %i' %(min))
    print('Minimal count of labels: %i' %(max))

    plt.figure(figsize=(20, 60))
    sn.countplot(y="label", data=df_train)
    plt.show()
