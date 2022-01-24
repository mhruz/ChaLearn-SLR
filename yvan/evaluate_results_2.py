import pandas as pd
import numpy as np

data_dir = './data/ensemble_cros/'

one = pd.read_csv(data_dir + '1.csv')
two = pd.read_csv(data_dir + '2.csv')
three = pd.read_csv(data_dir + '3.csv')
four = pd.read_csv(data_dir + '4.csv')
five = pd.read_csv(data_dir + '5.csv')


with open('./data/test.txt') as txt_file:
    ground_truth = txt_file.readlines()

correct = 0
total = 0

w = [1, 1, 1, 1, 1]    # tyto vahy chceme optimalizovat vzhledem k final accuracy

for i, gt in enumerate(ground_truth):
    gt_label = gt.split(';')[-1][:-1]

    res0 = np.array(one.iloc[i])
    res1 = np.array(two.iloc[i])
    res2 = np.array(three.iloc[i])
    res3 = np.array(four.iloc[i])
    res4 = np.array(five.iloc[i])

    final_res = (w[0] * res0 + w[1] * res1 + w[2] * res2 + w[3] * res3 + w[4] * res4) / len(w)
    prediction = final_res.argmax(0)

    if int(prediction) == int(gt_label):
        correct += 1
    total += 1

print('Final accuracy: %.4f' %(100*correct / total))
