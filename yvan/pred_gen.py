import pandas as pd

BASE_DIR = "../data/"
output_dir = 'D:/Work/Challenges/Action_Recognition/results/26/'

results = pd.read_csv('./data/predictions.csv', sep=',', header=None)
pred = pd.read_csv(output_dir+'softmax_values.csv', sep=',')

for i in range(0,len(results)):
    label = pred.iloc[i].argmax(0)
    results[1][i] = label

results.to_csv(output_dir+"predictions.csv",index=False, header=None)

results = pd.read_csv(output_dir+'predictions.csv',header=None, names=['video','label'])

with open('./data/test.txt') as txt_file:
    ground_truth = txt_file.readlines()

correct = 0
total = 0
for i, gt in enumerate(ground_truth):
    res = results['label'][i]
    gt_label = gt.split(';')[-1][:-1]
    if int(res)==int(gt_label):
        correct+=1
    total+=1


print(correct/total)
