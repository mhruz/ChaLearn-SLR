import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(226)
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


data_dir = './data/ensemble_fine/'

crop = pd.read_csv(data_dir + 'crop.csv')
keyframe = pd.read_csv(data_dir + 'keyframe.csv')
mask = pd.read_csv(data_dir + 'mask.csv')
keyframe_mask = pd.read_csv(data_dir + 'keyframe_mask.csv')

with open('./data/test.txt') as txt_file:
    ground_truth = txt_file.readlines()

correct = 0
total = 0

w = [1, 1, 1, 1]    # tyto vahy chceme optimalizovat vzhledem k final accuracy
cmt = np.zeros((226,226), dtype=int)
for i, gt in enumerate(ground_truth):
    gt_label = gt.split(';')[-1][:-1]

    res0 = np.array(crop.iloc[i])
    res1 = np.array(keyframe.iloc[i])
    res2 = np.array(mask.iloc[i])
    res3 = np.array(keyframe_mask.iloc[i])

    final_res = (w[0] * res0 + w[1] * res1 + w[2] * res2 + w[3] * res3) / len(w)
    prediction = final_res.argmax(0)
    cmt[int(gt_label), int(prediction)] +=1
    if int(prediction) == int(gt_label):
        correct += 1
    total += 1


print('Final accuracy: %.4f' %(100*correct / total))
plt.figure(figsize=(50,50))
plot_confusion_matrix(cmt)
plt.show()