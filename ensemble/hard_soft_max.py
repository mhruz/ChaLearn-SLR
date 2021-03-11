import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

with open(r'e:\ZCU\JSALT2020\ensemble_fine\test.txt') as txt_file:
    ground_truth = txt_file.readlines()

gt_labels = []

for i, gt in enumerate(ground_truth):
    gt_labels.append(gt.split(';')[-1][:-1])

data_dir = r"e:\ZCU\JSALT2020\ensemble_fine"
predicted_csv = ['crop.csv', 'crop_new.csv', 'mask.csv', 'keyframe_mask.csv', 'keyframe_new.csv', 'mask_new.csv',
                 'keyframe_mask_new.csv', 'openpose_41b.csv',
                 'vle_4.csv', 'vle_3.csv']

confusion_matrices = {}
for model in predicted_csv:
    confusion_matrices[model] = np.load("confusion_matrix_{}.npy".format(model))

print(predicted_csv)
predicts = {}

final_predictions = {}

for pcsv in predicted_csv:
    _csv = pd.read_csv(os.path.join(data_dir, pcsv))
    predicts[pcsv] = _csv

for i, model in enumerate(predicts):
    preds = np.asarray(predicts[model])
    max_confs = np.max(preds, axis=1)
    predicted_class = np.argmax(preds, axis=1)
    conf_thresh = max_confs * 0.5

    final_predictions[model] = np.zeros_like(preds)

    for j, c in enumerate(final_predictions[model]):
        preds[j, :] = preds[j, :] >= conf_thresh[j]

    preds = preds.astype(np.float32)

    final_predictions[model] = preds

    # add from confusion matrix
    for j, c in enumerate(final_predictions[model]):
        confusion = confusion_matrices[model][predicted_class[j], :]
        confusion[predicted_class[j]] = 0.0
        c += 5 * confusion

for model in final_predictions:
    x = pd.DataFrame(final_predictions[model])
    x.to_csv("{}_hard_soft_max_v2.csv".format(model), index=False)


final_decision = np.zeros_like(final_predictions["crop_new.csv"])
for model in final_predictions:
    final_decision += final_predictions[model]

acc = 0
for j, d in enumerate(gt_labels):
    max_confs = np.argmax(final_decision, axis=1)
    d = int(d)
    if max_confs[j] == d:
        acc += 1

acc /= predicts['crop_new.csv'].shape[0]


