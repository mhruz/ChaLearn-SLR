import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

with open(r'e:\ZCU\JSALT2020\ensemble_SL_sensors_2022\AUTSL_val.txt') as txt_file:
    ground_truth = txt_file.readlines()

gt_labels = []

for i, gt in enumerate(ground_truth):
    gt_labels.append(gt.split(',')[-1][:-1])

data_dir = r"e:\ZCU\JSALT2020\ensemble_SL_sensors_2022"
# predicted_csv = ['crop.csv', 'crop_new.csv', 'mask.csv', 'keyframe_mask.csv', 'keyframe.csv', 'keyframe_new.csv',
#                  'openpose_41b.csv', 'vle_4.csv', 'vle_3.csv', '1.csv', '2.csv', '3.csv', '4.csv', '5.csv']

# predicted_csv = ['crop_new.csv', 'keyframe_new.csv', 'mask_new.csv', 'keyframe_mask_new.csv', 'openpose_41b.csv',
#                  'vle_3.csv']

# predicted_csv = ['crop.csv', 'crop_new.csv', 'mask.csv', 'keyframe_mask.csv', 'keyframe_new.csv', 'mask_new.csv',
#                  'keyframe_mask_new.csv', 'openpose_41b.csv',
#                  'vle_4.csv', 'vle_3.csv']

# predicted_csv = ['spoter_mmpose_fixed.txt']
predicted_csv = os.listdir(data_dir)
predicted_csv = [pred for pred in predicted_csv if pred.endswith(".csv")]

print(predicted_csv)
predicts = {}

for pcsv in predicted_csv:
    _csv = pd.read_csv(os.path.join(data_dir, pcsv))
    predicts[pcsv] = _csv

acc_mat = np.zeros((predicts[predicted_csv[0]].shape[1], len(predicts.keys())))
class_counts = np.zeros((predicts[predicted_csv[0]].shape[1]))
confusion_matrices = np.zeros((len(predicts.keys()), predicts[predicted_csv[0]].shape[1], predicts[predicted_csv[0]].shape[1]))
model_picker = np.zeros((predicts[predicted_csv[0]].shape[0], len(predicts.keys())))
model_picker_acc = 0.0

class_picker = np.zeros_like(model_picker)

for i, model in enumerate(predicts):
    max_confs = np.argmax(np.asarray(predicts[model]), axis=1)

    class_picker[:, i] = np.max(np.asarray(predicts[model]), axis=1)
    for j, d in enumerate(gt_labels):
        d = int(d)
        confusion_matrices[i, max_confs[j], d] += 1
        if max_confs[j] == d:
            acc_mat[d, i] += 1
            model_picker[j, i] += 1

        if i == 0:
            class_counts[d] += 1

    acc_mat[:, i] /= class_counts

final_decision = []
for i, best_model in enumerate(np.argmax(class_picker, axis=1)):
    final_decision.append(np.argmax(np.asarray(predicts[predicted_csv[best_model]])[i, :]))

final_decision_conf_mat = np.zeros((predicts[predicted_csv[0]].shape[1], predicts[predicted_csv[0]].shape[1]))
final_acc = 0.0
for j, d in enumerate(gt_labels):
    d = int(d)
    final_decision_conf_mat[final_decision[j], d] += 1
    if final_decision[j] == d:
        final_acc += 1.0

final_acc /= predicts[predicted_csv[0]].shape[0]

confusion_matrices /= class_counts
sum_acc = np.sum(acc_mat, axis=1) / 9
max_acc = np.max(acc_mat, axis=1)
model_acc = np.sum(acc_mat, axis=0) / 226
model_picker = np.max(model_picker, axis=1)
model_picker_acc = np.sum(model_picker) / predicts[predicted_csv[0]].shape[0]

for i, cm in enumerate(confusion_matrices):
    plt.imsave("confusion_matrix_{}.png".format(predicted_csv[i]), cm)
    np.save("confusion_matrix_{}.npy".format(predicted_csv[i]), cm)

plt.imsave("confusion_matrix_final_decision.png", final_decision_conf_mat)
np.save("confusion_matrix_final_decision.png.npy", final_decision_conf_mat)
