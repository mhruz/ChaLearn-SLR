import pandas as pd
import numpy as np

data_dir = './data/ensemble_test_hard_v2/'

crop = pd.read_csv(data_dir + 'test_crop.csv_hard_soft_max_v2.csv')
crop_new = pd.read_csv(data_dir + 'test_crop_new.csv_hard_soft_max_v2.csv')
mask = pd.read_csv(data_dir + 'test_mask.csv_hard_soft_max_v2.csv')
keyframe_mask = pd.read_csv(data_dir + 'test_keyframe_mask.csv_hard_soft_max_v2.csv')
keyframe_new = pd.read_csv(data_dir + 'test_keyframe_new.csv_hard_soft_max_v2.csv')
mask_new = pd.read_csv(data_dir + 'test_mask_new.csv_hard_soft_max_v2.csv')
keyframe_mask_new = pd.read_csv(data_dir + 'test_keyframe_mask_new.csv_hard_soft_max_v2.csv')
openpose2 = pd.read_csv(data_dir + 'test_openpose_41b.csv_hard_soft_max_v2.csv')
vle = pd.read_csv(data_dir + 'test_vle_4.csv_hard_soft_max_v2.csv')
vle2 = pd.read_csv(data_dir + 'test_vle_3.csv_hard_soft_max_v2.csv')

results = pd.read_csv(data_dir+'/predictions.csv', sep=',', header=None)

# w = [0.26041384, 0.14270178, 0.1792352,  0.14685267, 0.0472671,  0.22352942]
# w = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
w = [ 0.07888936,  0.12174075,  0.00935273,  0.12080636,  0.13075182,  0.13058413,
 -0.02667294,  0.17764039,  0.16122622,  0.09568118]


for i in range(0,len(results)):
    res0 = np.array(crop.iloc[i])
    res1 = np.array(crop_new.iloc[i])
    res2 = np.array(mask.iloc[i])
    res3 = np.array(keyframe_mask.iloc[i])
    res4 = np.array(keyframe_new.iloc[i])
    res5 = np.array(mask_new.iloc[i])
    res6 = np.array(keyframe_mask_new.iloc[i])
    res7 = np.array(openpose2.iloc[i])
    res8 = np.array(vle.iloc[i])
    res9 = np.array(vle2.iloc[i])

    final_res = (w[0] * res0 + w[1] * res1 + w[2] * res2 + w[3] * res3 + w[4] * res4 + w[5]*res5 + w[6]*res6 + w[7]*res7 + w[8]*res8+w[9]*res9)
    prediction = final_res.argmax(0)
    results[1][i] = prediction

results.to_csv(data_dir+"predictions.csv",index=False, header=None)




with open(data_dir+'test.txt') as txt_file:
    ground_truth = txt_file.readlines()

results = pd.read_csv(data_dir+'predictions.csv',header=None, names=['video','label'])

correct = 0
total = 0

for i, gt in enumerate(ground_truth):
    res = results['label'][i]
    gt_label = gt.split(';')[-1][:-1]
    if int(res)==int(gt_label):
        correct+=1
    total+=1

print('Final accuracy: %.4f' %(100*correct / total))
