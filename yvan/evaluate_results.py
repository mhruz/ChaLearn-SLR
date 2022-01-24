import pandas as pd
import numpy as np

data_dir = './data/ensemble_fine/'

# crop = pd.read_csv(data_dir + 'crop_new.csv')
# mask = pd.read_csv(data_dir + 'mask.csv')
# keyframe_mask = pd.read_csv(data_dir + 'keyframe_mask.csv')
# openpose2 = pd.read_csv(data_dir + 'openpose_41b.csv')
# vle = pd.read_csv(data_dir + 'vle_4.csv')
# vle2 = pd.read_csv(data_dir + 'vle_3.csv')

crop = pd.read_csv(data_dir + 'crop.csv')
crop_new = pd.read_csv(data_dir + 'crop_new.csv')
keyframe = pd.read_csv(data_dir + 'keyframe.csv')
keyframe_new = pd.read_csv(data_dir + 'keyframe_new.csv')
mask = pd.read_csv(data_dir + 'mask.csv')
mask_new = pd.read_csv(data_dir + 'mask_new.csv')
keyframe_mask = pd.read_csv(data_dir + 'keyframe_mask.csv')
keyframe_mask_new = pd.read_csv(data_dir + 'keyframe_mask_new.csv')
openpose = pd.read_csv(data_dir + 'openpose_41b.csv')
vle = pd.read_csv(data_dir + 'vle_4.csv')
vle2 = pd.read_csv(data_dir + 'vle_3.csv')
vle12 = pd.read_csv(data_dir + 'vle_12.csv')
one = pd.read_csv(data_dir + '1.csv')
two = pd.read_csv(data_dir + '2.csv')
three = pd.read_csv(data_dir + '3.csv')
four = pd.read_csv(data_dir + '4.csv')
five = pd.read_csv(data_dir + '5.csv')

results = pd.read_csv(data_dir+'/predictions.csv', sep=',', header=None)

# w = [0.26041384, 0.14270178, 0.1792352,  0.14685267, 0.0472671,  0.22352942]
# w = [1,1,1,1,1,1]
# w = [0.09185617, 0.16844044, 0.04648995, 0.06553169, -0.02352196, 0.07401773, 0.10360531, 0.02910761, 0.14868509, 0.13212617, 0.1636618]
w = [0.04762968,  0.13529561,  0.05915348,  0.04292918, -0.02789492,  0.07635797,
  0.12705846,  0.03582622,  0.17253467,  0.08502941,  0.11312029,  0.02414777,
  0.053221,    0.12860186,  0.02714533, -0.05312429, -0.04703171]
w = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]


for i in range(0,len(results)):
    res0 = np.array(crop.iloc[i])
    res1 = np.array(crop_new.iloc[i])
    res2 = np.array(keyframe.iloc[i])
    res3 = np.array(keyframe_new.iloc[i])
    res4 = np.array(mask.iloc[i])
    res5 = np.array(mask_new.iloc[i])
    res6 = np.array(keyframe_mask.iloc[i])
    res7 = np.array(keyframe_mask_new.iloc[i])
    res8 = np.array(openpose.iloc[i])
    res9 = np.array(vle.iloc[i])
    res10 = np.array(vle2.iloc[i])
    res11 = np.array(vle12.iloc[i])
    res12 = np.array(one.iloc[i])
    res13 = np.array(two.iloc[i])
    res14 = np.array(three.iloc[i])
    res15 = np.array(four.iloc[i])
    res16 = np.array(five.iloc[i])

    # final_res = (w[0] * res0 + w[1] * res1 + w[2] * res2 + w[3] * res3)
    final_res = (w[0] * res0 + w[1]*res1 + w[2]*res2 + w[3]*res3 + w[4] * res4 + w[5]*res5 + w[6]*res6 + w[7]*res7 + w[8]*res8 + w[9]*res9 + w[10]*res10 + w[11]*res11 + w[12]*res12 + w[13]*res13 + w[14]*res14 + w[15]*res15 + w[16]*res16)
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
