import pandas as pd
import numpy as np
import cma

# dodat val csv!!!!!!
data_dir = './'
predicted_csv = ['crop.csv', 'crop_new.csv', 'keyframe.csv', 'keyframe_new.csv', 'mask.csv', 'mask_new.csv',
                 'keyframe_mask.csv', 'keyframe_mask_new.csv', 'openpose_41b.csv', 'vle_4.csv', 'vle_3.csv', 
                 'vle_12.csv', '1.csv', '2.csv', '3.csv', '4.csv', '5.csv']
print(predicted_csv)
predicts = []
for pcsv in predicted_csv:
    _csv = pd.read_csv(data_dir + pcsv)
    predicts.append(_csv)

with open('test.txt') as txt_file:
    ground_truth = txt_file.readlines()


def _fun(x):
    correct = 0
    total = 0

    weights = x / float(sum(x))    # tyto vahy chceme optimalizovat vzhledem k final accuracy

    for i, gt in enumerate(ground_truth):
        gt_label = gt.split(';')[-1][:-1]

        ensemble = 0
        for w,pred in zip(weights, predicts):
            ensemble += w * np.array(pred.iloc[i])
        prediction = ensemble.argmax(0)

        if int(prediction) == int(gt_label):
            correct += 1

        total += 1
    
    acc = - 100. * correct / total
    print(weights, 'Final accuracy: %.4f' %(acc))

    return(acc)

if __name__ == '__main__':
    _fun(np.array(17 *[.1]))
    _w, es = cma.fmin2(_fun, np.array(17 * [.1]), 1.)
    w = _w / sum(_w)

    crop = pd.read_csv(data_dir + 'crop/chalearn-rgb-i3d-resnet-50-ts-f16/softmax_values.csv')
    crop_new = pd.read_csv(data_dir + 'crop_new/chalearn-rgb-i3d-resnet-50-ts-f16/softmax_values.csv')
    keyframe = pd.read_csv(data_dir + 'keyframe/chalearn-rgb-i3d-resnet-50-ts-f16/softmax_values.csv')
    keyframe_new = pd.read_csv(data_dir + 'keyframe_new./chalearn-rgb-i3d-resnet-50-ts-f16/softmax_values.csv')
    mask = pd.read_csv(data_dir + 'mask/chalearn-rgb-i3d-resnet-50-ts-f16/softmax_values.csv')
    mask_new = pd.read_csv(data_dir + 'mask_new/chalearn-rgb-i3d-resnet-50-ts-f16/softmax_values.csv')
    keyframe_mask = pd.read_csv(data_dir + 'keyframe_mask/chalearn-rgb-i3d-resnet-50-ts-f16/softmax_values.csv')
    keyframe_mask_new = pd.read_csv(data_dir + 'keyframe_mask_new/chalearn-rgb-i3d-resnet-50-ts-f16/softmax_values.csv')
    openpose = pd.read_csv(data_dir + 'openpose_41b.csv')
    vle = pd.read_csv(data_dir + 'vle_4.csv')
    vle2 = pd.read_csv(data_dir + 'vle_3.csv')
    vle12 = pd.read_csv(data_dir + 'vle_12.csv')
    one = pd.read_csv(data_dir + '1/chalearn-rgb-i3d-resnet-50-ts-f16/softmax_values.csv')
    two = pd.read_csv(data_dir + '2/chalearn-rgb-i3d-resnet-50-ts-f16/softmax_values.csv')
    three = pd.read_csv(data_dir + '3/chalearn-rgb-i3d-resnet-50-ts-f16/softmax_values.csv')
    four = pd.read_csv(data_dir + '4/chalearn-rgb-i3d-resnet-50-ts-f16/softmax_values.csv')
    five = pd.read_csv(data_dir + '5/chalearn-rgb-i3d-resnet-50-ts-f16/softmax_values.csv')

    results = pd.read_csv(data_dir + 'predictions.csv', sep=',', header=None)

    for i in range(0, len(results)):
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

        final_res = (
                    w[0] * res0 + w[1] * res1 + w[2] * res2 + w[3] * res3 + w[4] * res4 + w[5] * res5 + w[6] * res6 + w[
                7] * res7
                    + w[8] * res8 + w[9] * res9 + w[10] * res10 + w[11] * res11 + w[12] * res12 + w[13] * res13 + w[
                        14] * res14
                    + w[15] * res15 + w[16] * res16)
        prediction = final_res.argmax(0)
        results[1][i] = prediction

    results.to_csv(data_dir + "predictions.csv", index=False, header=None)


    #['crop.csv', 'crop_new.csv', 'keyframe.csv', 'keyframe_new.csv', 'mask.csv', 
    # 'mask_new.csv', 'keyframe_mask.csv', 'keyframe_mask_new.csv', 'openpose_41b.csv', 
    # 'vle_4.csv', 'vle_3.csv', 'vle_12.csv', '1.csv', '2.csv', '3.csv', '4.csv', '5.csv']
    #[0.04762968  0.13529561  0.05915348  0.04292918 -0.02789492  
    #  0.07635797 0.12705846  0.03582622  0.17253467
    #  0.08502941  0.11312029  0.02414777 0.053221 0.12860186 0.02714533 -0.05312429 -0.04703171] Final accuracy: -95.5636