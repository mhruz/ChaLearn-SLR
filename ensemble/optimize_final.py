import pandas as pd
import numpy as np
import cma

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


    #['crop.csv', 'crop_new.csv', 'keyframe.csv', 'keyframe_new.csv', 'mask.csv', 
    # 'mask_new.csv', 'keyframe_mask.csv', 'keyframe_mask_new.csv', 'openpose_41b.csv', 
    # 'vle_4.csv', 'vle_3.csv', 'vle_12.csv', '1.csv', '2.csv', '3.csv', '4.csv', '5.csv']
    #[0.04762968  0.13529561  0.05915348  0.04292918 -0.02789492  
    #  0.07635797 0.12705846  0.03582622  0.17253467
    #  0.08502941  0.11312029  0.02414777 0.053221 0.12860186 0.02714533 -0.05312429 -0.04703171] Final accuracy: -95.5636