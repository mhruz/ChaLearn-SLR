import sys
import pandas as pd
import numpy as np
import scipy.optimize
import cma


def _fun(x):

    correct = 0
    total = 0

    weights = x / float(sum(x))  # tyto vahy chceme optimalizovat vzhledem k final accuracy

    for i, gt in enumerate(ground_truth):
        gt_label = gt.split(';')[-1][:-1]

        ensemble = 0
        for w, pred in zip(weights, predicts):
            ensemble += w * np.array(pred.iloc[i])
        prediction = ensemble.argmax(0)

        if int(prediction) == int(gt_label):
            correct += 1
        '''else:
            preds = final_res.argsort(0)[-2:]
            preds = np.flip(preds)
            if int(preds[1]) == int(gt_label):
                vals = final_res[preds]
                print(i, gt_label, end='')
                for v,p in zip(vals, preds):
                    print(' %d:%.2f' %(p, v), end='')
                print('')'''
        #   print("id:%s c:%s k:%s m:%s y:%s e:%s g:%s" %(i, label0, label1, label2, label3, prediction, gt_label))

        total += 1

    acc = - 100. * correct / total
    print(weights, 'Final accuracy: %.4f' % (acc))

    return (acc)


if __name__ == '__main__':

    predicted_csv = ['crop.csv', 'crop_new.csv', 'keyframe.csv', 'keyframe_new.csv', 'mask.csv', 'mask_new.csv',
                     'keyframe_mask.csv', 'keyframe_mask_new.csv', 'openpose_41b.csv', 'vle_4.csv', 'vle_3.csv']
    print(predicted_csv)
    predicts = []
    data_dir = './data/ensemble_fine/'
    for pcsv in predicted_csv:
        _csv = pd.read_csv(data_dir + pcsv)
        predicts.append(_csv)

    with open(data_dir+'test.txt') as txt_file:
        ground_truth = txt_file.readlines()
    starting_point = np.random.rand(len(predicted_csv),1)
    cma.fmin2(_fun, starting_point, 1.)
