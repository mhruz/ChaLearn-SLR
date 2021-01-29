import numpy as np
import numpy.linalg
import h5py
from skimage import transform as tf


def compute_hand_pose_distance(hp1, hp2):
    tform = tf.estimate_transform('similarity', hp1, hp2)
    hp1_homo = np.hstack((hp1, np.ones((hp1.shape[0], 1))))
    hp1_homo_T = hp1_homo.T
    reprojected_hp1 = np.matmul(tform.params[:2, :], hp1_homo_T)

    flat_hp1 = reprojected_hp1.T.flatten()
    flat_hp2 = hp2.flatten()

    return numpy.linalg.norm(flat_hp1 - flat_hp2)


if __name__ == "__main__":

    f_joints = h5py.File(r'w:\korpusy_cv\AUTSL\train_json_keypoints-raw.h5', 'r')
    video_fn = "signer9_sample96_color.mp4"
    joints = f_joints[video_fn[:-4]][0]

    hp1 = np.reshape(joints, (-1, 3))[29:49]
    hp1 = hp1[:, :2]

    for i in f_joints[video_fn[:-4]]:
        joints = i
        hp2 = np.reshape(joints, (-1, 3))[29:49]
        hp2 = hp2[:, :2]

        dist = compute_hand_pose_distance(hp1, hp2)

        print(dist)