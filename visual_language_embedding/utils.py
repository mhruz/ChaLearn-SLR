import numpy as np
import numpy.linalg
import h5py
from skimage import transform as tf
import cv2


def compute_hand_pose_distance(hp1, hp2, metric_one=1):
    tform = tf.estimate_transform('similarity', hp1, hp2)
    hp1_homo = np.hstack((hp1, np.ones((hp1.shape[0], 1))))
    reprojected_hp1 = np.matmul(tform.params[:2, :], hp1_homo.T)

    flat_hp1 = reprojected_hp1.T.flatten()
    flat_hp2 = hp2.flatten()

    return numpy.linalg.norm(flat_hp1 - flat_hp2) / metric_one


if __name__ == "__main__":

    f_joints = h5py.File(r'z:\korpusy_cv\AUTSL\train_json_keypoints-raw.h5', 'r')
    video_fn_ref = "signer7_sample15_color.mp4"
    video_fn_target = "signer28_sample312_color.mp4"

    joints = f_joints[video_fn_ref[:-4]][0]
    joints = np.reshape(joints, (-1, 3))
    shoulder_length = np.linalg.norm(joints[1, :2] - joints[2, :2]).item()

    joints_target = f_joints[video_fn_target[:-4]][0]
    joints_target = np.reshape(joints_target, (-1, 3))
    shoulder_length_target = np.linalg.norm(joints_target[1, :2] - joints_target[2, :2]).item()

    ref_idx = 26
    joints = f_joints[video_fn_ref[:-4]][ref_idx]
    joints = np.reshape(joints, (-1, 3))
    hp1 = joints[8:29, :2]

    f_hands = h5py.File(r"z:\korpusy_cv\AUTSL\train_hand_images.h5")
    print(f_hands[video_fn_ref]["left_hand"]["frames"][:])

    for i, j in enumerate(f_joints[video_fn_target[:-4]]):
        joints = j
        joints = np.reshape(joints, (-1, 3))[8:29]
        hp2 = joints[:, :2]

        dist = compute_hand_pose_distance(hp1, hp2, metric_one=shoulder_length)
        dist_euclid = np.linalg.norm(((hp1 - np.mean(hp1)) / shoulder_length).flatten() -
                                     ((hp2 - np.mean(hp2)) / shoulder_length_target).flatten())

        print(i, dist, i in f_hands[video_fn_target]["left_hand"]["frames"][:], np.mean(joints[:, 2]), dist_euclid)

    im1_idx = np.where(f_hands[video_fn_ref]["left_hand"]["frames"][:] == ref_idx)
    im1 = f_hands[video_fn_ref]["left_hand"]["images"][im1_idx][0]

    im2_idx = np.where(f_hands[video_fn_target]["left_hand"]["frames"][:] == 31)
    im2 = f_hands[video_fn_target]["left_hand"]["images"][im2_idx][0]

    cv2.namedWindow("im1", 0)
    cv2.imshow("im1", im1)
    cv2.namedWindow("im2", 0)
    cv2.imshow("im2", im2)

    cv2.waitKey()
    cv2.destroyAllWindows()
