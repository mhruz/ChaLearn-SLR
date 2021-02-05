import argparse
import csv
import os

import cv2
import h5py
import numpy as np
from skimage import transform as tf


def compute_hand_pose_distance(hp_source, hp_target, metric_one=1):
    """
    Computes the distance of hand shapes. The metric is in the 'hp_hp_target' space.

    :param hp_source: 2D locations of joints of the source hand
    :param hp_target: 2D locations of joints of the target hand
    :param metric_one: What distance should be returned as 1 (e.g. length of shoulder)
    :return:
    """
    tform = tf.estimate_transform('similarity', hp_source, hp_target)
    hp_source_homo = np.hstack((hp_source, np.ones((hp_source.shape[0], 1))))
    reprojected_hp1 = np.matmul(tform.params[:2, :], hp_source_homo.T)

    flat_hp_source = reprojected_hp1.T.flatten()
    flat_hp_target = hp_target.flatten()

    return np.linalg.norm(flat_hp_source - flat_hp_target) / metric_one


if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Clusters hands by the Open Pose similarity.')
    parser.add_argument('open_pose_h5', type=str, help='path to H5 with detected joint locations')
    parser.add_argument('video_to_class_csv', type=str, help='csv with video labels')
    parser.add_argument('--hand_crops', type=str, help='optional h5 with cropped images of hands')
    parser.add_argument('--out_path', type=str, help='output path for cluster images', default="sign_clusters")
    parser.add_argument('--visualize', type=bool, help='whether to visualize')
    parser.add_argument('--threshold', type=float, help='threshold of hand pose estimation reliability', default=0.7)
    parser.add_argument('--acceptance', type=float, help='acceptance rate of hand-shapes to be the same', default=0.1)
    parser.add_argument('out_h5', type=str, help='output h5 dataset')
    args = parser.parse_args()

    f_hand_crops = None
    if args.hand_crops is not None:
        f_hand_crops = h5py.File(args.hand_crops, "r")

    f_video_to_class = csv.reader(open(args.video_to_class_csv, "r"))
    sign_to_samples = {}
    samples_to_signs = {}
    for row in f_video_to_class:
        sample_filename = "{}_color".format(row[0])
        if row[1] not in sign_to_samples:
            sign_to_samples[row[1]] = []

        sign_to_samples[row[1]].append(sample_filename)
        samples_to_signs[sample_filename] = row[1]

    joints_h5 = h5py.File(args.open_pose_h5, "r")

    sign_hand_clusters = {}
    seeders = {}

    for video_fn in joints_h5:
        sign_class = samples_to_signs[video_fn]
        if sign_class not in sign_hand_clusters:
            sign_hand_clusters[sign_class] = {"sample": [], "frame": []}

        joints = joints_h5[video_fn][1]
        joints = np.reshape(joints, (-1, 3))
        reference_shoulder_length = 0.5 * np.linalg.norm(joints[1, :2] - joints[2, :2]).item()
        starting_location = joints[7, :2]

        if sign_class not in seeders:
            seeders[sign_class] = []

        for frame, _ in enumerate(joints_h5[video_fn]):
            if frame < 10:
                continue

            # find a reliable hand-shape in this video
            joints = joints_h5[video_fn][frame]
            joints = np.reshape(joints, (-1, 3))
            mlh = np.mean(joints[8:29, 2])

            if mlh >= args.threshold:
                reference_hand = joints[8:29, :2]
                # skip hands that are close to starting location
                if abs(joints[7][0] - starting_location[0]) < reference_shoulder_length or abs(
                        joints[7][1] - starting_location[1]) < reference_shoulder_length:
                    continue

                # is this hand-shape already in sign clusters?
                if (video_fn, frame) in zip(sign_hand_clusters[sign_class]["sample"],
                                            sign_hand_clusters[sign_class]["frame"]):
                    continue
                else:
                    sign_hand_clusters[sign_class]["sample"].append(video_fn)
                    sign_hand_clusters[sign_class]["frame"].append(frame)
                    seeders[sign_class].append("{}{}".format(video_fn, frame))

                # search for the same hand-shapes in samples of the same sign
                for sample in sign_to_samples[sign_class]:
                    # ignore measuring sample with its-self
                    if sample == video_fn:
                        continue

                    target_joints = joints_h5[sample][0]
                    target_joints = np.reshape(joints, (-1, 3))
                    target_shoulder_length = 0.5 * np.linalg.norm(target_joints[1, :2] - target_joints[2, :2]).item()
                    target_starting_location = target_joints[7, :2]

                    target_dists = 1000 * np.ones(len(joints_h5[sample]))
                    for target_frame, _ in enumerate(joints_h5[sample]):
                        # is this hand-shape already in sign clusters?
                        if (sample, target_frame) in zip(sign_hand_clusters[sign_class]["sample"],
                                                         sign_hand_clusters[sign_class]["frame"]):
                            continue

                        target_joints = joints_h5[sample][target_frame]
                        target_joints = np.reshape(target_joints, (-1, 3))
                        mlh = np.mean(target_joints[8:29, 2])
                        if mlh < args.threshold:
                            continue

                        # skip hands that are close to starting location
                        if abs(target_joints[7][0] - target_starting_location[0]) < target_shoulder_length or \
                                abs(target_joints[7][1] - target_starting_location[1]) < target_shoulder_length:
                            continue

                        target_hand = target_joints[8:29, :2]
                        target_dist = compute_hand_pose_distance(target_hand, reference_hand, reference_shoulder_length)

                        target_dists[target_frame] = target_dist

                    if len(target_dists) == 0:
                        continue

                    min_dist = np.min(target_dists)
                    if min_dist > 0.5:
                        continue

                    dist_threshold = min_dist + min_dist * args.acceptance

                    same_hand_shapes = np.argwhere(target_dists <= dist_threshold)

                    # add new found hand-shapes to the cluster
                    for hand_shape_index in same_hand_shapes:
                        sign_hand_clusters[sign_class]["sample"].append(sample)
                        sign_hand_clusters[sign_class]["frame"].append(hand_shape_index.item())

        if args.hand_crops is not None:
            print(video_fn)
            print(sign_class)
            im_ref = None
            for i, (sample, frame) in enumerate(
                    zip(sign_hand_clusters[sign_class]["sample"], sign_hand_clusters[sign_class]["frame"])):

                img_idx = np.where(f_hand_crops["{}.mp4".format(sample)]["left_hand"]["frames"][:] == frame)

                img = f_hand_crops["{}.mp4".format(sample)]["left_hand"]["images"][img_idx][0]

                if args.visualize is None:
                    os.makedirs(os.path.join(args.out_path, sign_class), exist_ok=True)
                    cv2.imwrite(os.path.join(args.out_path, sign_class, "{}{}.jpg".format(sample, frame)), img)

                if "{}{}".format(sample, frame) in seeders[samples_to_signs[sample]]:
                    print("{}{}".format(sample, frame))
                    im_ref = img

                if args.visualize is not None:
                    cv2.namedWindow("{}{}".format(sample, frame), 2)
                    cv2.moveWindow("{}{}".format(sample, frame), i % 12 % 4 * 350, i % 12 // 3 * 350)
                    cv2.imshow("{}{}".format(sample, frame), img)

                    if (i+1) % 12 == 0:
                        if im_ref is not None:
                            cv2.namedWindow("ref", 2)
                            cv2.moveWindow("ref", 4 * 350, 0)
                            cv2.imshow("ref", im_ref)
                        cv2.waitKey()
                        cv2.destroyAllWindows()

            if args.visualize is not None:
                if im_ref is not None:
                    cv2.namedWindow("ref", 2)
                    cv2.moveWindow("ref", 4 * 350, 0)
                    cv2.imshow("ref", im_ref)
                cv2.waitKey()
                cv2.destroyAllWindows()
