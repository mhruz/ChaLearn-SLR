import argparse
import h5py
import os
import numpy as np
from skimage import transform as tf
import cv2
import pickle


def compute_hand_pose_distance_weighted(hp_source, hp_target, metric_one=1):
    """
    Computes the distance of hand shapes. The metric is in the 'hp_hp_target' space.
    The optimal transformation is found from wrist and MCPs.
    The distance is computed from fingers.

    :param hp_source: 2D locations of joints of the source hand
    :param hp_target: 2D locations of joints of the target hand
    :param metric_one: What distance should be returned as 1 (e.g. length of shoulder)
    :return:
    """

    palm_source = hp_source[[0, 1, 5, 9, 13, 17], :]
    palm_target = hp_target[[0, 1, 5, 9, 13, 17], :]
    tform = tf.estimate_transform('similarity', palm_source, palm_target)
    hp_source_homo = np.hstack((hp_source, np.ones((hp_source.shape[0], 1))))
    reprojected_hp_source = np.matmul(tform.params[:2, :], hp_source_homo.T)

    weights = np.ones_like(hp_target)
    # fingertips
    weights[4:21:4, :] = 8.0
    # dips
    weights[3:21:4, :] = 4.0
    # pips
    weights[6:21:4, :] = 1.5

    weights = weights.flatten()
    # normalize weights
    weights /= np.sum(weights) / weights.shape[0]

    flat_hp_source = reprojected_hp_source.T.flatten()
    flat_hp_target = hp_target.flatten()
    diff = flat_hp_source - flat_hp_target

    diff *= weights

    return np.linalg.norm(diff) / metric_one


if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Prepare reference data from file tree.')
    parser.add_argument('data_root', type=str, help='root of reference hand images')
    parser.add_argument('joints_h5', type=str, help='path to dataset with hand joints')
    parser.add_argument('sign_clusters_h5', type=str, help='path to dataset hand poses by sign')
    parser.add_argument('--max_dist', type=float,
                        help='maximal distance between hand poses considered to be the same hand pose', default=0.6)
    parser.add_argument('--min_conf', type=float,
                        help='minimal accepted confidence of joint estimation', default=0.7)
    parser.add_argument('--max_samples', type=int,
                        help='maximum number of reference images (if there is more, the reference class is ignored)',
                        default=1000)
    parser.add_argument('--hand_crops', type=str, help='optional h5 with cropped images of hands')
    parser.add_argument('output', type=str, help='path to output with clustered hand poses')
    args = parser.parse_args()

    f_joints = h5py.File(args.joints_h5, "r")
    joints_data = {}
    for video_fn in f_joints:
        joints_data[video_fn] = f_joints[video_fn][:]

    f_signs = h5py.File(args.sign_clusters_h5, "r")
    sign_clusters_data = {}
    for sign_class in f_signs:
        sign_clusters_data[sign_class] = {}
        sign_clusters_data[sign_class]["samples"] = f_signs[sign_class]["samples"][:]
        sign_clusters_data[sign_class]["frames"] = f_signs[sign_class]["frames"][:]
        sign_clusters_data[sign_class]["seeders"] = {}
        sign_clusters_data[sign_class]["seeders"]["samples"] = f_signs[sign_class]["seeders"]["samples"][:]
        sign_clusters_data[sign_class]["seeders"]["frames"] = f_signs[sign_class]["seeders"]["frames"][:]

    if args.hand_crops is not None:
        f_hand_crops = h5py.File(args.hand_crops, "r")

    hand_pose_classes = os.listdir(args.data_root)

    hand_pose_classes = [x for x in hand_pose_classes if
                         not x.startswith("_") and os.path.isdir(os.path.join(args.data_root, x))]

    # sample_ref = "signer0_sample56_color"
    # frame_ref = 35
    # joints_ref = f_joints[sample_ref][frame_ref]
    # joints_ref = np.reshape(joints_ref, (-1, 3))
    #
    # hand_ref = joints_ref[8:29][:, :2]
    # shoulder_ref = np.linalg.norm(joints_ref[1, :2] - joints_ref[2, :2]).item()
    # hand_ref_conf = np.mean(joints_ref[8:29][:, 2])
    #
    # sample = "signer3_sample13_color"
    # frame = 52
    #
    # joints = f_joints[sample][frame]
    # joints = np.reshape(joints, (-1, 3))
    #
    # hand = joints[8:29][:, :2]
    # hand_conf = np.mean(joints[8:29][:, 2])
    #
    # dist = compute_hand_pose_distance_weighted(hand, hand_ref, shoulder_ref)

    all_ref_samples = []
    candidates = {}
    hand_pose_classes_valid = []
    for hand_class in hand_pose_classes:
        images_class = os.listdir(os.path.join(args.data_root, hand_class))
        images_class = [x for x in images_class if x.endswith(".jpg")]
        if len(images_class) < args.max_samples:
            candidates[hand_class] = []
            hand_pose_classes_valid.append(hand_class)
            for image_fn in images_class:
                all_ref_samples.append(image_fn)

    hand_pose_classes = hand_pose_classes_valid
    reference_hands = {}

    for hand_class in hand_pose_classes:
        print("Processing hand class: {}".format(hand_class))

        if hand_class not in reference_hands:
            reference_hands[hand_class] = []

        images_class = os.listdir(os.path.join(args.data_root, hand_class))
        images_class = [x for x in images_class if x.endswith(".jpg")]
        for image_fn in images_class:
            sample_ref = image_fn[:image_fn.find("_color") + 6]
            frame_ref = int(image_fn[image_fn.find("_color") + 6:-4])

            joints_ref = joints_data[sample_ref][frame_ref]
            joints_ref = np.reshape(joints_ref, (-1, 3))

            hand_ref = joints_ref[8:29][:, :2]
            shoulder_ref = np.linalg.norm(joints_ref[1, :2] - joints_ref[2, :2]).item()
            hand_ref_conf = np.mean(joints_ref[8:29][:, 2])
            if hand_ref_conf < args.min_conf:
                continue

            new_hand = True
            for ref_hand in reference_hands[hand_class]:
                dist = compute_hand_pose_distance_weighted(ref_hand, hand_ref, shoulder_ref)
                if dist < args.max_distance / 2:
                    new_hand = False
                    break

            if new_hand:
                reference_hands[hand_class].append(hand_ref)
            else:
                continue

            # img_idx = np.where(f_hand_crops["{}".format(sample_ref)]["left_hand"]["frames"][:] == frame_ref)
            # img_ref = f_hand_crops["{}".format(sample_ref)]["left_hand"]["images"][img_idx][0]

            ignored_sign_classes = list(range(35))
            ignored_sign_classes.extend(list(range(150, 163)))
            ignored_sign_classes.extend(list(range(200, 216)))
            for sign_class in sign_clusters_data:
                if sign_class in ignored_sign_classes:
                    continue

                for frame, sample in zip(sign_clusters_data[sign_class]["frames"],
                                         sign_clusters_data[sign_class]["samples"]):

                    if isinstance(sample, bytes):
                        sample = sample.decode("utf-8")
                        
                    if sample == sample_ref and frame == frame_ref:
                        continue

                    if "{}{}.jpg".format(sample, frame) in all_ref_samples:
                        continue

                    joints = joints_data[sample][frame]
                    joints = np.reshape(joints, (-1, 3))

                    hand = joints[8:29][:, :2]
                    hand_conf = np.mean(joints[8:29][:, 2])

                    if hand_conf < 0.55:
                        continue

                    dist = compute_hand_pose_distance_weighted(hand, hand_ref, shoulder_ref)

                    if dist < args.max_dist:
                        # look for the candidate in another class
                        sample_string = "{}{}.jpg".format(sample, frame)
                        found_same = False
                        end_loop = False
                        for candidate_class in candidates:
                            for candidate in candidates[candidate_class]:
                                if sample_string == candidate["sample"]:
                                    found_same = True
                                    if candidate["distance"] > dist:
                                        if hand_class != candidate_class:
                                            candidates[hand_class].append({"sample": sample_string, "distance": dist})
                                        else:
                                            candidate["dist"] = dist

                                    end_loop = True
                                    break

                            if end_loop:
                                break

                        if not found_same:
                            candidates[hand_class].append({"sample": sample_string, "distance": dist})

                        # img_idx = np.where(f_hand_crops["{}".format(sample)]["left_hand"]["frames"][:] == frame)
                        # img = f_hand_crops["{}".format(sample)]["left_hand"]["images"][img_idx][0]
                        #
                        # print(dist)
                        # print("ref_conf: {}, target_conf: {}".format(hand_ref_conf, hand_conf))
                        #
                        # cv2.namedWindow("ref", 2)
                        # cv2.namedWindow("close", 2)
                        # cv2.imshow("ref", img_ref)
                        # cv2.imshow("close", img)
                        # cv2.waitKey()
                        # cv2.destroyAllWindows()

    pickle.dump(candidates, open(args.output, "wb"))
