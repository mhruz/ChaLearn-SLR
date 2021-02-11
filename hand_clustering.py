import h5py
import argparse
import numpy as np
from skimage import transform as tf
import time
import pickle
import concurrent.futures
import cv2
import copy


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
    weights[4:21:4, :] = 3.0
    # dips
    weights[3:21:4, :] = 2.0
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


def compute_distances(index, sample_i, samples, joints_data, shoulder_length):
    distance_vector = np.ones(len(samples["samples"]))
    for j, sample_j in enumerate(samples["samples"]):
        if j <= index:
            continue

        joints_i = joints_data[sample_i.decode("utf-8")][samples["frames"][index]]
        joints_i = np.reshape(joints_i, (-1, 3))[8:29, :2]
        joints_j = joints_data[sample_j.decode("utf-8")][samples["frames"][j]]
        joints_j = np.reshape(joints_j, (-1, 3))[8:29, :2]
        dist = compute_hand_pose_distance_weighted(joints_j, joints_i, shoulder_length)
        distance_vector[j] = dist

    return distance_vector, index


def fast_triu_indices(dim, k=0):
    tmp_range = np.arange(dim - k)
    rows = np.repeat(tmp_range, (tmp_range + 1)[::-1])

    cols = np.ones(rows.shape[0], dtype=np.int)
    inds = np.cumsum(tmp_range[1:][::-1] + 1)

    np.put(cols, inds, np.arange(dim * -1 + 2 + k, 1))
    cols[0] = k
    np.cumsum(cols, out=cols)
    return rows, cols


def agglomerative_clustering(hand_samples, max_distance, distances, hand_crops, joints, visualize=False):
    clustering = {'hand_samples': hand_samples, 'clusters': []}
    clusters = []
    cluster_state = [[x] for x in range(len(hand_samples["samples"]))]
    orig_distances = np.copy(distances)

    for i, hs in enumerate(hand_samples["samples"]):
        clusters.append([{'idx': i, 'samples': hs}])

    s = time.time()

    while len(clusters) > 1:
        print('Step {}/{}'.format(len(hand_samples["samples"]) - len(clusters) + 1, len(hand_samples["samples"]) - 1))
        # find the minimum distance
        mask = np.ones_like(distances, dtype=np.bool)
        mask = np.tril(mask, 0)
        masked_distances = np.ma.masked_array(distances, mask)

        min_dist = masked_distances.min()

        print(min_dist)

        if min_dist > max_distance and visualize:
            if np.isinf(min_dist):
                break

            mins = np.where(orig_distances == min_dist)
            sample_a = hand_samples["samples"][mins[0][0]].decode("utf-8")
            sample_b = hand_samples["samples"][mins[1][0]].decode("utf-8")
            frame_a = hand_samples["frames"][mins[0][0]]
            frame_b = hand_samples["frames"][mins[1][0]]

            img_idx = np.where(hand_crops["{}".format(sample_a)]["left_hand"]["frames"][:] == frame_a)
            hand_img = hand_crops["{}".format(sample_a)]["left_hand"]["images"][img_idx][0]
            cv2.namedWindow("{}_{}".format(sample_a, frame_a), 2)
            cv2.moveWindow("{}_{}".format(sample_a, frame_a), 350, 350)
            cv2.imshow("{}_{}".format(sample_a, frame_a), hand_img)

            img_idx = np.where(hand_crops["{}".format(sample_b)]["left_hand"]["frames"][:] == frame_b)
            hand_img = hand_crops["{}".format(sample_b)]["left_hand"]["images"][img_idx][0]
            cv2.namedWindow("{}_{}".format(sample_b, frame_b), 2)
            cv2.moveWindow("{}_{}".format(sample_b, frame_b), 700, 350)
            cv2.imshow("{}_{}".format(sample_b, frame_b), hand_img)

            joints_sample_a = joints[sample_a][:].reshape(-1, 3)
            joints_sample_a = joints_sample_a[8:29, 2]

            print("Sample_a conf: {}".format(np.mean(joints_sample_a)))

            joints_sample_b = joints[sample_b][:].reshape(-1, 3)
            joints_sample_b = joints_sample_b[8:29, 2]

            print("Sample_b conf: {}".format(np.mean(joints_sample_b)))

        if min_dist > max_distance:
            print("I refuse to merge these images! distance={}".format(min_dist))
            cv2.waitKey()
            cv2.destroyAllWindows()
            break
        else:
            # cv2.waitKey()
            cv2.destroyAllWindows()

        mins = np.ma.where(masked_distances == masked_distances.min())
        # min_i is always the larger cluster
        if len(clusters[mins[0][0]]) >= len(clusters[mins[1][0]]):
            min_i = mins[0][0]
            min_j = mins[1][0]
        else:
            min_i = mins[1][0]
            min_j = mins[0][0]

        # check if the joining of the cluster would not break integrity
        integrity_threshold = 1.5 * max_distance
        integrity_check = True
        integrity_check_count = 0
        for idx_j in clusters[min_j]:
            for idx_i in clusters[min_i]:
                a = min(idx_i["idx"], idx_j["idx"])
                b = max(idx_i["idx"], idx_j["idx"])
                if orig_distances[a, b] > integrity_threshold:
                    integrity_check_count += 1
                    if integrity_check_count / (len(clusters[min_i] * len(clusters[min_j]))) > 0.5:
                        integrity_check = False
                        break

            # adding these clusters would break integrity (max distance in one cluster)
            if not integrity_check:
                break

        print("Integrity check when merging {} ({}) and {} ({}): {}"
              .format(min_i, len(clusters[min_i]), min_j,
                      len(clusters[min_j]),
                      integrity_check_count /
                      (len(clusters[min_i] * len(clusters[min_j])))))

        if not integrity_check:
            distances[min_i, min_j] = np.inf
            distances[min_j, min_i] = np.inf
            # skip to next sample
            continue

        # join clusters min_i and min_j
        clusters[min_i].extend(clusters[min_j])
        cluster_state[min_i].extend(cluster_state[min_j])

        # recompute the distances
        for i in range(len(distances)):
            if i < min_i:
                if i < min_j:
                    distances[i, min_i] = np.minimum(distances[i, min_i], distances[i, min_j])
                elif i > min_j:
                    distances[i, min_i] = np.minimum(distances[i, min_i], distances[min_j, i])
            elif i > min_i:
                if i < min_j:
                    distances[min_i, i] = np.minimum(distances[min_i, i], distances[i, min_j])
                elif i > min_j:
                    distances[min_i, i] = np.minimum(distances[min_i, i], distances[min_j, i])

        distances = np.delete(distances, min_j, 0)
        distances = np.delete(distances, min_j, 1)

        # remove the appropriate data
        del clusters[min_j]
        del cluster_state[min_j]

        clustering['clusters'].append({'clusters': copy.deepcopy(cluster_state)})

    e = time.time()
    print('Clustering computation: {} s'.format(e - s))

    # get rid of the singleton clusters
    clusters = clustering["clusters"][-1]["clusters"]
    out_clusters = []
    for c in clusters:
        if len(c) > 1:
            out_clusters.append(c)

    return out_clusters


if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Clusters hands by the Open Pose similarity.')
    parser.add_argument('open_pose_h5', type=str, help='path to H5 with detected joint locations')
    parser.add_argument('sign_clusters_h5', type=str, help='h5 with hand clusters by signs')
    parser.add_argument('--hand_crops', type=str, help='optional h5 with cropped images of hands')
    parser.add_argument('--out_path', type=str, help='output path for cluster images', default="hand_clusters")
    parser.add_argument('--visualize', type=bool, help='whether to visualize')
    parser.add_argument('--threshold', type=float, help='threshold of hand pose estimation reliability', default=0.5)
    parser.add_argument('--acceptance', type=float, help='acceptance rate of hand-shapes to be the same', default=0.0)
    parser.add_argument('--max_dist', type=float, help='distance threshold to accept as the same shape', default=1.0)
    parser.add_argument('--joints_to_mem', type=bool, help='read joints data to memory')
    parser.add_argument('--sign_clusters_to_mem', type=bool, help='read sign cluster data to memory')
    parser.add_argument('--sign_clusters_pickle', type=str, help='if provided, the clustering will be loaded from this')
    parser.add_argument('out_h5', type=str, help='output h5 dataset')
    args = parser.parse_args()

    if args.sign_clusters_pickle is not None:
        sign_hand_clusters = pickle.load(open(args.sign_clusters_pickle, "rb"))
    else:
        f_sign_clusters = h5py.File(args.sign_clusters_h5, "r")
        f_joints = h5py.File(args.open_pose_h5)

        # read joints into memory
        if args.joints_to_mem is not None:
            joints_data = {}
            for video_fn in f_joints:
                joints_data[video_fn] = f_joints[video_fn][:]
        else:
            joints_data = f_joints

        # read sign clusters into memory
        if args.sign_clusters_to_mem is not None:
            sign_clusters_data = {}
            for sign_class in f_sign_clusters:
                sign_clusters_data[sign_class] = {}
                sign_clusters_data[sign_class]["samples"] = f_sign_clusters[sign_class]["samples"][:]
                sign_clusters_data[sign_class]["frames"] = f_sign_clusters[sign_class]["frames"][:]
                sign_clusters_data[sign_class]["seeders"] = {}
                sign_clusters_data[sign_class]["seeders"]["samples"] = f_sign_clusters[sign_class]["seeders"][
                                                                           "samples"][:]
                sign_clusters_data[sign_class]["seeders"]["frames"] = f_sign_clusters[sign_class]["seeders"]["frames"][
                                                                      :]
        else:
            sign_clusters_data = f_sign_clusters

        f_hand_crops = None
        if args.hand_crops is not None:
            f_hand_crops = h5py.File(args.hand_crops, "r")

        sign_hand_clusters = {}

        # compute distances of hand-shapes in one sign to create sub-clusters of sign hand-shapes
        # since we have a per-pair custom distance function, we need to do it the old-fashioned way
        for idx_sign, sign_class in enumerate(sign_clusters_data):

            # if idx_sign < 2:
            #     continue

            print("Processing sign {}\n".format(sign_class))
            if sign_class not in sign_hand_clusters:
                sign_hand_clusters[sign_class] = []

            sample_strings = []

            try:
                distance_matrix = np.load("distances_{}.npy".format(sign_class))
                for i, sample_i in enumerate(sign_clusters_data[sign_class]["samples"]):
                    sample_strings.append(
                        "{}_{}".format(sample_i.decode("utf-8"), sign_clusters_data[sign_class]["frames"][i]))

            except FileNotFoundError:
                number_of_samples = len(sign_clusters_data[sign_class]["samples"])
                distance_matrix = np.ones((number_of_samples, number_of_samples))
                for i, sample_i in enumerate(sign_clusters_data[sign_class]["samples"]):
                    sample_strings.append(
                        "{}_{}".format(sample_i.decode("utf-8"), sign_clusters_data[sign_class]["frames"][i]))

                future_to_args = []
                start = time.time()

                with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
                    for i, sample_i in enumerate(sign_clusters_data[sign_class]["samples"]):
                        joints = joints_data[sample_i.decode("utf-8")][0]
                        joints = np.reshape(joints, (-1, 3))
                        shoulder_length = np.linalg.norm(joints[1, :2] - joints[2, :2]).item()

                        future_to_args.append(
                            executor.submit(compute_distances, i, sample_i,
                                            sign_clusters_data[sign_class], joints_data, shoulder_length))

                for future in concurrent.futures.as_completed(future_to_args):
                    try:
                        dist_vec, index = future.result()
                        distance_matrix[index] = dist_vec
                    except Exception as exc:
                        print("Generated an exception: {}".format(exc))

                end = time.time()
                print("Distance computation for {} elements: {} s".format(number_of_samples, end - start))

                np.save("distances_{}.npy".format(sign_class), distance_matrix)

            sign_hand_clusters[sign_class] = agglomerative_clustering(sign_clusters_data[sign_class], args.max_dist,
                                                                      distance_matrix, f_hand_crops, f_joints,
                                                                      args.visualize)

            print("Number of clusters: {}".format(len(sign_hand_clusters[sign_class])))

            if args.visualize is not None:
                for cluster_idx, cluster in enumerate(sign_hand_clusters[sign_class]):
                    for i, sample_idx in enumerate(cluster):
                        sample_string = sample_strings[sample_idx]
                        parts = sample_string.split("_")
                        sample = "_".join(parts[:3])
                        frame = int(parts[3])

                        img_idx = np.where(f_hand_crops["{}".format(sample)]["left_hand"]["frames"][:] == frame)
                        hand_img = f_hand_crops["{}".format(sample)]["left_hand"]["images"][img_idx][0]

                        cv2.namedWindow("{}_{}".format(cluster_idx, sample_string), 2)
                        cv2.moveWindow("{}_{}".format(cluster_idx, sample_string), i % 24 % 6 * 350, i % 24 // 6 * 350)
                        cv2.imshow("{}_{}".format(cluster_idx, sample_string), hand_img)

                        if (i + 1) % 24 == 0:
                            cv2.waitKey()
                            cv2.destroyAllWindows()

                    cv2.waitKey()
                    cv2.destroyAllWindows()

                cv2.waitKey()
                cv2.destroyAllWindows()

        print("Done")

        pickle.dump(sign_hand_clusters, open("sign_hand_clusters_v03_04.p", "wb"))

    # # cluster the sub-clusters
    # super_clusters = []
    # for i, cluster in enumerate(sign_hand_clusters):
    #     # create the 1st super cluster
    #     if i == 0:
    #         super_clusters.append(cluster)
