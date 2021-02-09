import h5py
import argparse
import numpy as np
from skimage import transform as tf
import time
import pickle
import concurrent.futures


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


def compute_distances(index, sample_i, samples, joints_data, shoulder_length):
    distance_vector = np.zeros(len(samples["samples"]))
    for j, sample_j in enumerate(samples["samples"]):
        if j <= index:
            continue

        joints_i = joints_data[sample_i.decode("utf-8")][samples["frames"][index]]
        joints_i = np.reshape(joints_i, (-1, 3))[8:29, :2]
        joints_j = joints_data[sample_j.decode("utf-8")][samples["frames"][j]]
        joints_j = np.reshape(joints_j, (-1, 3))[8:29, :2]
        dist = compute_hand_pose_distance(joints_j, joints_i, shoulder_length)
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
    parser.add_argument('--max_dist', type=float, help='distance threshold to accept as the same shape', default=0.42)
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
                sign_clusters_data[sign_class]["seeders"]["samples"] = f_sign_clusters[sign_class]["seeders"]["samples"][:]
                sign_clusters_data[sign_class]["seeders"]["frames"] = f_sign_clusters[sign_class]["seeders"]["frames"][:]
        else:
            sign_clusters_data = f_sign_clusters

        sign_hand_clusters = {}

        # compute distances of hand-shapes in one sign to create sub-clusters of sign hand-shapes
        # since we have a per-pair custom distance function, we need to do it the old-fashioned way
        for sign_class in sign_clusters_data:
            print("Processing sign {}\n".format(sign_class))
            if sign_class not in sign_hand_clusters:
                sign_hand_clusters[sign_class] = []

            sample_strings = []
            number_of_samples = len(sign_clusters_data[sign_class]["samples"])
            distance_matrix = np.zeros((number_of_samples, number_of_samples))

            future_to_args = []
            start = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
                for i, sample_i in enumerate(sign_clusters_data[sign_class]["samples"]):
                    joints = joints_data[sample_i.decode("utf-8")][0]
                    joints = np.reshape(joints, (-1, 3))
                    shoulder_length = np.linalg.norm(joints[1, :2] - joints[2, :2]).item()
                    sample_strings.append("{}_{}".format(sample_i.decode("utf-8"), sign_clusters_data[sign_class]["frames"][i]))

                    future_to_args.append(
                        executor.submit(compute_distances, i, sample_i, sign_clusters_data[sign_class], joints_data,
                                        shoulder_length))

            for future in concurrent.futures.as_completed(future_to_args):
                try:
                    dist_vec, index = future.result()
                    distance_matrix[index] = dist_vec
                except Exception as exc:
                    print("Generated an exception: {}".format(exc))

            end = time.time()
            print("Distance computation for {} elements: {} s".format(number_of_samples, end - start))

            # start clustering
            start = time.time()
            rows, cols = fast_triu_indices(distance_matrix.shape[0], k=1)
            same_clusters = np.argwhere(distance_matrix[rows, cols] <= args.max_dist)
            same_clusters = np.hstack((rows[same_clusters], cols[same_clusters]))
            hand_added = False
            for same_hand_pose_idx in same_clusters:
                # try to find the indexes in clusters
                for cluster in sign_hand_clusters[sign_class]:
                    # if the sample is already there, try to add the other one
                    if sample_strings[same_hand_pose_idx[0]] in cluster:
                        if sample_strings[same_hand_pose_idx[1]] not in cluster:
                            cluster.append(sample_strings[same_hand_pose_idx[1]])

                        hand_added = True
                        break

                    if sample_strings[same_hand_pose_idx[1]] in cluster:
                        if sample_strings[same_hand_pose_idx[0]] not in cluster:
                            cluster.append(sample_strings[same_hand_pose_idx[0]])

                        hand_added = True
                        break

                # if both of the samples were not in any cluster, add a new cluster
                if not hand_added:
                    sign_hand_clusters[sign_class].append(
                        [sample_strings[same_hand_pose_idx[0]], sample_strings[same_hand_pose_idx[1]]])

            end = time.time()
            print(
                "Clustering for sign {}, with {} entries: {} s".format(sign_class, distance_matrix.shape[0], end - start))

        print("Done")

        pickle.dump(sign_hand_clusters, open("sign_hand_clusters_v03_03.p", "wb"))

    # cluster the sub-clusters
    super_clusters = []
    for i, cluster in enumerate(sign_hand_clusters):
        # create the 1st super cluster
        if i == 0:
            super_clusters.append(cluster)


