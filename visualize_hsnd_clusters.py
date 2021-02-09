import argparse
import pickle
import os
import h5py
import numpy as np
import cv2


if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Visualize sign hand clusters.')
    parser.add_argument('sign_clusters_pickle', type=str, help='h5 with hand clusters by signs')
    parser.add_argument('hand_crops', type=str, help='h5 with hand crops')
    parser.add_argument('out_path', type=str, help='output path to save the images')
    args = parser.parse_args()

    clusters = pickle.load(open(args.sign_clusters_h5, "rb"))
    hand_crops = h5py.File(args.hand_crops, "r")

    for i, cluster in enumerate(clusters):
        cluster_path = os.path.join(args.sout_path, "{:03d}".format(i))
        os.makedirs(cluster_path, exist_ok=True)
        for hand_string in cluster:
            parts = hand_string.split("_")
            sample = "_".join(parts[:3])
            frame = int(parts[3])

            img_idx = np.where(hand_crops["{}".format(sample)]["left_hand"]["frames"][:] == frame)
            hand_img = hand_crops["{}".format(sample)]["left_hand"]["images"][img_idx][0]

            cv2.imwrite(os.path.join(cluster_path, hand_string + ".jpg"), hand_img)
