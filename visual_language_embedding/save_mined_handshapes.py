import argparse
import pickle
import os
import h5py
import numpy as np
import cv2


if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Save mined hand shapes as images.')
    parser.add_argument('hands_pickle', type=str, help='path to pickle with mined data')
    parser.add_argument('hand_crops', type=str, help='h5 with cropped images of hands')
    parser.add_argument('data_root', type=str, help='root of the output path')
    args = parser.parse_args()

    data = pickle.load(open(args.hands_pickle, "rb"))
    f_hand_crops = h5py.File(args.hand_crops, "r")

    for hand_class in data:
        if len(data[hand_class]) == 0:
            continue

        os.makedirs(os.path.join(args.data_root, hand_class), exist_ok=True)

        for sample in data[hand_class]:
            image_fn = sample["sample"]
            sample_ref = image_fn[:image_fn.find("_color") + 6]
            frame_ref = int(image_fn[image_fn.find("_color") + 6:-4])

            img_idx = np.where(f_hand_crops["{}".format(sample_ref)]["left_hand"]["frames"][:] == frame_ref)
            img = f_hand_crops["{}".format(sample_ref)]["left_hand"]["images"][img_idx][0]

            cv2.imwrite(os.path.join(args.data_root, hand_class, image_fn), img)
