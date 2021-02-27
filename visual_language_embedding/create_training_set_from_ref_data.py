import argparse
import os
import h5py
import numpy as np
import cv2

if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(
        description='Create training and validation sets of hands from reference (hand-picked) data.')
    parser.add_argument('hand_clusters', type=str, help='path to root with hand cluster images')
    parser.add_argument('train_set', type=str, help='path to output training h5 dataset')
    parser.add_argument('val_set', type=str, help='path to output validation h5 dataset')
    args = parser.parse_args()

    hand_pose_classes = os.listdir(args.hand_clusters)

    hand_pose_classes = [x for x in hand_pose_classes if
                         not x.startswith("_") and os.path.isdir(os.path.join(args.hand_clusters, x))]

    f_train_out = h5py.File(args.train_set, "w")
    f_train_out.create_dataset("images", (100, 70, 70, 3), maxshape=(None, 70, 70, 3), dtype=np.uint8)
    f_train_out.create_dataset("labels", (100, 1), maxshape=(None, 1), dtype=np.int)
    f_val_out = h5py.File(args.val_set, "w")
    f_val_out.create_dataset("images", (100, 70, 70, 3), maxshape=(None, 70, 70, 3), dtype=np.uint8)
    f_val_out.create_dataset("labels", (100, 1), maxshape=(None, 1), dtype=np.int)

    train_data_len = 0
    val_data_len = 0
    val_signers = ["signer8", "signer23"]
    for i, hand_class in enumerate(hand_pose_classes):
        print("Processing cluster {}/{}".format(i + 1, len(hand_pose_classes)))

        images_class = os.listdir(os.path.join(args.hand_clusters, hand_class))
        images_class = [x for x in images_class if x.endswith(".jpg")]

        for image in images_class:
            img = cv2.imread(os.path.join(args.hand_clusters, hand_class, image))

            if any(s in image for s in val_signers):
                f_val_out["images"][val_data_len, :, :, :] = img
                f_val_out["labels"][val_data_len] = i

                val_data_len += 1

                if val_data_len == len(f_val_out["labels"]):
                    f_val_out["images"].resize((val_data_len + 100, 70, 70, 3))
                    f_val_out["labels"].resize((val_data_len + 100, 1))
            else:
                f_train_out["images"][train_data_len, :, :, :] = img
                f_train_out["labels"][train_data_len] = i

                train_data_len += 1

                if train_data_len == len(f_train_out["labels"]):
                    f_train_out["images"].resize((train_data_len + 100, 70, 70, 3))
                    f_train_out["labels"].resize((train_data_len + 100, 1))

    # truncate to actual length
    f_train_out["images"].resize((train_data_len, 70, 70, 3))
    f_train_out["labels"].resize((train_data_len, 1))
    f_val_out["images"].resize((val_data_len, 70, 70, 3))
    f_val_out["labels"].resize((val_data_len, 1))

    f_train_out.close()
    f_val_out.close()
