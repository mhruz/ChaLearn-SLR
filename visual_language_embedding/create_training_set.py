import argparse
import pickle
import h5py
import numpy as np

if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Create training and validation sets of hands.')
    parser.add_argument('hand_crops', type=str, help='path to h5 with hand crop images')
    parser.add_argument('hand_clusters', type=str, help='path to pickle with hand clusters')
    parser.add_argument('train_set', type=str, help='path to output training h5 dataset')
    parser.add_argument('val_set', type=str, help='path to output validation h5 dataset')
    args = parser.parse_args()

    data = pickle.load(open(args.hand_clusters, "rb"))
    f_hand_crops = h5py.File(args.hand_crops, "r")

    f_train_out = h5py.File(args.train_set, "w")
    f_train_out.create_dataset("images", (100, 70, 70, 3), maxshape=(None, 70, 70, 3), dtype=np.uint8)
    f_train_out.create_dataset("labels", (100, 1), maxshape=(None, 1), dtype=np.int)
    f_val_out = h5py.File(args.val_set, "w")
    f_val_out.create_dataset("images", (100, 70, 70, 3), maxshape=(None, 70, 70, 3), dtype=np.uint8)
    f_val_out.create_dataset("labels", (100, 1), maxshape=(None, 1), dtype=np.int)

    final_clusters = data["hand_clusters"]
    index_to_representative = data["index_to_representative"]
    hand_samples = data["hand_samples"]
    sign_hand_clusters = data["sign_hand_clusters"]

    train_data_len = 0
    val_data_len = 0
    val_signers = ["signer40", "signer42"]
    for i, hand_cluster in enumerate(final_clusters):
        for hand_repre in hand_cluster:
            cluster_info = index_to_representative[hand_repre]
            sign_class = cluster_info["sign_class"]

            for hand in sign_hand_clusters[sign_class]["clusters"][cluster_info["cluster"]]:
                sample = sign_hand_clusters[sign_class]["samples"]["samples"][hand].decode("utf-8")
                frame = sign_hand_clusters[sign_class]["samples"]["frames"][hand]

                img_idx = np.where(f_hand_crops["{}".format(sample)]["left_hand"]["frames"][:] == frame)
                img = f_hand_crops["{}".format(sample)]["left_hand"]["images"][img_idx][0]

                if any(s in sample for s in val_signers):
                    f_val_out["images"][val_data_len, :, :, :] = img
                    f_val_out["labels"][val_data_len] = i

                    val_data_len += 1

                    if val_data_len == len(f_val_out["val_labels"]):
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
    f_train_out["images"].resize(train_data_len, 70, 70, 3)
    f_train_out["labels"].resize(train_data_len, 1)
    f_val_out["images"].resize(val_data_len, 70, 70, 3)
    f_val_out["labels"].resize(val_data_len, 1)

    f_train_out.close()
    f_val_out.close()
