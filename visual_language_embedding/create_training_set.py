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

    f_out = h5py.File(args.output, "w")
    f_out.create_dataset("train_images", (100, 70, 70, 3), maxshape=(None, 70, 70, 3), dtype=np.uint8)
    f_out.create_dataset("val_images", (100, 70, 70, 3), maxshape=(None, 70, 70, 3), dtype=np.uint8)
    f_out.create_dataset("train_labels", (100, 1), maxshape=(None, 1), dtype=np.int)
    f_out.create_dataset("val_labels", (100, 1), maxshape=(None, 1), dtype=np.int)

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
                    f_out["val_images"][val_data_len, :, :, :] = img
                    f_out["val_labels"][val_data_len] = i

                    val_data_len += 1

                    if val_data_len == len(f_out["val_labels"]):
                        f_out["val_images"].resize((val_data_len + 100, 70, 70, 3))
                        f_out["val_labels"].resize((val_data_len + 100, 1))
                else:
                    f_out["train_images"][train_data_len, :, :, :] = img
                    f_out["train_labels"][train_data_len] = i

                    train_data_len += 1

                    if train_data_len == len(f_out["labels"]):
                        f_out["train_images"].resize((train_data_len + 100, 70, 70, 3))
                        f_out["train_labels"].resize((train_data_len + 100, 1))

    # truncate to actual length
    f_out["train_images"].resize(train_data_len, 70, 70, 3)
    f_out["train_labels"].resize(train_data_len, 1)
    f_out["val_images"].resize(val_data_len, 70, 70, 3)
    f_out["val_labels"].resize(val_data_len, 1)

    f_out.close()
