import h5py
import numpy as np


def get_semantic_vector_location_vle(location_vectors, vle_data):
    sample_length = len(location_vectors)
    embedding_dim = vle_data["left_hand"]["embeddings"].shape[1]

    output = np.zeros((sample_length, location_vectors.shape[1] * location_vectors.shape[2] + 2 * embedding_dim))

    # tak the first embedding as the last known hand
    # if there is no embedding for the sample, set it to zeros (fallback)
    if len(vle_data["left_hand"]["embeddings"]) >= 0:
        known_left_hand = vle_data["left_hand"]["embeddings"][0]
    else:
        known_left_hand = np.zeros((1, embedding_dim))

    if len(vle_data["right_hand"]["embeddings"]) >= 0:
        known_right_hand = vle_data["right_hand"]["embeddings"][0]
    else:
        known_right_hand = np.zeros((1, embedding_dim))

    # set the location vectors (left_hand, right_hand)
    output[:, 0:15] = location_vectors[:, 0, :]
    output[:, 15:30] = location_vectors[:, 1, :]

    # iterate through the frames of the sample
    for frame in range(sample_length):
        # set the embeddings
        if frame in vle_data["left_hand"]["frames"]:
            vle_index = np.where(vle_data["left_hand"]["frames"][:] == frame)[0]
            output[frame, 30:30 + 1280] = vle_data["left_hand"]["embeddings"][vle_index]

            known_left_hand = vle_data["left_hand"]["embeddings"][vle_index]
        # if the current frame is not present in the embeddings data, use the last known embedding
        else:
            output[frame, 30:30 + 1280] = known_left_hand

        if frame in vle_data["right_hand"]["frames"]:
            vle_index = np.where(vle_data["right_hand"]["frames"][:] == frame)[0]
            output[frame, 30 + 1280:30 + 2560] = vle_data["right_hand"]["embeddings"][vle_index]

            known_right_hand = vle_data["right_hand"]["embeddings"][vle_index]
        else:
            output[frame, 30 + 1280:30 + 2560] = known_right_hand

    return output


def get_semantic_vector_location_vle_keyframes(location_vectors, vle_data, keyframes):
    sample_length = len(keyframes)
    embedding_dim = vle_data["left_hand"]["embeddings"].shape[1]

    output = np.zeros((sample_length, location_vectors.shape[1] * location_vectors.shape[2] + 2 * embedding_dim))

    # tak the first embedding as the last known hand
    # if there is no embedding for the sample, set it to zeros (fallback)
    if len(vle_data["left_hand"]["embeddings"]) >= 0:
        known_left_hand = vle_data["left_hand"]["embeddings"][0]
    else:
        known_left_hand = np.zeros((1, embedding_dim))

    if len(vle_data["right_hand"]["embeddings"]) >= 0:
        known_right_hand = vle_data["right_hand"]["embeddings"][0]
    else:
        known_right_hand = np.zeros((1, embedding_dim))

    # set the location vectors (left_hand, right_hand)
    keyframes = keyframes[:]
    output[:, 0:15] = location_vectors[keyframes, 0, :]
    output[:, 15:30] = location_vectors[keyframes, 1, :]

    # iterate through the frames of the sample
    output_idx = 0
    for frame in range(len(location_vectors)):
        # set the embeddings
        if frame in vle_data["left_hand"]["frames"]:
            vle_index = np.where(vle_data["left_hand"]["frames"][:] == frame)[0]
            known_left_hand = vle_data["left_hand"]["embeddings"][vle_index]

            if frame in keyframes:
                output[output_idx, 30:30 + 1280] = vle_data["left_hand"]["embeddings"][vle_index]

        # if the current frame is not present in the embeddings data, use the last known embedding
        elif frame in keyframes:
            output[output_idx, 30:30 + 1280] = known_left_hand

        if frame in vle_data["right_hand"]["frames"]:
            vle_index = np.where(vle_data["right_hand"]["frames"][:] == frame)[0]
            known_right_hand = vle_data["right_hand"]["embeddings"][vle_index]

            if frame in keyframes:
                output[output_idx, 30 + 1280:30 + 2560] = vle_data["right_hand"]["embeddings"][vle_index]

        elif frame in keyframes:
            output[output_idx, 30 + 1280:30 + 2560] = known_right_hand

        if frame in keyframes:
            output_idx += 1

    return output


if __name__ == "__main__":
    loc_vectors = h5py.File(r"z:\korpusy_cv\AUTSL\location_vectors.h5", "r")
    vle_data = h5py.File(r"z:\korpusy_cv\AUTSL\vle_hand_crops_train.h5", "r")
    keyframes = h5py.File(r"z:\cv\ChaLearnLAP\key_frames_16.h5", "r")

    sample = "signer0_sample1_color"

    #data = get_semantic_vector_location_vle(loc_vectors[sample], vle_data[sample])
    data = get_semantic_vector_location_vle_keyframes(loc_vectors[sample], vle_data[sample], keyframes[sample])

    print("Done")
