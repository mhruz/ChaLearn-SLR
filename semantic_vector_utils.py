import h5py
import numpy as np
import cv2


def location_vector_v1_to_v2(location_vectors):
    output = {"frame_number": np.arange(location_vectors.shape[0]), "vector": location_vectors}

    return output


def get_semantic_vector_location_vle(location_vectors, vle_data):
    sample_length = len(location_vectors)
    embedding_dim = vle_data["left_hand"]["embeddings"].shape[1]

    output = np.zeros((sample_length, location_vectors.shape[1] * location_vectors.shape[2] + 2 * embedding_dim))

    # take the first embedding as the last known hand
    # if there is no embedding for the sample, set it to zeros (fallback)
    if len(vle_data["left_hand"]["embeddings"]) > 0:
        known_left_hand = vle_data["left_hand"]["embeddings"][0]
    else:
        known_left_hand = np.zeros((1, embedding_dim))

    if len(vle_data["right_hand"]["embeddings"]) > 0:
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
            vle_index = np.where(vle_data["left_hand"]["frames"][:] == frame)[0][0]
            output[frame, 30:30 + 1280] = vle_data["left_hand"]["embeddings"][vle_index]

            known_left_hand = vle_data["left_hand"]["embeddings"][vle_index]
        # if the current frame is not present in the embeddings data, use the last known embedding
        else:
            output[frame, 30:30 + 1280] = known_left_hand

        if frame in vle_data["right_hand"]["frames"]:
            vle_index = np.where(vle_data["right_hand"]["frames"][:] == frame)[0][0]
            output[frame, 30 + 1280:30 + 2560] = vle_data["right_hand"]["embeddings"][vle_index]

            known_right_hand = vle_data["right_hand"]["embeddings"][vle_index]
        else:
            output[frame, 30 + 1280:30 + 2560] = known_right_hand

    return output


def get_semantic_vector_location_vle_v2(location_vectors, vle_data):
    sample_length = np.max(location_vectors["frame_number"][:]) + 1
    embedding_dim = vle_data["left_hand"]["embeddings"].shape[1]

    output = np.zeros(
        (sample_length, location_vectors["vector"].shape[1] * location_vectors["vector"].shape[2] + 2 * embedding_dim))

    # take the first embedding as the last known hand
    # if there is no embedding for the sample, set it to zeros (fallback)
    if len(vle_data["left_hand"]["embeddings"]) > 0:
        known_left_hand = vle_data["left_hand"]["embeddings"][0]
    else:
        known_left_hand = np.zeros((1, embedding_dim))

    if len(vle_data["right_hand"]["embeddings"]) > 0:
        known_right_hand = vle_data["right_hand"]["embeddings"][0]
    else:
        known_right_hand = np.zeros((1, embedding_dim))

    if len(location_vectors["vector"]) > 0:
        known_loc_vector = location_vectors["vector"][0]
    else:
        known_loc_vector = np.zeros(2, 15)
        known_loc_vector[0, 0] = 1
        known_loc_vector[0, 1] = 1

    # iterate through the frames of the sample
    output_idx = 0
    for frame in range(sample_length):
        # set the location vectors (left_hand, right_hand)
        if frame in location_vectors["frame_number"]:
            loc_index = np.where(location_vectors["frame_number"][:] == frame)[0][0]
            output[output_idx, 0:15] = location_vectors["vector"][loc_index, 0]
            output[output_idx, 15:30] = location_vectors["vector"][loc_index, 1]

            known_loc_vector = location_vectors["vector"]
        else:
            output[output_idx, 0:15] = known_loc_vector[0]
            output[output_idx, 15:30] = known_loc_vector[1]

        # set the embeddings
        if frame in vle_data["left_hand"]["frames"]:
            vle_index = np.where(vle_data["left_hand"]["frames"][:] == frame)[0][0]
            output[output_idx, 30:30 + 1280] = vle_data["left_hand"]["embeddings"][vle_index]

            known_left_hand = vle_data["left_hand"]["embeddings"][vle_index]
        # if the current frame is not present in the embeddings data, use the last known embedding
        else:
            output[output_idx, 30:30 + 1280] = known_left_hand

        if frame in vle_data["right_hand"]["frames"]:
            vle_index = np.where(vle_data["right_hand"]["frames"][:] == frame)[0][0]
            output[output_idx, 30 + 1280:30 + 2560] = vle_data["right_hand"]["embeddings"][vle_index]

            known_right_hand = vle_data["right_hand"]["embeddings"][vle_index]
        else:
            output[output_idx, 30 + 1280:30 + 2560] = known_right_hand

        output_idx += 1

    return output


def get_semantic_vector_location_vle_keyframes(location_vectors, vle_data, keyframes):
    sample_length = len(keyframes)
    embedding_dim = vle_data["left_hand"]["embeddings"].shape[1]

    # hdf5 to numpy
    location_vectors = location_vectors[:, :, :]

    output = np.zeros((sample_length, location_vectors.shape[1] * location_vectors.shape[2] + 2 * embedding_dim))

    # take the first embedding as the last known hand
    # if there is no embedding for the sample, set it to zeros (fallback)
    if len(vle_data["left_hand"]["embeddings"]) > 0:
        known_left_hand = vle_data["left_hand"]["embeddings"][0]
    else:
        known_left_hand = np.zeros((1, embedding_dim))

    if len(vle_data["right_hand"]["embeddings"]) > 0:
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
            vle_index = np.where(vle_data["left_hand"]["frames"][:] == frame)[0][0]
            known_left_hand = vle_data["left_hand"]["embeddings"][vle_index]

            if frame in keyframes:
                output[output_idx, 30:30 + 1280] = vle_data["left_hand"]["embeddings"][vle_index]

        # if the current frame is not present in the embeddings data, use the last known embedding
        elif frame in keyframes:
            output[output_idx, 30:30 + 1280] = known_left_hand

        if frame in vle_data["right_hand"]["frames"]:
            vle_index = np.where(vle_data["right_hand"]["frames"][:] == frame)[0][0]
            known_right_hand = vle_data["right_hand"]["embeddings"][vle_index]

            if frame in keyframes:
                output[output_idx, 30 + 1280:30 + 2560] = vle_data["right_hand"]["embeddings"][vle_index]

        elif frame in keyframes:
            output[output_idx, 30 + 1280:30 + 2560] = known_right_hand

        if frame in keyframes:
            output_idx += 1

    return output


def get_semantic_vector_location_vle_keyframes_v2(location_vectors, vle_data, keyframes):
    sample_length = len(keyframes)
    embedding_dim = vle_data["left_hand"]["embeddings"].shape[1]

    output = np.zeros(
        (sample_length, location_vectors["vector"].shape[1] * location_vectors["vector"].shape[2] + 2 * embedding_dim))

    # take the first embedding as the last known hand
    # if there is no embedding for the sample, set it to zeros (fallback)
    if len(vle_data["left_hand"]["embeddings"]) > 0:
        known_left_hand = vle_data["left_hand"]["embeddings"][0]
    else:
        known_left_hand = np.zeros((1, embedding_dim))

    if len(vle_data["right_hand"]["embeddings"]) > 0:
        known_right_hand = vle_data["right_hand"]["embeddings"][0]
    else:
        known_right_hand = np.zeros((1, embedding_dim))

    if len(location_vectors["vector"]) > 0:
        known_loc_vector = location_vectors["vector"][0]
    else:
        known_loc_vector = np.zeros(2, 15)
        known_loc_vector[0, 0] = 1
        known_loc_vector[0, 1] = 1

    # iterate through the frames of the sample
    output_idx = 0
    for frame in range(len(location_vectors["frame_number"])):
        # set the location vectors (left_hand, right_hand)
        if frame in location_vectors["frame_number"]:
            loc_index = np.where(location_vectors["frame_number"][:] == frame)[0][0]
            known_loc_vector = location_vectors["vector"][loc_index]

            if frame in keyframes:
                output[output_idx, 0:15] = location_vectors["vector"][loc_index, 0]
                output[output_idx, 15:30] = location_vectors["vector"][loc_index, 1]
        elif frame in keyframes:
            output[output_idx, 0:15] = known_loc_vector[0]
            output[output_idx, 15:30] = known_loc_vector[1]

        # set the embeddings
        if frame in vle_data["left_hand"]["frames"]:
            vle_index = np.where(vle_data["left_hand"]["frames"][:] == frame)[0][0]
            known_left_hand = vle_data["left_hand"]["embeddings"][vle_index]

            if frame in keyframes:
                output[output_idx, 30:30 + 1280] = vle_data["left_hand"]["embeddings"][vle_index]

        # if the current frame is not present in the embeddings data, use the last known embedding
        elif frame in keyframes:
            output[output_idx, 30:30 + 1280] = known_left_hand

        if frame in vle_data["right_hand"]["frames"]:
            vle_index = np.where(vle_data["right_hand"]["frames"][:] == frame)[0][0]
            known_right_hand = vle_data["right_hand"]["embeddings"][vle_index]

            if frame in keyframes:
                output[output_idx, 30 + 1280:30 + 2560] = vle_data["right_hand"]["embeddings"][vle_index]

        elif frame in keyframes:
            output[output_idx, 30 + 1280:30 + 2560] = known_right_hand

        if frame in keyframes:
            output_idx += 1

    return output


def normalize_hand_image(image, width):
    if len(image.shape) > 2:
        hand_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        hand_image = image

    if hand_image.shape[1] != width:
        hand_image = cv2.resize(hand_image, (width, width))

    return hand_image / 255.0


def get_semantic_vector_location_hand_crop(location_vectors, hand_crops, width=None):
    if isinstance(location_vectors, np.ndarray):
        location_vectors = location_vector_v1_to_v2(location_vectors)

    elif isinstance(location_vectors, h5py.Dataset):
        location_vectors = location_vector_v1_to_v2(location_vectors[:])

    sample_length = np.max(location_vectors["frame_number"][:]) + 1

    if width is None:
        width = hand_crops["left_hand"][0].shape[1]

    crop_dim = width ** 2

    output = np.zeros(
        (sample_length, location_vectors["vector"].shape[1] * location_vectors["vector"].shape[2] + 2 * crop_dim))

    # take the first embedding as the last known hand
    # if there is no embedding for the sample, set it to zeros (fallback)
    if len(hand_crops["left_hand"]["images"]) > 0:
        hand_image = normalize_hand_image(hand_crops["left_hand"]["images"][0], width)
        known_left_hand = hand_image
    else:
        known_left_hand = np.zeros((1, width))

    if len(hand_crops["right_hand"]["images"]) > 0:
        hand_image = normalize_hand_image(hand_crops["right_hand"]["images"][0], width)
        known_right_hand = hand_image
    else:
        known_right_hand = np.zeros((1, width))

    if len(location_vectors["vector"]) > 0:
        known_loc_vector = location_vectors["vector"][0]
    else:
        known_loc_vector = np.zeros(2, 15)
        known_loc_vector[0, 0] = 1
        known_loc_vector[0, 1] = 1

    # iterate through the frames of the sample
    output_idx = 0
    for frame in range(sample_length):
        # set the location vectors (left_hand, right_hand)
        if frame in location_vectors["frame_number"]:
            loc_index = np.where(location_vectors["frame_number"][:] == frame)[0][0]
            output[output_idx, 0:15] = location_vectors["vector"][loc_index, 0]
            output[output_idx, 15:30] = location_vectors["vector"][loc_index, 1]

            known_loc_vector = location_vectors["vector"]
        else:
            output[output_idx, 0:15] = known_loc_vector[0]
            output[output_idx, 15:30] = known_loc_vector[1]

        # set the hand crops
        if frame in hand_crops["left_hand"]["frames"]:
            hand_index = np.where(hand_crops["left_hand"]["frames"][:] == frame)[0][0]
            hand_image = normalize_hand_image(hand_crops["left_hand"]["images"][hand_index], width)

            output[output_idx, 30:30 + crop_dim] = hand_image.flatten()

            known_left_hand = hand_image
        # if the current frame is not present in the embeddings data, use the last known embedding
        else:
            output[output_idx, 30:30 + crop_dim] = known_left_hand.flatten()

        if frame in hand_crops["right_hand"]["frames"]:
            hand_index = np.where(hand_crops["right_hand"]["frames"][:] == frame)[0][0]
            hand_image = normalize_hand_image(hand_crops["right_hand"]["images"][hand_index], width)

            output[output_idx, 30 + crop_dim:30 + 2 * crop_dim] = hand_image.flatten()

            known_right_hand = hand_image
            # if the current frame is not present in the embeddings data, use the last known embedding
        else:
            output[output_idx, 30 + crop_dim:30 + 2 * crop_dim] = known_right_hand.flatten()

        output_idx += 1

    return output


def get_semantic_vector_location_hand_crop_keyframes(location_vectors, hand_crops, keyframes, width=None):
    if isinstance(location_vectors, np.ndarray):
        location_vectors = location_vector_v1_to_v2(location_vectors)

    elif isinstance(location_vectors, h5py.Dataset):
        location_vectors = location_vector_v1_to_v2(location_vectors[:])

    sample_length = len(keyframes)

    if width is None:
        width = hand_crops["left_hand"][0].shape[1]

    crop_dim = width ** 2

    output = np.zeros(
        (sample_length, location_vectors["vector"].shape[1] * location_vectors["vector"].shape[2] + 2 * crop_dim))

    # take the first embedding as the last known hand
    # if there is no embedding for the sample, set it to zeros (fallback)
    if len(hand_crops["left_hand"]["images"]) > 0:
        hand_image = normalize_hand_image(hand_crops["left_hand"]["images"][0], width)
        known_left_hand = hand_image
    else:
        known_left_hand = np.zeros((1, width))

    if len(hand_crops["right_hand"]["images"]) > 0:
        hand_image = normalize_hand_image(hand_crops["right_hand"]["images"][0], width)
        known_right_hand = hand_image
    else:
        known_right_hand = np.zeros((1, width))

    if len(location_vectors["vector"]) > 0:
        known_loc_vector = location_vectors["vector"][0]
    else:
        known_loc_vector = np.zeros(2, 15)
        known_loc_vector[0, 0] = 1
        known_loc_vector[0, 1] = 1

    # iterate through the frames of the sample
    output_idx = 0
    for frame in range(len(location_vectors["frame_number"])):
        # set the location vectors (left_hand, right_hand)
        if frame in location_vectors["frame_number"]:
            loc_index = np.where(location_vectors["frame_number"][:] == frame)[0][0]
            known_loc_vector = location_vectors["vector"]

            if frame in keyframes:
                output[output_idx, 0:15] = location_vectors["vector"][loc_index, 0]
                output[output_idx, 15:30] = location_vectors["vector"][loc_index, 1]

        elif frame in keyframes:
            output[output_idx, 0:15] = known_loc_vector[0]
            output[output_idx, 15:30] = known_loc_vector[1]

        # set the hand crops
        if frame in hand_crops["left_hand"]["frames"]:
            hand_index = np.where(hand_crops["left_hand"]["frames"][:] == frame)[0][0]
            hand_image = normalize_hand_image(hand_crops["left_hand"]["images"][hand_index], width)
            known_left_hand = hand_image

            if frame in keyframes[:]:
                output[output_idx, 30:30 + crop_dim] = hand_image.flatten()

        # if the current frame is not present in the embeddings data, use the last known embedding
        else:
            output[output_idx, 30:30 + crop_dim] = known_left_hand.flatten()

        if frame in hand_crops["right_hand"]["frames"]:
            hand_index = np.where(hand_crops["right_hand"]["frames"][:] == frame)[0][0]
            hand_image = normalize_hand_image(hand_crops["right_hand"]["images"][hand_index], width)
            known_right_hand = hand_image

            if frame in keyframes[:]:
                output[output_idx, 30 + crop_dim:30 + 2 * crop_dim] = hand_image.flatten()

            # if the current frame is not present in the embeddings data, use the last known embedding
        else:
            output[output_idx, 30 + crop_dim:30 + 2 * crop_dim] = known_right_hand.flatten()

        if frame in keyframes[:]:
            output_idx += 1

    return output


if __name__ == "__main__":
    loc_vectors = h5py.File(r"z:\korpusy_cv\AUTSL\location_vectors_val.h5", "r")
    vle_h5 = h5py.File(r"z:\korpusy_cv\AUTSL\vle_hand_crops_val_v2.h5", "r")
    keyframes = h5py.File(r"z:\cv\ChaLearnLAP\val_key_frames_16.h5", "r")
    hand_crops = h5py.File(r"z:\korpusy_cv\AUTSL\val_hand_images.h5", "r")

    read_to_mem = False

    width = 32

    if read_to_mem:
        loc_vectors_data = {}
        vle_data = {}
        keyframes_data = {}
        hand_crops_data = {}

        for sample in loc_vectors:
            # v1
            # loc_vectors_data[sample] = loc_vectors[sample][:]
            # v2
            loc_vectors_data[sample] = {}
            loc_vectors_data[sample]["frame_number"] = loc_vectors[sample]["frame_number"][:]
            loc_vectors_data[sample]["vector"] = loc_vectors[sample]["vector"][:]

            vle_data[sample] = {}
            vle_data[sample]["left_hand"] = {}
            vle_data[sample]["left_hand"]["frames"] = vle_h5[sample]["left_hand"]["frames"][:]
            vle_data[sample]["left_hand"]["embeddings"] = vle_h5[sample]["left_hand"]["embeddings"][:]

            vle_data[sample]["right_hand"] = {}
            vle_data[sample]["right_hand"]["frames"] = vle_h5[sample]["right_hand"]["frames"][:]
            vle_data[sample]["right_hand"]["embeddings"] = vle_h5[sample]["right_hand"]["embeddings"][:]

            keyframes_data[sample] = keyframes[sample][:]

            hand_crops_data[sample] = {}
            hand_crops_data[sample]["left_hand"] = {}
            hand_crops_data[sample]["left_hand"]["frames"] = hand_crops[sample]["left_hand"]["frames"][:]
            hand_crops_data[sample]["left_hand"]["images"] = np.array(
                [normalize_hand_image(x, width) for x in hand_crops[sample]["left_hand"]["images"][:]])

            hand_crops_data[sample]["right_hand"] = {}
            hand_crops_data[sample]["right_hand"]["frames"] = hand_crops[sample]["right_hand"]["frames"][:]
            hand_crops_data[sample]["right_hand"]["images"] = np.array(
                [normalize_hand_image(x, width) for x in hand_crops[sample]["right_hand"]["images"][:]])

    else:
        loc_vectors_data = loc_vectors
        vle_data = vle_h5
        keyframes_data = keyframes
        hand_crops_data = hand_crops

    # sample = "signer1_sample1_color"
    sample = "signer35_sample9_color"
    data = get_semantic_vector_location_vle_v2(loc_vectors_data[sample], vle_data[sample])
    data2 = get_semantic_vector_location_vle_keyframes_v2(loc_vectors_data[sample], vle_data[sample],
                                                          keyframes_data[sample])
    data3 = get_semantic_vector_location_hand_crop(loc_vectors_data[sample], hand_crops_data[sample], 32)
    data4 = get_semantic_vector_location_hand_crop_keyframes(loc_vectors_data[sample], hand_crops_data[sample],
                                                             keyframes[sample], 10)

    print("Done")
