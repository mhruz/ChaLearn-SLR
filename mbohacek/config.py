
AREAS_TENSOR_ORDER = ["neutral_space", "above_head", "upper_face", "eyes", "nose", "mouth", "lower_face", "cheeks", "ears", "neck", "shoulders", "chest", "waist", "arm", "other_hand"]

HAND_POINT_WEIGHT = {
    "pointing": {"centroid": 0, "index_tip": 1},
    "open_palm": {"centroid": 0.5, "index_tip": 0.5},
    "other": {"centroid": 0.5, "index_tip": 0.5}
}
