
import cv2
import torch

from mbohacek.location_area_checks import *
from mbohacek.config import *


def get_other_arm_to_hand_point(hand: list, body_landmarks: list) -> str:
    if get_landmarks_euclidean_distance(hand[0], body_landmarks[4]) < get_landmarks_euclidean_distance(hand[0], body_landmarks[7]):
        # Point belongs to the right hand -> the other hand is left
        return "left"
    else:
        # Point belongs to the left hand -> the other hand is right
        return "right"


def normalize_dictionary(d: dict, target=1.0):
    raw = sum(d.values())
    factor = target / raw
    return {key: value*factor for key, value in d.items()}


def area_dictionary_to_tensor(d: dict):
    # Order the Tensor values according to config
    tensor_values = [0] * len(AREAS_TENSOR_ORDER)
    for key, value in d.items():
        tensor_values[AREAS_TENSOR_ORDER.index(key)] = value

    # Convert to Torch Tensor
    return torch.as_tensor(tensor_values)


def get_frames_from_video(path):
    output = []
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        read, frame = cap.read()
        if not read:
            break

        output.append(frame)

    cap.release()

    return output
