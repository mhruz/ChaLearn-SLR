
import cv2

import numpy as np
import mediapipe as mp


MP_DRAWING = mp.solutions.drawing_utils
MP_POSE = mp.solutions.pose

# For static images:
POSE_MODEL = MP_POSE.Pose(static_image_mode=True, min_detection_confidence=0.5)


def analyze_body_landmarks(image: np.ndarray):
    """
    Analyzes the image, finds body landmarks and returns the structured data along with an annotated image.

    :param image: Image to analyze
    :return: Tuple of structured landmark coordinates and annotated image
    """

    results = POSE_MODEL.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    output = [(landmark.x, landmark.y) for landmark in results.pose_landmarks.ListFields()[0][1]]
    MP_DRAWING.draw_landmarks(image, results.pose_landmarks, MP_POSE.POSE_CONNECTIONS)

    return output, image


def convert_mp_to_lim_op(landmarks: list):
    """
    Converts the landmarks from the MediaPipe format into the OpenPose format. Root is approximately calculated.

    :param landmarks: List of tuples (MP format)
    :return: List of tuples (OpenPose format)
    """

    output = [None] * 18

    # Root
    output[1] = (landmarks[12][0] + ((landmarks[11][0] - landmarks[12][0]) / 2), landmarks[12][1] +
                 ((landmarks[11][1] - landmarks[12][1]) / 2))

    # Right arm
    output[2] = landmarks[12]
    output[3] = landmarks[14]
    output[4] = landmarks[16]

    # Left arm
    output[5] = landmarks[11]
    output[6] = landmarks[13]
    output[7] = landmarks[15]

    # Right leg
    output[8] = landmarks[24]
    output[9] = landmarks[26]
    output[10] = landmarks[28]

    # Left leg
    output[11] = landmarks[23]
    output[12] = landmarks[25]
    output[13] = landmarks[27]

    # Face
    output[0] = landmarks[0]
    output[14] = landmarks[5]
    output[15] = landmarks[2]
    output[16] = landmarks[8]
    output[17] = landmarks[7]

    return output


if __name__ == "__main__":
    pass
