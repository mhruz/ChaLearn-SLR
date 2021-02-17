
import cv2

import numpy as np
import mediapipe as mp


MP_DRAWING = mp.solutions.drawing_utils
MP_HANDS = mp.solutions.hands

HANDS_MODEL = MP_HANDS.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)


def analyze_hand_landmarks(image: np.ndarray):
    """
    Analyzes the image, finds hand landmarks and returns the structured data along with an annotated image.

    :param image: Image to analyze
    :return: Tuple of structured landmark coordinates and annotated image
    """

    output = []

    results = HANDS_MODEL.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height, image_width, _ = image.shape

    for hand_landmarks in results.multi_hand_landmarks:
        output.append([(landmark.x, landmark.y) for landmark in hand_landmarks.ListFields()[0][1]])
        MP_DRAWING.draw_landmarks(image, hand_landmarks, MP_HANDS.HAND_CONNECTIONS)

    return output, cv2.flip(image, 1)


if __name__ == "__main__":
    pass
