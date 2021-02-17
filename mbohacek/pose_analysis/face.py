
import os
import cv2
import dlib
import numpy as np

from imutils import face_utils


DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              "../models/shape_predictor_68_face_landmarks.dat")))


def analyze_face_landmarks(image: np.ndarray):
    """
    Analyzes the image, finds face landmarks and returns the structured data along with an annotated image.

    :param image: Image to analyze
    :return: Tuple of structured landmark coordinates and annotated image
    """

    output = []

    grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_height, image_width, _ = image.shape

    for rect in DETECTOR(grayed_image, 0)[:1]:

        shape = PREDICTOR(grayed_image, rect)
        shape = face_utils.shape_to_np(shape)

        output = [(x / image_width, y / image_height) for (x, y) in shape]

        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    return output, image


if __name__ == "__main__":
    pass
