
import os
import cv2
import dlib
import logging
import numpy as np

from imutils import face_utils


DETECTOR_DLIB = dlib.get_frontal_face_detector()
DETECTOR_CV2 = cv2.CascadeClassifier(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                                  "../models/haarcascade_frontalface_default.xml")))

PREDICTOR = dlib.shape_predictor(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              "../models/shape_predictor_68_face_landmarks.dat")))


def analyze_face_landmarks(image: np.ndarray):
    """
    Analyzes the image, finds face landmarks and returns the structured data along with an annotated image.

    :param image: Image to analyze
    :return: Tuple of structured landmark coordinates and annotated image
    """

    grayed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_height, image_width, _ = image.shape

    if DETECTOR_DLIB(grayed_image, 0):
        # Analyze face bounding box using dlib
        face_rectangle = DETECTOR_DLIB(grayed_image, 0)[0]
    else:
        # Analyze face bounding box using OpenCV as fallbach
        x, y, width, height = detect_face_bbox(image)
        face_rectangle = dlib.rectangle(x, y, x + width, y + height)

    # Recognize the individual landmarks
    shape = PREDICTOR(grayed_image, face_rectangle)
    shape = face_utils.shape_to_np(shape)

    # Structure the landmarks into coordinates
    output = [(x / image_width, y / image_height) for (x, y) in shape]

    # Visualize landmarks
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    return output, image


def detect_face_bbox(image: np.ndarray) -> (float, float, float, float):
    """
    Detects the bounding box of the face using OpenCV as the fallback.

    :param image: Image to analyze
    :return: Start x position, start y position, width, height
    """

    # Analyze the grayed image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = DETECTOR_CV2.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        return faces[0]
    else:
        logging.warning("No face detected at all.")
        return None


if __name__ == "__main__":
    pass
