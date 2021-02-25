
import cv2

from mbohacek.pose_analysis import *
from mbohacek.util import *


def analyze_hand_position(hands_data: list) -> (str, tuple):
    """
    Analyzes the hand landmarks and classifies one of the pre-defined hand gestures and calculates the centroid.

    :param hands_data: Hand landmarks, list of tuple coordinates (values expected relative to the image size)
    :return: Hand gesture classification (str), centroid (tuple(float, float))
    """

    x_values = [landmark[0] for landmark in hands_data]
    y_values = [landmark[1] for landmark in hands_data]

    # Find the surrounding bounding box
    starting_point = (min(x_values), min(y_values))
    ending_point = (max(x_values), max(y_values))
    width = ending_point[0] - starting_point[0]
    height = ending_point[1] - starting_point[1]

    # Construct surrounding bounding box as a square
    extrawidth = max(0, height - width)
    extraheight = max(0, width - height)
    starting_point = (starting_point[0] - (extrawidth / 2), starting_point[1] - (extraheight / 2))
    ending_point = (ending_point[0] + (extrawidth / 2), ending_point[1] + (extraheight / 2))

    # Normalize the data according to the squared bounding box
    x_convert = lambda x: (x - starting_point[0]) / (ending_point[0] - starting_point[0])
    y_convert = lambda y: (y - starting_point[1]) / (ending_point[1] - starting_point[1])
    hands_data = [(x_convert(landmark[0]), y_convert(landmark[1])) for landmark in hands_data]

    # Identify which fingers are up
    index_finger_up = round(get_landmarks_euclidean_distance(hands_data[8], hands_data[5]), 2) > 0.5
    middle_finger_up = round(get_landmarks_euclidean_distance(hands_data[12], hands_data[9]), 2) > 0.5
    ring_finger_up = round(get_landmarks_euclidean_distance(hands_data[16], hands_data[13]), 2) > 0.5
    little_finger_up = round(get_landmarks_euclidean_distance(hands_data[20], hands_data[17]), 2) > 0.5

    # Analyze the hand gesture scenario
    hand_gesture = "other"

    if index_finger_up and not middle_finger_up and not ring_finger_up and not little_finger_up:
        hand_gesture = "pointing"
    elif index_finger_up and middle_finger_up and ring_finger_up and little_finger_up:
        hand_gesture = "open_palm"

    # Calculate the centroid of the hand
    centroid = (starting_point[0] + (ending_point[0] - starting_point[0]) / 2, starting_point[1] + (ending_point[1] - starting_point[1]) / 2)

    # print("-->", hand_gesture)
    return hand_gesture, centroid


def analyze_hands_areas(body_landmarks: list, hands: list, face_landmarks: list, landmarks_analysis_confidence: float = 1):
    """
    The main function which orchestrates the individual areas managers and flags all of the areas that the hand is
    interacting with or is simply located within.

    :param body_landmarks: List of body landmarks
    :param hands: List of lists of hands landmarks
    :param face_landmarks: List of face landmarks
    :param landmarks_analysis_confidence: Confidence of the landmarks analysis based on which to determine the subjected
           landmarks within for the analysis
    :return: List of resulting dictionaries with scores for the areas for each hand
    """

    output = []

    # Prevent from passing two identical hands as different
    if len(hands) == 2:
        if hands[0] == hands[1]:
            hands = [hands[0]]

    # Iterate over the hands (first two found)
    for hand_index, hand in enumerate(hands[:2]):
        hand_gesture, centroid = analyze_hand_position(hand)

        results = {"centroid": {}, "index_tip": {}}

        subjected_points_config = [("centroid", centroid), ("index_tip", hand[8])]
        if landmarks_analysis_confidence < 0.3:
            subjected_points_config = [("centroid", centroid)]

        # Iterate over the individual points to compare and flag for areas
        for subjected_point_id, subjected_point_coordinates in subjected_points_config:

            if not body_landmarks:
                continue

            if face:
                # Area: Above head
                contained, score = is_in_area_above_head(subjected_point_coordinates, body_landmarks, face_landmarks)
                if contained:
                    results[subjected_point_id]["above_head"] = score * HAND_POINT_WEIGHT[hand_gesture][subjected_point_id]

                # Area: Upper face
                contained, score = is_in_area_upper_face(subjected_point_coordinates, body_landmarks, face_landmarks)
                if contained:
                    results[subjected_point_id]["upper_face"] = score * HAND_POINT_WEIGHT[hand_gesture][subjected_point_id]

                # Area: Lower face
                contained, score = is_in_area_lower_face(subjected_point_coordinates, body_landmarks, face_landmarks)
                if contained:
                    results[subjected_point_id]["lower_face"] = score * HAND_POINT_WEIGHT[hand_gesture][subjected_point_id]

                # Area: Eyes
                contained, score = is_in_area_eyes(subjected_point_coordinates, body_landmarks, face_landmarks)
                if contained:
                    results[subjected_point_id]["eyes"] = score * HAND_POINT_WEIGHT[hand_gesture][subjected_point_id]

                # Area: Nose
                contained, score = is_in_area_nose(subjected_point_coordinates, body_landmarks, face_landmarks)
                if contained:
                    results[subjected_point_id]["nose"] = score * HAND_POINT_WEIGHT[hand_gesture][subjected_point_id]

                # Area: Mouth
                contained, score = is_in_area_mouth(subjected_point_coordinates, body_landmarks, face_landmarks)
                if contained:
                    results[subjected_point_id]["mouth"] = score * HAND_POINT_WEIGHT[hand_gesture][subjected_point_id]

                # Area: Cheeks
                contained, score = is_in_area_cheeks(subjected_point_coordinates, body_landmarks, face_landmarks)
                if contained:
                    results[subjected_point_id]["cheeks"] = score * HAND_POINT_WEIGHT[hand_gesture][subjected_point_id]

                # Area: Neck
                contained, score = is_in_area_neck(subjected_point_coordinates, body_landmarks, face_landmarks)
                if contained:
                    results[subjected_point_id]["neck"] = score * HAND_POINT_WEIGHT[hand_gesture][subjected_point_id]

            # Area: Ears
            contained, score = is_in_area_ears(subjected_point_coordinates, body_landmarks)
            if contained:
                results[subjected_point_id]["ears"] = score * HAND_POINT_WEIGHT[hand_gesture][subjected_point_id]

            # Area: Shoulders
            contained, score = is_in_area_shoulders(subjected_point_coordinates, body_landmarks)
            if contained:
                results[subjected_point_id]["shoulders"] = score * HAND_POINT_WEIGHT[hand_gesture][subjected_point_id]

            # Area: Chest
            contained, score = is_in_area_chest(subjected_point_coordinates, body_landmarks)
            if contained:
                results[subjected_point_id]["chest"] = score * HAND_POINT_WEIGHT[hand_gesture][subjected_point_id]

            # Area: Waist
            #contained, score = is_in_area_waist(subjected_point_coordinates, body_landmarks)
            #if contained:
            #    results[subjected_point_id]["waist"] = score * HAND_POINT_WEIGHT[hand_gesture][subjected_point_id]

            # Area: Other hand
            if len(hands) >= 2:
                if hand_index == 0:
                    other_hand_index = 1
                else:
                    other_hand_index = 0

                contained, score = is_in_area_other_hand(subjected_point_coordinates, body_landmarks, hands[other_hand_index])
                if contained:
                    results[subjected_point_id]["other_hand"] = score * HAND_POINT_WEIGHT[hand_gesture][subjected_point_id]

            # Area: Arm
            contained, score = is_in_area_arm(subjected_point_coordinates, body_landmarks,
                                              get_other_arm_to_hand_point(hand, body_landmarks))
            if contained:
                results[subjected_point_id]["arm"] = score * HAND_POINT_WEIGHT[hand_gesture][subjected_point_id]

        # Combine the results for all the subjected points
        overall_results = {}
        for key_point, value_score_dict in results.items():
            for key_area, value_score in value_score_dict.items():
                if value_score == 0:
                    value_score = 0.01

                if key_area not in overall_results:
                    overall_results[key_area] = 0

                overall_results[key_area] += value_score

        # If no areas were flagged, neutral_space is the fallback
        if overall_results == {}:
            overall_results = {"neutral_space": 1}

        # Normalize the values to 1
        normalized_results = normalize_dictionary(overall_results)

        output.append(normalized_results)

    # Append neutral hand analyses until at least 2 are present
    while len(output) < 2:
        overall_results = {"neutral_space": 1}
        output.append(normalize_dictionary(overall_results))

    return output


if __name__ == "__main__":
    pass
