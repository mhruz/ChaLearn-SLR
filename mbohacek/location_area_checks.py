
import numpy as np


def get_landmarks_euclidean_distance(point_0: tuple, point_1: tuple) -> float:
    return np.linalg.norm(np.array((point_0[0], point_0[1])) - np.array((point_1[0], point_1[1])))


def calculate_head_height_metric(body_landmarks: list) -> float:
    return get_landmarks_euclidean_distance(body_landmarks[1], body_landmarks[5])


def get_centroid(arr) -> tuple:
    arr = np.array(arr)
    length, dim = arr.shape
    return tuple(np.array([np.sum(arr[:, i]) / length for i in range(dim)]).tolist())


def is_point_in_area(point: tuple, area_center: tuple, area_size: tuple) -> bool:
    starting_point = (area_center[0] - area_size[0] / 2, area_center[1] - area_size[1] / 2)
    ending_point = (area_center[0] + area_size[0] / 2, area_center[1] + area_size[1] / 2)

    if starting_point[0] <= point[0] <= ending_point[0] and starting_point[1] <= point[1] <= ending_point[1]:
        return True

    return False


def get_area_radius(area_center: tuple, area_size: tuple) -> float:
    starting_point = (area_center[0] - area_size[0] / 2, area_center[1] - area_size[1] / 2)
    return get_landmarks_euclidean_distance(starting_point, area_center)


def is_in_area_above_head(point: tuple, body_landmarks: list, face_landmarks: list) -> (bool, float):
    """
    Determines whether the given point is in the `above head` area.

    :param point: Subjected coordinate (tuple(float, float))
    :param body_landmarks: Full body landmarks (list(tuple(float, float)))
    :param face_landmarks: Full face landmarks (list(tuple(float, float)))
    :return: Is the point in the designated area (bool), score for this particular area calculated from the distance to the center (float)
    """

    centered_point = (face_landmarks[27][0],
                      face_landmarks[27][1] - calculate_head_height_metric(body_landmarks) * 2)

    contained = is_point_in_area(point, centered_point, (calculate_head_height_metric(body_landmarks),
                                                         calculate_head_height_metric(body_landmarks) * 2))
    distance = get_landmarks_euclidean_distance(point, centered_point)

    return contained, 1 - distance / get_area_radius(centered_point, (calculate_head_height_metric(body_landmarks),
                                                                      calculate_head_height_metric(body_landmarks) * 2))


def is_in_area_upper_face(point: tuple, body_landmarks: list, face_landmarks: list) -> (bool, float):
    """
    Determines whether the given point is in the `upper face` area.

    :param point: Subjected coordinate (tuple(float, float))
    :param body_landmarks: Full body landmarks (list(tuple(float, float)))
    :param face_landmarks: Full face landmarks (list(tuple(float, float)))
    :return: Is the point in the designated area (bool), score for this particular area calculated from the distance to the center (float)
    """

    centered_point = (face_landmarks[27][0],
                      face_landmarks[27][1] - calculate_head_height_metric(body_landmarks) / 2)

    contained = is_point_in_area(point, centered_point, (calculate_head_height_metric(body_landmarks),
                                                         calculate_head_height_metric(body_landmarks) / 2))
    distance = get_landmarks_euclidean_distance(point, centered_point)

    return contained, 1 - distance / get_area_radius(centered_point, (calculate_head_height_metric(body_landmarks),
                                                                      calculate_head_height_metric(body_landmarks) / 2))


def is_in_area_eyes(point: tuple, body_landmarks: list, face_landmarks: list) -> (bool, float):
    """
    Determines whether the given point is in the `eyes` area.

    :param point: Subjected coordinate (tuple(float, float))
    :param body_landmarks: Full body landmarks (list(tuple(float, float)))
    :param face_landmarks: Full face landmarks (list(tuple(float, float)))
    :return: Is the point in the designated area (bool), score for this particular area calculated from the distance to the center (float)
    """

    if get_landmarks_euclidean_distance(point, face_landmarks[41]) < get_landmarks_euclidean_distance(point, face_landmarks[46]):
        closer_eye = face_landmarks[40]
    else:
        closer_eye = face_landmarks[47]

    contained = is_point_in_area(point, closer_eye, (calculate_head_height_metric(body_landmarks) * 0.35,
                                                             calculate_head_height_metric(body_landmarks) * 0.38))
    distance = get_landmarks_euclidean_distance(point, closer_eye)

    return contained, 1 - distance / get_area_radius(closer_eye, (
        calculate_head_height_metric(body_landmarks) * 0.35, calculate_head_height_metric(body_landmarks) * 0.38))


def is_in_area_nose(point: tuple, body_landmarks: list, face_landmarks: list) -> (bool, float):
    """
    Determines whether the given point is in the `nose` area.

    :param point: Subjected coordinate (tuple(float, float))
    :param body_landmarks: Full body landmarks (list(tuple(float, float)))
    :param face_landmarks: Full face landmarks (list(tuple(float, float)))
    :return: Is the point in the designated area (bool), score for this particular area calculated from the distance to the center (float)
    """

    contained = is_point_in_area(point, face_landmarks[29], (calculate_head_height_metric(body_landmarks) * 0.35,
                                                             calculate_head_height_metric(body_landmarks) * 0.7))
    distance = get_landmarks_euclidean_distance(point, face_landmarks[29])

    return contained, 1 - distance / get_area_radius(face_landmarks[29], (
        calculate_head_height_metric(body_landmarks) * 0.35, calculate_head_height_metric(body_landmarks) * 0.7))


def is_in_area_mouth(point: tuple, body_landmarks: list, face_landmarks: list) -> (bool, float):
    """
    Determines whether the given point is in the `mouth area.

    :param point: Subjected coordinate (tuple(float, float))
    :param body_landmarks: Full body landmarks (list(tuple(float, float)))
    :param face_landmarks: Full face landmarks (list(tuple(float, float)))
    :return: Is the point in the designated area (bool), score for this particular area calculated from the distance to the center (float)
    """

    contained = is_point_in_area(point, face_landmarks[66], (calculate_head_height_metric(body_landmarks) * 0.7,
                                                             calculate_head_height_metric(body_landmarks) * 0.3))
    distance = get_landmarks_euclidean_distance(point, face_landmarks[66])

    return contained, 1 - distance / get_area_radius(face_landmarks[66], (
        calculate_head_height_metric(body_landmarks) * 0.7, calculate_head_height_metric(body_landmarks) * 0.3))


def is_in_area_lower_face(point: tuple, body_landmarks: list, face_landmarks: list) -> (bool, float):
    """
    Determines whether the given point is in the `lower face` area.

    :param point: Subjected coordinate (tuple(float, float))
    :param body_landmarks: Full body landmarks (list(tuple(float, float)))
    :param face_landmarks: Full face landmarks (list(tuple(float, float)))
    :return: Is the point in the designated area (bool), score for this particular area calculated from the distance to the center (float)
    """

    contained = is_point_in_area(point, face_landmarks[8], (calculate_head_height_metric(body_landmarks),
                                                            calculate_head_height_metric(body_landmarks) / 2))
    distance = get_landmarks_euclidean_distance(point, face_landmarks[8])

    return contained, 1 - distance / get_area_radius(face_landmarks[8], (
        calculate_head_height_metric(body_landmarks), calculate_head_height_metric(body_landmarks) / 2))


def is_in_area_cheeks(point: tuple, body_landmarks: list, face_landmarks: list) -> (bool, float):
    """
    Determines whether the given point is in the `cheecks` area.

    :param point: Subjected coordinate (tuple(float, float))
    :param body_landmarks: Full body landmarks (list(tuple(float, float)))
    :param face_landmarks: Full face landmarks (list(tuple(float, float)))
    :return: Is the point in the designated area (bool), score for this particular area calculated from the distance to the center (float)
    """

    if get_landmarks_euclidean_distance(point, face_landmarks[3]) < get_landmarks_euclidean_distance(point, face_landmarks[13]):
        closer_cheek = (face_landmarks[3][0] + calculate_head_height_metric(body_landmarks) / 6,
                      face_landmarks[3][1])
    else:
        closer_cheek = (face_landmarks[13][0] - calculate_head_height_metric(body_landmarks) / 6,
                      face_landmarks[13][1])

    contained = is_point_in_area(point, closer_cheek, (calculate_head_height_metric(body_landmarks) * 0.3,
                                                     calculate_head_height_metric(body_landmarks) * 0.8))
    distance = get_landmarks_euclidean_distance(point, closer_cheek)

    return contained, 1 - distance / get_area_radius(closer_cheek, (
        calculate_head_height_metric(body_landmarks) * 0.3, calculate_head_height_metric(body_landmarks) * 0.8))


def is_in_area_ears(point: tuple, body_landmarks: list) -> (bool, float):
    """
    Determines whether the given point is in the `ears` area.

    :param point: Subjected coordinate (tuple(float, float))
    :param body_landmarks: Full body landmarks (list(tuple(float, float)))
    :param face_landmarks: Full face landmarks (list(tuple(float, float)))
    :return: Is the point in the designated area (bool), score for this particular area calculated from the distance to the center (float)
    """

    if get_landmarks_euclidean_distance(point, body_landmarks[16]) < get_landmarks_euclidean_distance(point, body_landmarks[17]):
        closer_ear = (body_landmarks[16][0] - calculate_head_height_metric(body_landmarks) / 8,
                      body_landmarks[16][1] + calculate_head_height_metric(body_landmarks) / 5)
    else:
        closer_ear = (body_landmarks[17][0] + calculate_head_height_metric(body_landmarks) / 8,
                      body_landmarks[17][1] + calculate_head_height_metric(body_landmarks) / 5)

    contained = is_point_in_area(point, closer_ear, (calculate_head_height_metric(body_landmarks) * 0.3,
                                                     calculate_head_height_metric(body_landmarks) * 0.6))
    distance = get_landmarks_euclidean_distance(point, closer_ear)

    return contained, 1 - distance / get_area_radius(closer_ear, (
        calculate_head_height_metric(body_landmarks) * 0.3, calculate_head_height_metric(body_landmarks) * 0.6))


def is_in_area_neck(point: tuple, body_landmarks: list, face_landmarks: list) -> (bool, float):
    """
    Determines whether the given point is in the `neck` area.

    :param point: Subjected coordinate (tuple(float, float))
    :param body_landmarks: Full body landmarks (list(tuple(float, float)))
    :param face_landmarks: Full face landmarks (list(tuple(float, float)))
    :return: Is the point in the designated area (bool), score for this particular area calculated from the distance to the center (float)
    """

    centered_point = (body_landmarks[1][0],
                      body_landmarks[1][1] - calculate_head_height_metric(body_landmarks))

    contained = is_point_in_area(point, centered_point, (calculate_head_height_metric(body_landmarks),
                                                         calculate_head_height_metric(body_landmarks) * 0.8))
    distance = get_landmarks_euclidean_distance(point, centered_point)

    return contained, 1 - distance / get_area_radius(centered_point, (calculate_head_height_metric(body_landmarks),
                                                                      calculate_head_height_metric(body_landmarks) * 0.8))


def is_in_area_shoulders(point: tuple, body_landmarks: list) -> (bool, float):
    """
    Determines whether the given point is in the `shoulders` area.

    :param point: Subjected coordinate (tuple(float, float))
    :param body_landmarks: Full body landmarks (list(tuple(float, float)))
    :param face_landmarks: Full face landmarks (list(tuple(float, float)))
    :return: Is the point in the designated area (bool), score for this particular area calculated from the distance to the center (float)
    """

    if get_landmarks_euclidean_distance(point, body_landmarks[2]) < get_landmarks_euclidean_distance(point, body_landmarks[5]):
        closer_shoulder = body_landmarks[2]
    else:
        closer_shoulder = body_landmarks[5]

    contained = is_point_in_area(point, closer_shoulder, (calculate_head_height_metric(body_landmarks),
                                                          calculate_head_height_metric(body_landmarks)))
    distance = get_landmarks_euclidean_distance(point, closer_shoulder)

    return contained, 1 - distance / get_area_radius(closer_shoulder, (
        calculate_head_height_metric(body_landmarks), calculate_head_height_metric(body_landmarks)))


def is_in_area_chest(point: tuple, body_landmarks: list) -> (bool, float):
    """
    Determines whether the given point is in the `chest` area.

    :param point: Subjected coordinate (tuple(float, float))
    :param body_landmarks: Full body landmarks (list(tuple(float, float)))
    :param face_landmarks: Full face landmarks (list(tuple(float, float)))
    :return: Is the point in the designated area (bool), score for this particular area calculated from the distance to the center (float)
    """

    centered_point = (body_landmarks[1][0],
                      body_landmarks[1][1] + calculate_head_height_metric(body_landmarks))

    contained = is_point_in_area(point, centered_point, (2.5 * calculate_head_height_metric(body_landmarks),
                                                         2.5 * calculate_head_height_metric(body_landmarks)))
    distance = get_landmarks_euclidean_distance(point, centered_point)

    return contained, 1 - distance / get_area_radius(centered_point, (2.5 * calculate_head_height_metric(body_landmarks),
                                                                      2.5 * calculate_head_height_metric(body_landmarks)))


def is_in_area_waist(point: tuple, body_landmarks: list) -> (bool, float):
    """
    Determines whether the given point is in the `waist` area.

    :param point: Subjected coordinate (tuple(float, float))
    :param body_landmarks: Full body landmarks (list(tuple(float, float)))
    :param face_landmarks: Full face landmarks (list(tuple(float, float)))
    :return: Is the point in the designated area (bool), score for this particular area calculated from the distance to the center (float)
    """

    centered_point = (body_landmarks[8][0] + ((body_landmarks[11][0] - body_landmarks[8][0]) / 2),
                      body_landmarks[8][1] + ((body_landmarks[11][1] - body_landmarks[8][1]) / 2))

    contained = is_point_in_area(point, centered_point, (2 * calculate_head_height_metric(body_landmarks),
                                                         calculate_head_height_metric(body_landmarks)))
    distance = get_landmarks_euclidean_distance(point, centered_point)

    return contained, 1 - distance / get_area_radius(centered_point, (2 * calculate_head_height_metric(body_landmarks),
                                                         calculate_head_height_metric(body_landmarks)))


def is_in_area_arm(point: tuple, body_landmarks: list, other_hand: str) -> (bool, float):
    """
    Determines whether the given point is in the `arm` area.

    :param point: Subjected coordinate (tuple(float, float))
    :param body_landmarks: Full body landmarks (list(tuple(float, float)))
    :param other_hand:
    :return: Is the point in the designated area (bool), score for this particular area calculated from the distance to the center (float)
    """

    closest_distance_to_arm = 1

    if other_hand == "right":
        arm_lines = [(body_landmarks[2], body_landmarks[3]), (body_landmarks[3], body_landmarks[4])]
    else:
        arm_lines = [(body_landmarks[5], body_landmarks[6]), (body_landmarks[6], body_landmarks[7])]

    for a, b in arm_lines:
        distance = np.abs(np.linalg.norm(np.cross(np.asarray(a) - np.asarray(b), np.asarray(b) - np.asarray(point))) /
                   np.linalg.norm(np.asarray(a) - np.asarray(b)))

        if distance < closest_distance_to_arm:
            closest_distance_to_arm = distance

    return closest_distance_to_arm < calculate_head_height_metric(body_landmarks) / 4, 1 - closest_distance_to_arm / (
                calculate_head_height_metric(body_landmarks) / 4)


def is_in_area_other_hand(point: tuple, body_landmarks: list, hand_landmarks: list) -> (bool, float):
    """
    Determines whether the given point is in the `other hand` area.

    :param point: Subjected coordinate (tuple(float, float))
    :param body_landmarks: Full body landmarks (list(tuple(float, float)))
    :param hand_landmarks: Full landmarks of the other hand that the point does not come from (list(tuple(float, float)))
    :return: Is the point in the designated area (bool), score for this particular area calculated from the distance to the center (float)
    """

    other_hand = get_centroid(hand_landmarks)

    contained = is_point_in_area(point, other_hand, (calculate_head_height_metric(body_landmarks),
                                                          calculate_head_height_metric(body_landmarks) * 2.5))
    distance = get_landmarks_euclidean_distance(point, other_hand)

    return contained, 1 - distance / get_area_radius(other_hand, (
        calculate_head_height_metric(body_landmarks), calculate_head_height_metric(body_landmarks) * 2.5))


if __name__ == "__main__":
    pass
