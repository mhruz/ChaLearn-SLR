import cv2
import numpy as np


def find_mask(image, hand_points, lower_coef=0.75, upper_coef=1.5):
    """
    :param image: image to mask - bgr from opencv
    :param hand_points: concatenated 'hand_right_keypoints_2d' + 'hand_left_keypoints_2d' from open pose
    :param lower_coef: augmentation of lower hsv limit
    :param upper_coef: augmentation of upper hsv limit
    :return: mask of skincolor areas
    """

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lhx = hand_points[::3]
    lhy = hand_points[1::3]

    colorm = np.mean(image_hsv[lhy.astype(np.int), lhx.astype(np.int)], axis=0)
    colorstd = np.std(image_hsv[lhy.astype(np.int), lhx.astype(np.int)], axis=0)

    lower = np.array(np.clip((colorm - colorstd) * lower_coef, 0, 255), dtype="uint8")
    upper = np.array(np.clip((colorm + colorstd) * upper_coef, 0, 255), dtype="uint8")

    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    return skinMask


def hands_head_filter(mask, left_hand_points, right_hand_points, face_points, shift=5):
    """
    :param mask: mask from find_mask function
    :param left_hand_points: hand_left_keypoints_2d' from open pose
    :param right_hand_points: hand_right_keypoints_2d' from open pose
    :param face_points: 'face_keypoints_2d' from openpose
    :param shift: extension of bounding box area around the keypoints
    :return: original mask with head and hands only
    """
    lhx = left_hand_points[::3]
    lhy = left_hand_points[1::3]
    rhx = right_hand_points[::3]
    rhy = right_hand_points[1::3]
    fx = face_points[::3]
    fy = face_points[1::3]

    lhbox = np.array([np.array(x, dtype=np.int) for x in zip(lhx, lhy)])
    rhbox = np.array([np.array(x, dtype=np.int) for x in zip(rhx, rhy)])
    facebox = np.array([np.array(x, dtype=np.int) for x in zip(fx, fy)])

    black = np.zeros_like(mask)

    x, y, w, h = cv2.boundingRect(lhbox)
    black[(y - shift):(y + h + shift), (x - shift):(x + w + shift)] = 255

    x, y, w, h = cv2.boundingRect(rhbox)
    black[(y - shift):(y + h + shift), (x - shift):(x + w + shift)] = 255

    x, y, w, h = cv2.boundingRect(facebox)
    black[(y - shift):(y + h + shift), (x - shift):(x + w + shift)] = 255

    filtered_mask = cv2.bitwise_and(mask, black)

    return filtered_mask

#example_code

# from csv import reader
# import json
# import matplotlib.pyplot as plt
#
# with open('labels_jpg.csv', 'r') as read_obj:
#     csv_reader = reader(read_obj, delimiter=' ')
#
#     for idx, row in enumerate(csv_reader):
#         if idx > 1:
#
#             split = row[1].split('.')[0]
#             image = cv2.imread('./train_jpg/' + row[0])
#             plt.figure(figsize=(10, 10))
#             plt.imshow(image[:, :, ::-1])
#             json_path = r'd:/work/AUTSL/train_json/' + split + r'/' + split + '_' + \
#                         "{:012d}".format(int(row[2])) + '_keypoints.json'
#
#             lh = []
#             rh = []
#
#             with open(json_path, 'r') as fj:
#                 data = json.load(fj)
#                 lh = np.array(data['people'][0]['hand_right_keypoints_2d'])
#                 rh = np.array(data['people'][0]['hand_left_keypoints_2d'])
#                 fp = np.array(data['people'][0]['face_keypoints_2d'])
#
#             hp = np.concatenate((lh, rh))
#             plt.plot(hp[::3], hp[1::3], 'xr')
#             plt.show()
#
#             skinMask = find_mask(image, hand_points=hp)
#
#             #             plt.imshow(skinMask,cmap='gray')
#             #             plt.show()
#
#             skin = cv2.bitwise_and(image, image, mask=skinMask)
#             filtered_skin = hands_head_filter(skinMask, lh, rh, fp)
#             skin_fil = cv2.bitwise_and(image, image, mask=filtered_skin)
#
#             plt.figure(figsize=(10, 10))
#             plt.imshow(skin[:, :, ::-1])
#             plt.show()
#
#             plt.figure(figsize=(10, 10))
#             plt.imshow(skin_fil[:, :, ::-1])
#             plt.show()
#
#             if idx > 3:
#                 break
