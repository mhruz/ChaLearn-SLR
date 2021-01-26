import argparse
import h5py
import os
import cv2
import numpy as np
import random
import h5py


def draw_joints(im, joints):
    joints = np.reshape(joints, (-1, 3))

    for i, joint in enumerate(joints):
        cv2.circle(im, (joint[0], joint[1]), 2, (0, 255, 0), thickness=-1)


def get_right_hand(im, joints, border=0.2, square=False):
    joints = np.reshape(joints, (-1, 3))
    right_hand_joints = joints[29:49]

    hand_image, m, s = get_sub_image(im, right_hand_joints, border, square)

    return hand_image, m, s


def get_left_hand(im, joints, border=0.2, square=False):
    joints = np.reshape(joints, (-1, 3))
    left_hand_joints = joints[8:28]

    hand_image, m, s = get_sub_image(im, left_hand_joints, border, square)

    return hand_image, m, s


def get_sub_image(im, joints, border=0.2, square=False):
    x0 = np.min(joints[:, 0])
    x1 = np.max(joints[:, 0])

    y0 = np.min(joints[:, 1])
    y1 = np.max(joints[:, 1])

    width = x1 - x0
    height = y1 - y0

    border_width = width * border / 2.0
    border_height = height * border / 2.0

    x0 = np.maximum(0, x0 - border_width)
    x1 = np.minimum(im.shape[1], x1 + border_width)
    y0 = np.maximum(0, y0 - border_height)
    y1 = np.minimum(im.shape[0], y1 + border_height)

    x0 = int(np.round(x0))
    x1 = int(np.round(x1))
    y0 = int(np.round(y0))
    y1 = int(np.round(y1))

    if square:
        final_width = x1 - x0
        final_height = y1 - y0
        center = ((x0 + x1) / 2, (y0 + y1) / 2)
        rectangle_size_half = np.maximum(final_height, final_width) // 2

        x0 = center[0] - rectangle_size_half
        x1 = center[0] + rectangle_size_half
        y0 = center[1] - rectangle_size_half
        y1 = center[1] + rectangle_size_half

        # check whether we are out of bounds
        if x0 < 0:
            x1 -= x0
            x0 -= x0

        if x1 > im.shape[1]:
            x0 -= x1 - im.shape[1]
            x1 -= x1 - im.shape[1]

        if y0 < 0:
            y1 -= y0
            y0 -= y0

        if y1 > im.shape[0]:
            y0 -= y1 - im.shape[0]
            y1 -= y1 - im.shape[0]

        # final out of bounds check (if the original region is larger than image)
        x0 = np.maximum(0, x0)
        x1 = np.minimum(im.shape[1], x1)
        y0 = np.maximum(0, y0)
        y1 = np.minimum(im.shape[0], y1)

        x0 = int(np.round(x0))
        x1 = int(np.round(x1))
        y0 = int(np.round(y0))
        y1 = int(np.round(y1))

    sub_image = im[y0:y1, x0:x1, :]

    conf = joints[:, 2]
    mean = np.mean(conf)
    suma = np.sum(conf)

    return sub_image, mean, suma


if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Extract confident hand images from videos that have OpenPose joints.')
    parser.add_argument('video_path', type=str, help='path to videos with signs')
    parser.add_argument('open_pose_h5', type=str, help='path to H5 with detected joint locations')
    parser.add_argument('--threshold', type=float, help='optional confidence threshold, default=0.5', default=0.5)
    args = parser.parse_args()

    joints_h5 = h5py.File(args.open_pose_h5, "r")

    video_filenames = os.listdir(args.video_path)
    video_filenames = [x for x in video_filenames if x.endswith("_color.mp4")]

    pause = 40

    cv2.namedWindow("image", 0)
    cv2.namedWindow("left hand", 0)
    cv2.namedWindow("right hand", 0)

    random.shuffle(video_filenames)

    for video_fn in video_filenames:
        video = cv2.VideoCapture(os.path.join(args.video_path, video_fn))
        frame = 0

        while True:

            ret, im = video.read()
            if not ret:
                break

            joints = joints_h5[video_fn[:-4]][frame]
            draw_joints(im, joints)

            # get the right hand image
            right_hand_image, mrh, srh = get_right_hand(im, joints, square=True)
            # get the left hand image
            left_hand_image, mlh, slh = get_left_hand(im, joints, square=True)

            cv2.imshow("left hand", left_hand_image)
            cv2.imshow("right hand", right_hand_image)

            print("Left hand image size: {}".format(left_hand_image.shape))
            print("Right hand image size: {}".format(right_hand_image.shape))
            print("Left Hand conf: mean {}, sum {}\nRight Hadn conf: mean {}, sum {}".format(mlh, slh, mrh, srh))

            cv2.imshow("image", im)
            key = cv2.waitKey(pause)
            if key == 32:
                pause = abs(pause - 40)

            frame += 1


    cv2.destroyWindow()