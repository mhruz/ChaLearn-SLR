import argparse
import os
import cv2
import numpy as np
import random
import h5py
import csv
import matplotlib.pyplot as plt


def draw_joints(im, joints):
    joints = np.reshape(joints, (-1, 3))

    for i, joint in enumerate(joints):
        cv2.circle(im, (joint[0], joint[1]), 2, (0, 255, 0), thickness=-1)


def get_right_hand(im, joints, border=0.2, square=False):
    joints = np.reshape(joints, (-1, 3))
    right_hand_joints = joints[29:50]

    hand_image, m = get_sub_image(im, right_hand_joints, border, square)

    return hand_image, m


def get_left_hand(im, joints, border=0.2, square=False):
    joints = np.reshape(joints, (-1, 3))
    left_hand_joints = joints[8:29]

    hand_image, m = get_sub_image(im, left_hand_joints, border, square)

    return hand_image, m


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

    return sub_image, mean


if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Extract confident hand images from videos that have OpenPose joints.')
    parser.add_argument('video_path', type=str, help='path to videos with signs')
    parser.add_argument('open_pose_h5', type=str, help='path to H5 with detected joint locations')
    parser.add_argument('--threshold', type=float, help='optional confidence threshold, default=0.4', default=0.4)
    parser.add_argument('--visualize', type=bool, help='optional visualization, default=False', default=False)
    parser.add_argument('--draw_joints', type=bool, help='optional visualization of joints, default=False',
                        default=False)
    parser.add_argument('--out_h5', type=str, help='optional output h5 dataset')
    parser.add_argument('--out_stat_dir', type=str, help='optional output stats directory (data & images)')
    parser.add_argument('--video_to_class_csv', type=str, help='optional csv with video labels')
    parser.add_argument('--out_size', type=int, help='size of images in h5 file, default=70', default=70)
    args = parser.parse_args()

    # variable initialization
    pause = 40
    right_hands = None
    left_hands = None
    left_hand_frame = 0
    right_hand_frame = 0
    left_hands_frames = None
    right_hands_frames = None
    f = None

    stats = {}

    if args.out_h5 is not None:
        f = h5py.File(args.out_h5, "w")

    if args.out_stat_dir is not None:
        if args.video_to_class_csv is None:
            print("When using stat dir you need to specify the --video_to_class_csv arg.")

        f_video_to_class = csv.reader(open(args.video_to_class_csv, "r"))
        os.makedirs(args.out_stat_dir, exist_ok=True)

    joints_h5 = h5py.File(args.open_pose_h5, "r")

    if args.visualize:
        cv2.namedWindow("image", 0)
        cv2.namedWindow("left hand", 0)
        cv2.namedWindow("right hand", 0)

    # random.shuffle(video_filenames)

    #kill = 0
    for video_fn in joints_h5:

        # kill += 1
        # if kill == 10:
        #     break

        video = cv2.VideoCapture(os.path.join(args.video_path, video_fn + ".mp4"))
        number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        number_of_joint_frames = len(joints_h5[video_fn])
        print("Processing video: {}".format(video_fn))
        frame = 0

        if number_of_frames != number_of_joint_frames:
            print("Warning! Inconsistent number of frames"
                  " of video and OpenPose joints ({}/{})".format(number_of_frames, number_of_joint_frames))

        if args.out_h5 is not None:
            left_hands = np.zeros((number_of_joint_frames, args.out_size, args.out_size, 3), dtype=np.uint8)
            right_hands = np.zeros((number_of_joint_frames, args.out_size, args.out_size, 3), dtype=np.uint8)

            left_hands_frames = []
            right_hands_frames = []

            left_hand_frame = 0
            right_hand_frame = 0

        if args.out_stat_dir is not None:
            left_hand_means = np.zeros(number_of_joint_frames)
            right_hand_means = np.zeros(number_of_joint_frames)

            speaker, sample = video_fn.split("_")[:2]

        while frame < number_of_joint_frames:

            joints = joints_h5[video_fn][frame]
            # draw_joints(im, joints)

            if args.out_h5 is not None:
                # grab image
                ret, im = video.read()
                if not ret:
                    break
                # get the right hand image
                right_hand_image, mrh = get_right_hand(im, joints, square=True)
                # get the left hand image
                left_hand_image, mlh = get_left_hand(im, joints, square=True)
            else:
                joints = np.reshape(joints, (-1, 3))
                mrh = np.mean(joints[29:50, 2])
                mlh = np.mean(joints[8:29, 2])

            if args.out_stat_dir is not None:
                left_hand_means[frame] = mlh
                right_hand_means[frame] = mrh

            if args.out_h5 is not None:
                if mrh >= args.threshold:
                    right_hand_image = cv2.resize(right_hand_image, (args.out_size, args.out_size))
                    right_hands[right_hand_frame] = right_hand_image
                    right_hand_frame += 1
                    right_hands_frames.append(frame)

                if mlh >= args.threshold:
                    left_hand_image = cv2.resize(left_hand_image, (args.out_size, args.out_size))
                    left_hands[left_hand_frame] = left_hand_image
                    left_hand_frame += 1
                    left_hands_frames.append(frame)

            if args.visualize:
                cv2.imshow("left hand", left_hand_image)
                cv2.imshow("right hand", right_hand_image)

                print("Left hand image size: {}".format(left_hand_image.shape))
                print("Right hand image size: {}".format(right_hand_image.shape))
                print("Left Hand conf: mean {}, sum {}\nRight Hand conf: mean {}, sum {}".format(mlh, mrh))

                cv2.imshow("image", im)
                key = cv2.waitKey(pause)
                if key == 32:
                    pause = abs(pause - 40)

            frame += 1

        if args.out_stat_dir is not None:
            if speaker not in stats:
                stats[speaker] = {}

            if sample not in stats[speaker]:
                stats[speaker][sample] = {}

            stats[speaker][sample]["left_hand_means"] = np.mean(left_hand_means)
            stats[speaker][sample]["left_hand_stds"] = np.std(left_hand_means)
            stats[speaker][sample]["right_hand_means"] = np.mean(right_hand_means)
            stats[speaker][sample]["right_hand_stds"] = np.std(right_hand_means)

        if args.out_h5 is not None:
            f.create_group(video_fn)
            f[video_fn].create_group("left_hand")
            f[video_fn].create_group("right_hand")
            f[video_fn]["left_hand"].create_dataset("images", shape=(left_hand_frame, args.out_size, args.out_size, 3),
                                                    dtype=np.uint8, data=left_hands[:left_hand_frame])
            f[video_fn]["left_hand"].create_dataset("frames", shape=(left_hand_frame,), dtype=np.int,
                                                    data=left_hands_frames)
            f[video_fn]["right_hand"].create_dataset("images",
                                                     shape=(right_hand_frame, args.out_size, args.out_size, 3),
                                                     dtype=np.uint8, data=right_hands[:right_hand_frame])
            f[video_fn]["right_hand"].create_dataset("frames", shape=(right_hand_frame,), dtype=np.int,
                                                     data=right_hands_frames)

            f.flush()

    if args.out_h5 is not None:
        f.close()

    if args.visualize:
        cv2.destroyAllWindows()

    if args.out_stat_dir:
        for speaker in stats:
            print("Saving stats for {}".format(speaker))
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.5)
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

            for sample in stats[speaker]:
                ax1.set_title("Left Hand")
                ax1.scatter(stats[speaker][sample]["left_hand_means"], stats[speaker][sample]["left_hand_stds"],
                            c="blue", s=4)

                ax2.set_title("Right Hand")
                ax2.scatter(stats[speaker][sample]["right_hand_means"], stats[speaker][sample]["right_hand_stds"],
                            c="red", s=4)

            ax1.set_xlabel("Mean")
            ax1.set_ylabel("Std")
            ax2.set_xlabel("Mean")
            ax2.set_ylabel("Std")

            ax1.set_xticks(np.arange(0, 1, 0.1))
            ax2.set_xticks(np.arange(0, 1, 0.1))

            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            fig.savefig(os.path.join(args.out_stat_dir, "{}.png".format(speaker)))
            fig.close()

        f_stats = h5py.File(os.path.join(args.out_stat_dir, "stats.h5"), "w")
        f_stats.create_group(speaker)
        data = stats[speaker][sample]
        f_stats[speaker].create_dataset(sample, (4,),
                                        data=[data["left_hand_means"], data["left_hand_stds"], data["right_hand_means"],
                                              data["right_hand_stds"]])
