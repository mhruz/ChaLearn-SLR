import argparse
import os
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Extracts key-frames representing a SL video.')
    parser.add_argument('joints_h5', type=str, help='path to h5 with hand hand joints')
    parser.add_argument('key_frames', type=int, help='how many key-frames extract per video')
    parser.add_argument('--video_root', type=str, help='root of vide files for optional verbose')
    parser.add_argument('output', type=str, help='output h5 with per-video key-frames')
    args = parser.parse_args()

    joints_h5 = h5py.File(args.joints_h5, "r")
    indexes = list(range(len(joints_h5)))
    random.shuffle(indexes)

    f_out = h5py.File(args.output, "w")

    for index in indexes:
        sample = list(joints_h5.keys())[index]
        f_out.create_dataset(sample, shape=(args.key_frames, ), dtype=np.int)

        if args.video_root is not None:
            cap = cv2.VideoCapture(os.path.join(args.video_root, sample + ".mp4"))

        joints = joints_h5[sample]
        joints = np.reshape(joints, (-1, 50, 3))

        # hand_conf = np.mean(joints[:, 8:29, 2], axis=1)

        # compute the movement speed of joints
        velocity = np.diff(joints[:, :, :2], axis=0)
        speed = np.linalg.norm(velocity, axis=2)

        whole_body_speed = np.sum(speed, axis=1)

        # idx = np.argpartition(whole_body_speed, args.key_frames)
        # idx = idx[:args.key_frames]

        min_idxs = []
        window = 3
        whole_body_speed_copy = np.copy(whole_body_speed)
        for i in range(args.key_frames):
            idx2 = np.argmin(whole_body_speed_copy)
            min_idxs.append(idx2)
            whole_body_speed_copy[idx2-window:idx2+window] = np.inf
            whole_body_speed_copy[idx2] = np.inf

        min_idxs = np.sort(min_idxs)
        f_out[sample][:] = min_idxs[:]

        if args.video_root is not None:
            for n, i in enumerate(min_idxs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, im = cap.read()

                cv2.namedWindow(str(i), 2)
                cv2.imshow(str(i), im)
                cv2.moveWindow(str(i), n % 5 * 300, n // 5 * 300 + 100)

            # cv2.waitKey()

            plt.plot(whole_body_speed)
            # plt.plot(idx, whole_body_speed[idx], 'o')
            plt.plot(min_idxs, whole_body_speed[min_idxs], 'x')
            plt.title(sample)
            plt.show()

            cv2.destroyAllWindows()

    f_out.close()
