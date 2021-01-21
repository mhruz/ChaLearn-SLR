import argparse
import h5py
import os

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

    for video_fn in video_filenames:
