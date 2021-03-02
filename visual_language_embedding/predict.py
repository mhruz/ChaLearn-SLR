import argparse
import h5py
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional
import torch
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

embedding_vectors = []


def embedding_hook(module, input):
    global embedding_vectors
    embedding_vectors = input


if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Predict VLE on ChaLearn-SLR data.')
    parser.add_argument('hand_crops_h5', type=str, help='h5 with cropped images of hands')
    parser.add_argument('net', type=str, help='path to network you want to test ')
    parser.add_argument('model', type=str, help='resnet-18, mobilenet, vle')
    parser.add_argument('num_classes', type=int, help='number of classes')
    parser.add_argument('output_h5', type=str, help='output h5 with predictions')
    parser.add_argument('--batch_size', type=int, help='number data in one batch', default=32)
    parser.add_argument('--data_to_mem', type=bool, help='load data to memory')
    parser.add_argument('--resize', type=int, help='resize images to this size')
    parser.add_argument('--device', type=int, help='device number', default=0)
    parser.add_argument('--min_conf', type=float, help='predict only hand-shapes with OpenPose confidence > min_conf',
                        default=0.4)
    parser.add_argument('--open_pose_h5', type=str, help='OpenPose predictions')
    parser.add_argument('--joints_to_mem', type=bool, help='read joints data to memory')
    args = parser.parse_args()

    f_hand_crops = h5py.File(args.hand_crops_h5, "r")
    f_out = h5py.File(args.output_h5, "w")

    if args.open_pose_h5 is not None:
        # read joints into memory
        f_joints = h5py.File(args.open_pose_h5)
        if args.joints_to_mem is not None:
            joints_data = {}
            for video_fn in f_joints:
                joints_data[video_fn] = f_joints[video_fn][:]
        else:
            joints_data = f_joints

    batch_size = args.batch_size

    if args.model == "resnet-18":
        net = models.resnet18(pretrained=False)
        # replace classification layer
        net.fc = nn.Linear(512, args.num_classes)
        embedding_dim = 512
    elif args.model == "mobilenet":
        net = models.mobilenet_v2(pretrained=False)
        # replace classification layer
        net.classifier[1] = nn.Linear(1280, args.num_classes)
        embedding_dim = 1280
        net.classifier[1].register_forward_pre_hook(embedding_hook)

    # net = VLE_01(65)
    net.cuda(device="cuda:{}".format(args.device))
    start_epoch = 0

    if args.net.endswith(".tar"):
        checkpoint = torch.load(args.net, map_location="cuda:{}".format(args.device))
        net.load_state_dict(checkpoint["model_state_dict"])
    else:
        net.load_state_dict(torch.load(args.net), map_location="cuda:{}".format(args.device))

    if args.resize is None:
        val_transform = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        val_transform = A.Compose([
            A.Resize(args.resize, args.resize),
            A.Normalize(),
            ToTensorV2()
        ])

    # compute test acc
    net.eval()

    num_batches = 0
    input_data = []
    batch_groups = []
    batch_indexes = []
    batch_frames = []
    batch_full = False

    for sample in f_hand_crops:
        group_left = f_out.create_group("{}/left_hand".format(sample))
        group_right = f_out.create_group("{}/right_hand".format(sample))

        group_left.create_dataset("frames", shape=f_hand_crops[sample]["left_hand"]["frames"].shape, dtype=np.int,
                                  chunks=True)
        group_left.create_dataset("embeddings",
                                  shape=(len(f_hand_crops[sample]["left_hand"]["frames"]), embedding_dim),
                                  dtype=np.float, chunks=True)
        group_right.create_dataset("frames", shape=f_hand_crops[sample]["right_hand"]["frames"].shape, dtype=np.int,
                                   chunks=True)
        group_right.create_dataset("embeddings",
                                   shape=(len(f_hand_crops[sample]["right_hand"]["frames"]), embedding_dim),
                                   dtype=np.float, chunks=True)
        # save left hands
        data_idx_left = 0
        for i, (frame, hand) in enumerate(zip(f_hand_crops[sample]["left_hand"]["frames"],
                                              f_hand_crops[sample]["left_hand"]["images"])):
            # check OpenPose min confidence
            if args.open_pose_h5 is not None:
                target_joints = f_joints[sample][frame]
                target_joints = np.reshape(target_joints, (-1, 3))
                mlh = np.mean(target_joints[8:29, 2]).item()
                # low Open Pose confidence
                if mlh < args.min_conf:
                    continue

            data_sample = val_transform(image=hand)["image"]
            input_data.append(data_sample)
            batch_groups.append(group_left)
            batch_indexes.append(data_idx_left)
            batch_frames.append(frame)
            data_idx_left += 1

            if len(input_data) < batch_size:
                continue

            batch_full = True

            inputs = torch.stack(input_data).to("cuda:{}".format(args.device))

            outputs = net(inputs)

            num_batches += 1

            # save data to h5
            for idx, frame_number, group, embedding in zip(batch_indexes, batch_frames, batch_groups,
                                                           embedding_vectors[0].cpu().detach().numpy()):
                group["frames"][idx] = frame_number
                group["embeddings"][idx] = embedding

            input_data = []
            batch_groups = []
            batch_indexes = []
            batch_frames = []

        # save right hands
        data_idx_right = 0
        for i, (frame, hand) in enumerate(zip(f_hand_crops[sample]["right_hand"]["frames"],
                                              f_hand_crops[sample]["right_hand"]["images"])):
            # check OpenPose min confidence
            if args.open_pose_h5 is not None:
                target_joints = f_joints[sample][frame]
                target_joints = np.reshape(target_joints, (-1, 3))
                mrh = np.mean(target_joints[29:50, 2]).item()
                # low Open Pose confidence
                if mrh < args.min_conf:
                    continue

            data_sample = val_transform(image=hand)["image"]
            input_data.append(data_sample)
            batch_groups.append(group_right)
            batch_indexes.append(data_idx_right)
            batch_frames.append(frame)
            data_idx_right += 1

            if len(input_data) < batch_size:
                continue

            batch_full = True

            inputs = torch.stack(input_data).to("cuda:{}".format(args.device))

            outputs = net(inputs)

            num_batches += 1

            # save data to h5
            for idx, frame_number, group, embedding in zip(batch_indexes, batch_frames, batch_groups,
                                                           embedding_vectors[0].cpu().detach().numpy()):
                group["frames"][idx] = frame_number
                group["embeddings"][idx] = embedding

            input_data = []
            batch_groups = []
            batch_indexes = []
            batch_frames = []

        group_left["frames"].resize((data_idx_left, ))
        group_left["embeddings"].resize((data_idx_left, embedding_dim))
        group_right["frames"].resize((data_idx_right,))
        group_right["embeddings"].resize((data_idx_right, embedding_dim))

    # check whether last data were processed
    if not batch_full:
        inputs = torch.stack(input_data).to("cuda:{}".format(args.device))
        outputs = net(inputs)

        # save data to h5
        for idx, frame_number, group, embedding in zip(batch_indexes, batch_frames, batch_groups,
                                                       embedding_vectors[0].cpu().detach().numpy()):
            group["frames"][idx] = frame_number
            group["embeddings"][idx] = embedding

    f_out.close()
