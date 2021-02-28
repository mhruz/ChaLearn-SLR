import argparse
import h5py
import torchvision.models as models
import torch.nn as nn
import torch
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from architecture import VLE_01

if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Train VLE on ChaLearn-SLR data.')
    parser.add_argument('test_h5', type=str, help='path to dataset with testing hands')
    parser.add_argument('net', type=str, help='path to network you want to test ')
    parser.add_argument('model', type=str, help='resnet-18, mobilenet, vle')
    parser.add_argument('num_classes', type=int, help='number of classes')
    parser.add_argument('--max_epoch', type=int, help='number of max epochs', default=10)
    parser.add_argument('--batch_size', type=int, help='number data in one batch', default=32)
    parser.add_argument('--data_to_mem', type=bool, help='load data to memory')
    parser.add_argument('--resize', type=int, help='resize images to this size')
    parser.add_argument('--device', type=int, help='device number', default=0)
    parser.add_argument('output', type=str, help='path to output')
    args = parser.parse_args()

    test_data = h5py.File(args.test_h5, "r")

    if args.data_to_mem is not None:
        data = {"images": test_data["images"][:], "labels": test_data["labels"][:]}
    else:
        data = test_data

    num_samples = len(data["labels"])

    batch_size = args.batch_size

    if args.model == "resnet-18":
        net = models.resnet18(pretrained=False)
        # replace classification layer
        net.fc = nn.Linear(512, args.num_classes)
    elif args.model == "mobilenet":
        net = models.mobilenet_v2(pretrained=False)

        net.classifier[1] = nn.Linear(1280, args.num_classes)

    # net = VLE_01(65)
    net.cuda()
    start_epoch = 0

    if args.net.endswith(".tar"):
        checkpoint = torch.load(args.init_net)
        net.load_state_dict(checkpoint["model_state_dict"])
    else:
        net.load_state_dict(torch.load(args.net))

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
    val_loss = 0
    num_batches = 0
    acc = 0.0
    for idx in range(0, num_samples, batch_size):
        input_data = []
        input_labels = []
        for data_idx in range(idx, min(idx + batch_size, num_samples)):
            data_sample = data["images"][data_idx]
            data_sample = val_transform(image=data_sample)["image"]
            label = data["labels"][data_idx, 0]
            input_data.append(data_sample)
            input_labels.append(label)

        inputs = torch.stack(input_data).to("cuda:{}".format(args.device))
        labels = torch.tensor(input_labels, dtype=torch.long, device="cuda:{}".format(args.device))

        outputs = net(inputs)

        acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
        num_batches += 1

        for data_idx in range(idx, min(idx + batch_size, num_samples)):
            i = data_idx - idx
            print("sample {}, pred={}, label={} *{}*".format(data_idx, torch.argmax(outputs[i]).item(),
                                                             labels[i].item(),
                                                             (torch.argmax(outputs[i]) == labels[i]).item()))

            os.makedirs("g:/hands_test/{}".format(torch.argmax(outputs[i]).item()), exist_ok=True)
            cv2.imwrite(os.path.join("g:/hands_test/{}".format(torch.argmax(outputs[i]).item()),
                                     "{}_{}.jpg".format(idx + i, labels[i].item())),
                        data["images"][data_idx])

    print("Test acc: {}".format(acc / num_samples))
