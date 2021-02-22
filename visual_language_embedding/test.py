import argparse
import h5py
import torchvision.models as models
import torch.nn as nn
import torch
import os
import cv2
from architecture import VLE_01

if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Train VLE on ChaLearn-SLR data.')
    parser.add_argument('test_h5', type=str, help='path to dataset with testing hands')
    parser.add_argument('net', type=str, help='path to network you want to test ')
    parser.add_argument('--max_epoch', type=int, help='number of max epochs', default=10)
    parser.add_argument('--batch_size', type=int, help='number data in one batch', default=32)
    parser.add_argument('--data_to_mem', type=bool, help='load data to memory')
    parser.add_argument('output', type=str, help='path to output')
    args = parser.parse_args()

    test_data = h5py.File(args.test_h5, "r")

    if args.data_to_mem is not None:
        data = {"images": test_data["images"][:], "labels": test_data["labels"][:]}
    else:
        data = test_data

    num_samples = len(data["labels"])

    batch_size = args.batch_size

    net = models.resnet18()
    # replace classification layer
    net.fc = nn.Linear(512, 65)
    # net = VLE_01(65)
    net.cuda()

    start_epoch = 0

    if args.net.endswith(".tar"):
        checkpoint = torch.load(args.init_net)
        net.load_state_dict(checkpoint["model_state_dict"])
    else:
        net.load_state_dict(torch.load(args.net))

    # compute test acc
    net.eval()
    val_loss = 0
    num_batches = 0
    acc = 0.0
    for idx in range(0, num_samples, batch_size):
        idx_max = min(idx + batch_size, num_samples)
        data_samples = data["images"][idx:idx_max].swapaxes(3, 1) / 255.0
        inputs = torch.tensor(data_samples, dtype=torch.float, device="cuda:0")
        labels = torch.tensor(data["labels"][idx:idx_max, 0], dtype=torch.long, device="cuda:0")

        outputs = net(inputs)

        acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
        num_batches += 1

        for i, sample in enumerate(inputs):
            print("sample {}, pred={}, label={} *{}*".format(idx + i, torch.argmax(outputs[i]).item(),
                                                             labels[i].item(),
                                                             (torch.argmax(outputs[i]) == labels[i]).item()))

            os.makedirs("g:/hands_test/{}".format(torch.argmax(outputs[i]).item()), exist_ok=True)
            cv2.imwrite(os.path.join("g:/hands_test/{}".format(torch.argmax(outputs[i]).item()),
                                     "{}_{}.jpg".format(idx + i, labels[i].item())),
                        sample.cpu().numpy().swapaxes(2, 0) * 255)

    print("Test acc: {}".format(acc / num_samples))
