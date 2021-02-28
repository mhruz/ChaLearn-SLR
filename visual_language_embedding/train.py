import argparse
import h5py
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch
import random
import os
from architecture import VLE_01
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


# ====================================================
# Label Smoothing
# ====================================================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        """

        :param classes: number of classes
        :param smoothing: the max likelihood value = 1 - smoothing
        :param dim:
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Train VLE on ChaLearn-SLR data.')
    parser.add_argument('train_h5', type=str, help='path to dataset with hands')
    parser.add_argument('val_h5', type=str, help='path to dataset with hands')
    parser.add_argument('model', type=str, help='resnet-18, mobilenet, vle')
    parser.add_argument('num_classes', type=int, help='number of classes')
    parser.add_argument('--pretrained', type=bool, help='whether to use pretrained model', default=False)
    parser.add_argument('--init_net', type=str, help='path to network you want to start from')
    parser.add_argument('--max_epoch', type=int, help='number of max epochs', default=10)
    parser.add_argument('--batch_size', type=int, help='number data in one batch', default=32)
    parser.add_argument('--data_to_mem', type=bool, help='load data to memory')
    parser.add_argument('--save_epoch', type=int, help='after how many epoch to save the model', default=1)
    parser.add_argument('--resize', type=int, help='resize images to this size')
    parser.add_argument('--device', type=int, help='device number', default=0)
    parser.add_argument('output', type=str, help='path to output network')
    args = parser.parse_args()

    train_data = h5py.File(args.train_h5, "r")
    val_data = h5py.File(args.val_h5, "r")

    if args.data_to_mem is not None:
        data = {"images": train_data["images"][:], "labels": train_data["labels"][:]}
    else:
        data = train_data

    num_samples = len(data["labels"])
    num_val_samples = len(val_data["labels"])

    indexes = list(range(num_samples))

    batch_size = args.batch_size

    if args.model == "resnet-18":
        if args.pretrained:
            net = models.resnet18(pretrained=True)
        else:
            net = models.resnet18(pretrained=False)
        # replace classification layer
        net.fc = nn.Linear(512, args.num_classes)
    elif args.model == "mobilenet":
        if args.pretrained:
            net = models.mobilenet_v2(pretrained=True)
        else:
            net = models.mobilenet_v2(pretrained=False)

        net.classifier[1] = nn.Linear(1280, args.num_classes)

    # net = VLE_01(65)
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingLoss(65, 0.2)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=3e-4)

    start_epoch = 0

    if args.init_net is not None:
        checkpoint = torch.load(args.init_net)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]

    if args.resize is None:
        transform = A.Compose([
            A.ColorJitter(hue=0.05),
            A.HorizontalFlip(),
            A.GaussNoise(var_limit=(5, 15)),
            A.GridDistortion(),
            A.MotionBlur(),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.RGBShift(r_shift_limit=15, b_shift_limit=15, g_shift_limit=15),
            A.RandomResizedCrop(70, 70, scale=(0.85, 1.0)),
            A.Rotate(10),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(args.resize, args.resize),
            A.ColorJitter(hue=0.05),
            A.HorizontalFlip(),
            A.GaussNoise(var_limit=(5, 15)),
            A.GridDistortion(),
            A.MotionBlur(),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.RGBShift(r_shift_limit=15, b_shift_limit=15, g_shift_limit=15),
            A.RandomResizedCrop(args.resize, args.resize, scale=(0.85, 1.0)),
            A.Rotate(10),
            A.Normalize(),
            ToTensorV2()
        ])

    # im = data["images"][9879]
    # cv2.namedWindow("im", 2)
    # cv2.namedWindow("im_aug", 2)
    # cv2.imshow("im", im)
    #
    # for i in range(10):
    #     im2 = transform(image=im)["image"]
    #     cv2.imshow("im_aug", im2)
    #     cv2.waitKey()
    #
    # cv2.destroyAllWindows()

    for epoch in range(start_epoch, args.max_epoch):  # loop over the dataset multiple times

        batch_num = 0
        running_loss = 0.0
        running_acc = 0.0

        random.shuffle(indexes)
        net.train()

        for idx in range(0, num_samples, batch_size):
            input_data = []
            input_labels = []
            for data_idx in range(idx, min(idx + batch_size, num_samples)):
                data_sample = data["images"][indexes[data_idx]]
                data_sample = transform(image=data_sample)["image"]
                label = data["labels"][indexes[data_idx], 0]
                input_data.append(data_sample)
                input_labels.append(label)

            inputs = torch.stack(input_data).to("cuda:{}".format(args.device))
            labels = torch.tensor(input_labels, dtype=torch.long, device="cuda:{}".format(args.device))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            running_acc += (torch.argmax(outputs, dim=1) == labels).sum().item() / labels.shape[0]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if batch_num % 10 == 9:
                print('[%d, %5d] loss: %.3f, acc: %.3f' %
                      (epoch + 1, idx + 1, running_loss / 10, running_acc / 10))
                running_loss = 0.0
                running_acc = 0.0

            batch_num += 1

        # compute validation loss
        net.eval()
        val_loss = 0
        num_batches = 0
        acc = 0.0
        input_data = []
        input_labels = []
        for idx in range(0, num_val_samples, batch_size):
            for data_idx in range(idx, min(idx + batch_size, num_samples)):
                data_sample = val_data["images"][data_idx]
                data_sample = A.Compose([A.Normalize(), ToTensorV2()])(image=data_sample)["image"]
                label = data["labels"][indexes[data_idx], 0]
                input_data.append(data_sample)
                input_labels.append(label)

            inputs = torch.stack(input_data).to("cuda:{}".format(args.device))
            labels = torch.tensor(input_labels, dtype=torch.long, device="cuda:{}".format(args.device))

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            acc += (torch.argmax(outputs, dim=1) == labels).sum().item()
            val_loss += loss.item()
            num_batches += 1

        print("Validation loss Epoch {}/{} = {}".format(epoch, args.max_epoch, val_loss / num_batches))
        print("Validation acc Epoch {}/{} = {}".format(epoch, args.max_epoch, acc / num_val_samples))

        # save the model
        if (epoch + 1) % args.save_epoch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss / num_batches,
            }, os.path.join(args.output) + "epoch_{}.tar".format(epoch))

    print('Finished Training')
    torch.save(net.state_dict(), args.output)
