import argparse
import h5py
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch
import random
import os


if __name__ == "__main__":
    # parse commandline
    parser = argparse.ArgumentParser(description='Train VLE on ChaLearn-SLR data.')
    parser.add_argument('train_h5', type=str, help='path to dataset with hands')
    parser.add_argument('val_h5', type=str, help='path to dataset with hands')
    parser.add_argument('--max_epoch', type=int, help='number of max epochs', default=10)
    parser.add_argument('--batch_size', type=int, help='number data in one batch', default=32)
    parser.add_argument('--data_to_mem', type=bool, help='load data to memory')
    parser.add_argument('--save_epoch', type=int, help='after how many epoch to save the model', default=1)
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
    random.shuffle(indexes)

    batch_size = args.batch_size

    net = models.resnet18()
    # replace classification layer
    net.fc = nn.Linear(512, 65)
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(args.max_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        net.train()
        for idx in range(0, num_samples, batch_size):
            input_data = []
            input_labels = []
            for data_idx in range(idx, min(idx + batch_size, num_samples)):
                data_sample = data["images"][indexes[data_idx]].swapaxes(2, 0) / 255.0
                label = data["labels"][indexes[data_idx], 0]
                input_data.append(data_sample)
                input_labels.append(label)

            inputs = torch.tensor(input_data, dtype=torch.float, device="cuda:0")
            labels = torch.tensor(input_labels, dtype=torch.long, device="cuda:0")

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if idx % 10 == 9:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, idx + 1, running_loss / 10))
                running_loss = 0.0

        # compute validation loss
        net.eval()
        val_loss = 0
        num_batches = 0
        for idx in range(0, num_val_samples, batch_size):
            idx_max = min(idx + batch_size, num_val_samples)
            data_samples = val_data["images"][idx:idx_max].swapaxes(3, 1) / 255.0
            inputs = torch.tensor(data_samples, dtype=torch.float, device="cuda:0")
            labels = torch.tensor(val_data["labels"][idx:idx_max, 0], dtype=torch.long, device="cuda:0")

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            num_batches += 1

        print("Validation loss Epoch {}/{} = {}".format(epoch, args.max_epoch, val_loss / num_batches))

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
